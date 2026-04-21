import os
import logging
import wandb
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from experiments.cityscapes.data import Cityscapes
from experiments.cityscapes.models import SegNet, SegNetMtan
from experiments.cityscapes.utils import delta_fn, depth_error
from experiments.cityscapes.utils import ConfMatrix, GradEstimator
from experiments.utils import (common_parser,
                               extract_weight_method_parameters_from_args,
                               get_device, set_logger, set_seed, str2bool,
                               enable_running_stats, disable_running_stats)
from methods.weight_methods import WeightMethods
from collections import defaultdict

set_logger()


def compile_if_available(func):
    if hasattr(torch, "compile"):
        return torch.compile(func)
    return func


@compile_if_available
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of a matrix.
    Source: https://github.com/KellerJordan/Muon
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(0) > G.size(1):
        X = X.T
    return X.type_as(G)


def muon_perturbation(g, p, rho):
    """Compute Muon-style orthogonalized perturbation with ASAM weight-norm scaling."""
    if g.norm() < 1e-12:
        return None
    if g.ndim >= 2:
        orig_shape = g.shape
        view_2d = g.view(g.size(0), -1)
        g_ortho = zeropower_via_newtonschulz5(view_2d, steps=5)
        weight_norm = p.norm(2).clamp(min=1e-12)
        return g_ortho.view(orig_shape) * rho * weight_norm
    else:
        norm_val = g.norm(2)
        if norm_val > 1e-12:
            return (g / norm_val) * rho
        return None


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device

    binary_mask = (torch.sum(x_output, dim=1)
                   != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)

    if task_type == "depth":
        loss = torch.sum(
            torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
                binary_mask, as_tuple=False).size(0)

    return loss


def main(args, device):
    model = dict(segnet=SegNet(), mtan=SegNetMtan())[args.model]
    model = model.to(device)

    log_str = ("Applying data augmentation on Cityscapes."
               if args.apply_augmentation else
               "Standard training strategy without data augmentation.")
    logging.info(log_str)

    cityscapes_train_set = Cityscapes(root=args.data_path.as_posix(),
                                      train=True,
                                      augmentation=args.apply_augmentation)
    cityscapes_test_set = Cityscapes(root=args.data_path.as_posix(),
                                     train=False)

    train_loader = torch.utils.data.DataLoader(dataset=cityscapes_train_set,
                                               batch_size=args.batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=cityscapes_test_set,
                                              batch_size=args.batch_size,
                                              shuffle=False)

    n_tasks = 2
    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    weight_method = WeightMethods(args.method,
                                  n_tasks=n_tasks,
                                  device=device,
                                  **weight_methods_parameters[args.method])

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.lr),
        dict(params=weight_method.parameters(), lr=args.method_params_lr),
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=100,
                                                gamma=0.5)

    epochs = args.n_epochs
    epoch_iter = trange(epochs)
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([epochs, 12], dtype=np.float32)
    custom_step = -1
    conf_mat = ConfMatrix(model.segnet.class_nb)
    deltas = np.zeros([epochs], dtype=np.float32)
    loss_list = []

    for epoch in epoch_iter:
        cost = np.zeros(12, dtype=np.float32)

        for j, batch in enumerate(train_loader):
            custom_step += 1

            model.train()

            train_data, train_label, train_depth = batch
            train_data, train_label = train_data.to(
                device), train_label.long().to(device)
            train_depth = train_depth.to(device)

            enable_running_stats(model)
            train_pred, features = model(train_data, return_representation=True)

            losses = torch.stack((
                calc_loss(train_pred[0], train_label, "semantic"),
                calc_loss(train_pred[1], train_depth, "depth"),
            ))
            losses.mean().backward()

            # Zeroth-order gradient estimation per task
            zeroth_grads = {}
            for task in range(n_tasks):
                targets = train_label if task == 0 else train_depth
                zeroth_grads[task] = GradEstimator(
                    model, eps=args.zo_eps).forward(train_data, targets, task)

            ############################## Stage I: Compute Muon perturbations ##############################
            shared_params = dict()
            for n, p in model.named_parameters():
                if "pred" not in n:
                    shared_params[n] = p.data.clone()
            task_params = defaultdict(dict)

            shared_epsilon_params = defaultdict(dict)
            task_epsilon_params = defaultdict(dict)

            for task in range(n_tasks):
                # Task-specific perturbation
                for n, p in model.named_parameters():
                    if f"pred_task{task + 1}" in n:
                        task_ep = torch.zeros_like(p).data.clone()
                        if p.grad is not None:
                            task_params[task][n] = p.data.clone()
                            ep = muon_perturbation(p.grad.data.clone(), p, args.rho)
                            if ep is not None:
                                task_ep = ep
                        task_epsilon_params[task][n] = task_ep

                # Shared perturbation — blend mean grad with zeroth-order grad (SAMO-style)
                for n, p in model.named_parameters():
                    if "pred" in n:
                        continue
                    shared_ep = torch.zeros_like(p).data
                    if p.grad is not None:
                        n1 = p.grad.norm(p=2).item()
                        n2 = zeroth_grads[task][n].norm(p=2).item()
                        g = (1 - args.beta) * p.grad.data.clone(
                        ) + args.beta * n1 * (zeroth_grads[task][n] / (n2 + 1e-12))
                        ep = muon_perturbation(g, p, args.rho)
                        if ep is not None:
                            shared_ep = ep
                    shared_epsilon_params[task][n] = shared_ep

            ############################## Stage II: Perturbed forward/backward ##############################
            disable_running_stats(model)
            shared_sam_grad = defaultdict(dict)
            model.zero_grad()

            for task in range(n_tasks):
                for n, p in model.named_parameters():
                    if "pred" in n:
                        if f"pred_task{task + 1}" in n:
                            p.data = (task_params[task][n] +
                                      task_epsilon_params[task][n]).data.clone()
                    else:
                        if p.grad is not None:
                            p.grad.zero_()
                        p.data = (shared_params[n] +
                                  shared_epsilon_params[task][n]).data.clone()

                train_pred_pert, _ = model(train_data, return_representation=True)

                if task == 0:
                    calc_loss(train_pred_pert[0], train_label, "semantic").backward()
                elif task == 1:
                    calc_loss(train_pred_pert[1], train_depth, "depth").backward()
                else:
                    raise ValueError(f"Task {task} not supported")

                for n, p in model.named_parameters():
                    if "pred" in n:
                        if f"pred_task{task + 1}" in n:
                            p.data = task_params[task][n].data.clone()
                        continue
                    if p.grad is not None:
                        shared_sam_grad[task][n] = p.grad.data.clone()
                        p.grad.zero_()

            del task_epsilon_params, shared_epsilon_params

            # Restore shared parameters
            for n, p in model.named_parameters():
                if "pred" not in n:
                    p.data = shared_params[n]

            loss, extra_outputs = weight_method.backward(
                losses=None,
                shared_grads=shared_sam_grad,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features,
            )

            loss_list.append(losses.detach().cpu())
            optimizer.step()

            if "famo" in args.method:
                with torch.no_grad():
                    train_pred_famo = model(train_data, return_representation=False)
                    new_losses = torch.stack((
                        calc_loss(train_pred_famo[0], train_label, "semantic"),
                        calc_loss(train_pred_famo[1], train_depth, "depth"),
                    ))
                    weight_method.method.update(new_losses.detach())

            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            avg_cost[epoch, :6] += cost[:6] / train_batch

            epoch_iter.set_description(
                f"[{epoch+1}  {j+1}/{train_batch}] semantic loss: {losses[0].item():.3f}, "
                f"depth loss: {losses[1].item():.3f}")

        scheduler.step()
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        model.eval()
        conf_mat = ConfMatrix(model.segnet.class_nb)
        with torch.no_grad():
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth = test_dataset.next()
                test_data, test_label = test_data.to(
                    device), test_label.long().to(device)
                test_depth = test_depth.to(device)

                test_pred = model(test_data)
                test_loss = torch.stack((
                    calc_loss(test_pred[0], test_label, "semantic"),
                    calc_loss(test_pred[1], test_depth, "depth"),
                ))

                conf_mat.update(test_pred[0].argmax(1).flatten(),
                                test_label.flatten())

                cost[6] = test_loss[0].item()
                cost[9] = test_loss[1].item()
                cost[10], cost[11] = depth_error(test_pred[1], test_depth)
                avg_cost[epoch, 6:] += cost[6:] / test_batch

            avg_cost[epoch, 7:9] = conf_mat.get_metrics()

            test_delta_m = delta_fn(avg_cost[epoch, [7, 8, 10, 11]])
            deltas[epoch] = test_delta_m

            print(
                f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR "
            )
            print(
                f"Epoch: {epoch:04d} | TRAIN: {avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} "
                f"| {avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} | {avg_cost[epoch, 6]:.4f} "
                f"TEST: {avg_cost[epoch, 7]:.4f} {avg_cost[epoch, 8]:.4f} {avg_cost[epoch, 9]:.4f} | "
                f"{avg_cost[epoch, 10]:.4f} {avg_cost[epoch, 11]:.4f}"
                f"| {test_delta_m:.3f}")

            if wandb.run is not None:
                wandb.log({"Train Semantic Loss": avg_cost[epoch, 0]}, step=epoch)
                wandb.log({"Train Mean IoU": avg_cost[epoch, 1]}, step=epoch)
                wandb.log({"Train Pixel Accuracy": avg_cost[epoch, 2]}, step=epoch)
                wandb.log({"Train Depth Loss": avg_cost[epoch, 3]}, step=epoch)
                wandb.log({"Train Absolute Error": avg_cost[epoch, 4]}, step=epoch)
                wandb.log({"Train Relative Error": avg_cost[epoch, 5]}, step=epoch)
                wandb.log({"Test Semantic Loss": avg_cost[epoch, 6]}, step=epoch)
                wandb.log({"Test Mean IoU": avg_cost[epoch, 7]}, step=epoch)
                wandb.log({"Test Pixel Accuracy": avg_cost[epoch, 8]}, step=epoch)
                wandb.log({"Test Depth Loss": avg_cost[epoch, 9]}, step=epoch)
                wandb.log({"Test Absolute Error": avg_cost[epoch, 10]}, step=epoch)
                wandb.log({"Test Relative Error": avg_cost[epoch, 11]}, step=epoch)
                wandb.log({"Test ∆m": test_delta_m}, step=epoch)

            keys = [
                "Train Semantic Loss", "Train Mean IoU", "Train Pixel Accuracy",
                "Train Depth Loss", "Train Absolute Error", "Train Relative Error",
                "Test Semantic Loss", "Test Mean IoU", "Test Pixel Accuracy",
                "Test Depth Loss", "Test Absolute Error", "Test Relative Error",
            ]
            name = f"{args.method}_rho{args.rho}_beta{args.beta}_sd{args.seed}_muonsamo"

            torch.save(
                {
                    "delta_m": deltas,
                    "keys": keys,
                    "avg_cost": avg_cost,
                    "losses": loss_list,
                }, f"./save/{name}.stats")


if __name__ == "__main__":
    parser = ArgumentParser("Cityscapes", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=1e-4,
        n_epochs=200,
        batch_size=8,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mtan",
        choices=["segnet", "mtan"],
        help="model type",
    )
    parser.add_argument("--apply-augmentation",
                        type=str2bool,
                        default=True,
                        help="data augmentations")
    parser.add_argument("--rho",
                        type=float,
                        default=0.003,
                        help="Rho for perturbation radius in MuonSAMO.")
    parser.add_argument(
        "--beta",
        default=0.1,
        type=float,
        help="Interpolation coefficient for zeroth-order gradient blend.")
    parser.add_argument("--zo_eps",
                        default=0.01,
                        type=float,
                        help="Epsilon for zeroth-order gradient estimation.")
    parser.add_argument("--wandb_project",
                        type=str,
                        default=None,
                        help="Name of Weights & Biases Project.")
    parser.add_argument("--wandb_entity",
                        type=str,
                        default=None,
                        help="Name of Weights & Biases Entity.")
    args = parser.parse_args()

    set_seed(args.seed)

    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity,
                   config=args)

    device = get_device(gpus=args.gpu)
    main(args=args, device=device)

    if wandb.run is not None:
        wandb.finish()
