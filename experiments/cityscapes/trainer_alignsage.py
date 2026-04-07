import os
import logging
try:
    import wandb
except ImportError:
    class DummyWandB:
        run = None
        def init(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
        def finish(self): pass
    wandb = DummyWandB()
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn.functional as F
import itertools
from tqdm import trange
from experiments.cityscapes.data import Cityscapes
from experiments.cityscapes.models import SegNet, SegNetMtan
from experiments.cityscapes.utils import ConfMatrix, delta_fn, depth_error
from experiments.utils import (common_parser,
                               extract_weight_method_parameters_from_args,
                               get_device, set_logger, set_seed, str2bool,
                               enable_running_stats, disable_running_stats)
from methods.weight_methods import WeightMethods

set_logger()

# Conditional compile decorator for PyTorch < 2.0 compatibility
def compile_if_available(func):
    if hasattr(torch, "compile"):
        return torch.compile(func)
    return func

@compile_if_available
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power of a matrix (orthogonalization).
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

def calculate_similarity(grads_list, cos_fn):
    if len(grads_list) < 2:
        return torch.tensor(1.0, device=grads_list[0].device)

    pairwise_combinations = list(itertools.combinations(range(len(grads_list)), 2))
    all_sims = [cos_fn(grads_list[i], grads_list[j], dim=0) for i, j in pairwise_combinations]
    return sum(all_sims) / len(pairwise_combinations)

def spectral_perturbation_aligned(param, clean_grad, task_grads, rho, lam):
    """
    Alignment-aware spectral perturbation for one layer.

    Returns ε_W = ρ ‖W‖_F [ λ · Q_D  +  (1−λ) · Q ]

    where Q  = NS(Ḡ)   (isotropic, as in vanilla SAGE)
          Q_D = NS(D)   (disagreement-biased)
          D   = element-wise std-dev of {G_k} ⊙ sign(Ḡ)
    """
    g = clean_grad
    if g is None:
        return None

    if g.ndim < 2:
        norm_val = g.norm(2)
        if norm_val > 1e-12:
            return (g / norm_val) * rho
        return torch.zeros_like(g)

    orig_shape = g.shape
    g_2d = g.view(g.size(0), -1) if g.ndim > 2 else g

    transposed = False
    if g_2d.shape[0] < g_2d.shape[1]:
        g_2d = g_2d.T
        transposed = True

    Q = zeropower_via_newtonschulz5(g_2d)

    if lam > 1e-6 and len(task_grads) >= 2:
        stacked = torch.stack(task_grads, dim=0)
        D = stacked.std(dim=0)
        D = D * g.sign()

        D_2d = D.view(D.size(0), -1) if D.ndim > 2 else D
        if transposed:
            D_2d = D_2d.T

        D_norm = D_2d.norm()
        if D_norm > 1e-12:
            Q_D = zeropower_via_newtonschulz5(D_2d)
        else:
            Q_D = Q
    else:
        Q_D = Q
        lam = 0.0

    Q_blend = lam * Q_D + (1.0 - lam) * Q

    if transposed:
        Q_blend = Q_blend.T

    weight_norm = param.norm(2).clamp(min=1e-12)
    return Q_blend.view(orig_shape) * rho * weight_norm

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
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

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
            optimizer.zero_grad()

            train_data, train_label, train_depth = batch
            train_data, train_label = train_data.to(device), train_label.long().to(device)
            train_depth = train_depth.to(device)

            enable_running_stats(model)

            # ==========================================================
            # Phase 1: Forward Pass → per-task & clean gradients
            # ==========================================================
            train_pred, features = model(train_data, return_representation=True)

            losses = [
                calc_loss(train_pred[0], train_label, "semantic"),
                calc_loss(train_pred[1], train_depth, "depth"),
            ]

            n_params = sum(1 for _ in model.parameters())
            task_grads_per_param = [[] for _ in range(n_params)]
            task_grads_flattened = []
            clean_grad_w = [torch.zeros_like(p) for p in model.parameters()]
            loss_clean_val = 0.0

            for i, loss_i in enumerate(losses):
                weight_i = 1.0 / n_tasks
                loss_clean_val += loss_i.item() * weight_i

                retain = (i < n_tasks - 1)
                grad_i = torch.autograd.grad(loss_i, model.parameters(),
                                             retain_graph=retain, allow_unused=True)

                valid_grads = []
                for p, g in zip(model.parameters(), grad_i):
                    if g is not None:
                        valid_grads.append(g.flatten())
                    else:
                        valid_grads.append(torch.zeros_like(p).flatten())
                task_grads_flattened.append(torch.cat(valid_grads))

                for k, g in enumerate(grad_i):
                    if g is not None:
                        task_grads_per_param[k].append(g.detach())
                        clean_grad_w[k] += g.detach() * weight_i

            # ==========================================================
            # Phase 2: Task Similarity & Mixing Coefficient λ
            # ==========================================================
            avg_sim = calculate_similarity(task_grads_flattened, F.cosine_similarity)
            lam = (args.lambda_max * (1.0 - avg_sim)).clamp(0.0, args.lambda_max)
            lam_val = lam.item() if isinstance(lam, torch.Tensor) else lam

            # ==========================================================
            # Phase 3: Alignment-Aware Spectral Perturbations
            # ==========================================================
            eps = []
            params_list = list(model.parameters())
            for j_p, (p, g) in enumerate(zip(params_list, clean_grad_w)):
                if g.norm() < 1e-12:
                    eps.append(None)
                    continue

                eps_w = spectral_perturbation_aligned(
                    param=p,
                    clean_grad=g,
                    task_grads=task_grads_per_param[j_p],
                    rho=args.rho,
                    lam=lam_val,
                )
                eps.append(eps_w)

            # ==========================================================
            # Phase 4: Ascent Step
            # ==========================================================
            with torch.no_grad():
                for p, v in zip(params_list, eps):
                    if v is not None:
                        p.add_(v)

            # ==========================================================
            # Phase 5: Adversarial Forward & Backward Pass
            # ==========================================================
            disable_running_stats(model)
            model.zero_grad()
            optimizer.zero_grad()

            train_pred_pert, features_pert = model(train_data, return_representation=True)
            losses_pert = torch.stack((
                calc_loss(train_pred_pert[0], train_label, "semantic"),
                calc_loss(train_pred_pert[1], train_depth, "depth"),
            ))

            loss_pert, extra_outputs = weight_method.backward(
                losses=losses_pert,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features_pert,
            )

            # ==========================================================
            # Phase 6: Restore Weights
            # ==========================================================
            with torch.no_grad():
                for p, v in zip(params_list, eps):
                    if v is not None:
                        p.sub_(v)

            # ==========================================================
            # Phase 7: Optimizer Step
            # ==========================================================
            optimizer.step()

            loss_list.append(torch.stack(losses).detach().cpu())

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

            sim_val = avg_sim.item() if isinstance(avg_sim, torch.Tensor) else avg_sim

            epoch_iter.set_description(
                f"[{epoch+1}  {j+1}/{train_batch}] semantic loss: {losses[0].item():.3f}, "
                f"depth loss: {losses[1].item():.3f}, "
                f"sim: {sim_val:.3f}, lambda: {lam_val:.3f}")

        scheduler.step()
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        model.eval()
        conf_mat = ConfMatrix(model.segnet.class_nb)
        with torch.no_grad():
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.long().to(device)
                test_depth = test_depth.to(device)

                test_pred = model(test_data)
                test_loss = torch.stack((
                    calc_loss(test_pred[0], test_label, "semantic"),
                    calc_loss(test_pred[1], test_depth, "depth"),
                ))

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[6] = test_loss[0].item()
                cost[9] = test_loss[1].item()
                cost[10], cost[11] = depth_error(test_pred[1], test_depth)
                avg_cost[epoch, 6:] += cost[6:] / test_batch

            avg_cost[epoch, 7:9] = conf_mat.get_metrics()

            test_delta_m = delta_fn(avg_cost[epoch, [7, 8, 10, 11]])
            deltas[epoch] = test_delta_m

            print(
                f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR | ∆m (test)"
            )
            print(
                f"Epoch: {epoch:04d} | TRAIN: {avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} "
                f"| {avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} | "
                f"TEST: {avg_cost[epoch, 6]:.4f} {avg_cost[epoch, 7]:.4f} {avg_cost[epoch, 8]:.4f} | "
                f"{avg_cost[epoch, 9]:.4f} {avg_cost[epoch, 10]:.4f} {avg_cost[epoch, 11]:.4f} "
                f"| {test_delta_m:.3f}")

            if wandb.run is not None:
                wandb.log({"Train Gradient Similarity": sim_val}, step=epoch)
                wandb.log({"Train Lambda Mix": lam_val}, step=epoch)

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

            name = f"{args.method}_rho{args.rho}_lmax{args.lambda_max}_sd{args.seed}_alignsage"

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
    # AlignmentAwareSAGE settings
    parser.add_argument("--rho",
                        type=float,
                        default=0.003,
                        help="Rho for perturbation radius.")
    parser.add_argument("--lambda_max",
                        type=float,
                        default=0.8,
                        help="Maximum mixing weight toward disagreement perturbation.")

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
