import logging
import os
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
from experiments.nyuv2.data import NYUv2
from experiments.nyuv2.models import SegNet, SegNetMtan
from experiments.nyuv2.utils import delta_fn, depth_error, normal_error
from experiments.nyuv2.utils import ConfMatrix
from experiments.utils import (common_parser,
                               extract_weight_method_parameters_from_args,
                               get_device, set_logger, set_seed, str2bool)
from methods.weight_methods import WeightMethods

set_logger()

# Methods whose get_weighted_loss returns a differentiable scalar without
# calling .backward() or overwriting .grad — safe to compose with DECOR.
DECOR_COMPATIBLE_METHODS = {
    "ls", "uw", "rlw", "dwa", "stl", "scaleinvls",
    "famo", "mgda", "log_mgda", "nashmtl", "imtl", "log_imtl",
}


def calc_loss(x_pred, x_output, task_type):
    device = x_pred.device
    binary_mask = (torch.sum(x_output, dim=1) != 0).float().unsqueeze(1).to(device)

    if task_type == "semantic":
        loss = F.nll_loss(x_pred, x_output, ignore_index=-1)
    if task_type == "depth":
        loss = torch.sum(
            torch.abs(x_pred - x_output) * binary_mask) / torch.nonzero(
                binary_mask, as_tuple=False).size(0)
    if task_type == "normal":
        loss = 1 - torch.sum(
            (x_pred * x_output) * binary_mask) / torch.nonzero(
                binary_mask, as_tuple=False).size(0)
    return loss


def decor_step(model, optimizer, weight_method, losses, features,
               lambda_c, lambda_a, fisher_eps):
    """
    DECOR training step using the K×K Gram matrix of per-task gradients.

    Under the empirical-Fisher approximation H_bar ≈ F:
      curvature term : tr(H_bar)             ≈ (1/K) * tr(G^T G) = diag(M).mean()
      alignment term : tr(H_bar^{-1} Sigma_g) via Woodbury on the K×K Gram

    ERM is weighted by weight_method instead of a plain mean, so any
    compatible scalar-weighting strategy (ls, uw, dwa, famo, mgda, …) can
    be tested on top of the DECOR regularisation terms.
    """
    K = len(losses)
    params = [p for p in model.parameters() if p.requires_grad]

    # Per-task gradients, differentiable (create_graph=True for 2nd-order backward)
    G_cols = []
    for loss_k in losses:
        grads_k = torch.autograd.grad(
            loss_k, params,
            create_graph=True, retain_graph=True, allow_unused=True,
        )
        grads_k = [g if g is not None else torch.zeros_like(p)
                   for g, p in zip(grads_k, params)]
        G_cols.append(torch.cat([g.reshape(-1) for g in grads_k]))

    # K×K Gram matrix, differentiable in theta
    M = torch.stack([
        torch.stack([torch.dot(G_cols[k], G_cols[l]) for l in range(K)])
        for k in range(K)
    ])

    # Curvature term: (1/K) * tr(G^T G) = diag(M).mean()
    curv_term = torch.diagonal(M).mean()

    # Alignment term via Woodbury; alpha is relative to M's scale
    M_scale = torch.diagonal(M).mean().detach()
    alpha = fisher_eps * M_scale  # fisher_eps is a multiplier, not an absolute shift

    eye_K = torch.eye(K, device=M.device, dtype=M.dtype)
    C = eye_K - torch.ones(K, K, device=M.device, dtype=M.dtype) / K
    CMC = C @ M @ C
    MC = M @ C
    A = alpha * K * eye_K + M
    sol = torch.linalg.solve(A, MC)
    align_term = (torch.trace(CMC) - torch.trace(C @ M @ sol)) / (alpha * K)

    # Weighted ERM via the chosen weight method (graph is still alive via retain_graph=True)
    losses_tensor = torch.stack(losses)
    weighted_erm, _ = weight_method.get_weighted_loss(
        losses_tensor,
        shared_parameters=list(model.shared_parameters()),
        task_specific_parameters=list(model.task_specific_parameters()),
        last_shared_parameters=list(model.last_shared_parameters()),
        representation=features,
    )

    total_loss = weighted_erm + lambda_c * curv_term + lambda_a * align_term

    # Diagnostics (detached)
    with torch.no_grad():
        eigvals_M = torch.linalg.eigvalsh(M)
        avg_sim = 1.0
        per_task_sims = {}
        if K > 1:
            norms = torch.sqrt(torch.diagonal(M).clamp_min(1e-12))
            cos_mat = M / (norms.unsqueeze(0) * norms.unsqueeze(1))
            pairs = list(itertools.combinations(range(K), 2))
            avg_sim = torch.stack([cos_mat[i, j] for i, j in pairs]).mean().item()
            for i in range(K):
                sims_i = [cos_mat[i, j].item() for j in range(K) if j != i]
                per_task_sims[f"gradient_sim_task_{i}"] = sum(sims_i) / len(sims_i)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {
        "erm_loss": weighted_erm.item(),
        "total_loss": total_loss.item(),
        "curv_term": curv_term.item(),
        "align_term": align_term.item(),
        "align_minus_trivial": align_term.item() - (K - 1),
        "M_scale": M_scale.item(),
        "M_min_eig": eigvals_M.min().item(),
        "M_max_eig": eigvals_M.max().item(),
        "alpha_eff": alpha.item(),
        "gradient_similarity": avg_sim,
        **per_task_sims,
    }


def main(args, device):
    if args.method not in DECOR_COMPATIBLE_METHODS:
        raise ValueError(
            f"Method '{args.method}' is not compatible with DECOR (it calls "
            f".backward() or overwrites .grad inside get_weighted_loss). "
            f"Compatible methods: {sorted(DECOR_COMPATIBLE_METHODS)}"
        )

    model = dict(segnet=SegNet(), mtan=SegNetMtan())[args.model]
    model = model.to(device)

    log_str = ("Applying data augmentation on NYUv2."
               if args.apply_augmentation else
               "Standard training strategy without data augmentation.")
    logging.info(log_str)

    nyuv2_train_set = NYUv2(root=args.data_path.as_posix(), train=True,
                            augmentation=args.apply_augmentation)
    nyuv2_test_set = NYUv2(root=args.data_path.as_posix(), train=False)

    train_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset=nyuv2_test_set, batch_size=args.batch_size, shuffle=False)

    n_tasks = 3
    weight_methods_parameters = extract_weight_method_parameters_from_args(args)
    weight_method = WeightMethods(
        args.method, n_tasks=n_tasks, device=device,
        **weight_methods_parameters[args.method],
    )

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=args.lr),
        dict(params=weight_method.parameters(), lr=args.method_params_lr),
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    epochs = args.n_epochs
    epoch_iter = trange(epochs)
    train_batch = len(train_loader)
    test_batch = len(test_loader)
    avg_cost = np.zeros([epochs, 24], dtype=np.float32)
    conf_mat = ConfMatrix(model.segnet.class_nb)
    deltas = np.zeros([epochs], dtype=np.float32)
    loss_list = []

    for epoch in epoch_iter:
        cost = np.zeros(24, dtype=np.float32)
        epoch_stats = {k: 0.0 for k in ["erm_loss", "total_loss", "curv_term",
                                         "align_term", "gradient_similarity"]}

        for j, batch in enumerate(train_loader):
            model.train()

            train_data, train_label, train_depth, train_normal = batch
            train_data = train_data.to(device)
            train_label = train_label.long().to(device)
            train_depth = train_depth.to(device)
            train_normal = train_normal.to(device)

            train_pred, features = model(train_data, return_representation=True)

            losses = [
                calc_loss(train_pred[0], train_label, "semantic"),
                calc_loss(train_pred[1], train_depth, "depth"),
                calc_loss(train_pred[2], train_normal, "normal"),
            ]

            stats = decor_step(
                model, optimizer, weight_method, losses, features,
                args.lambda_c, args.lambda_a, args.fisher_eps,
            )

            # FAMO requires a second forward pass after the optimizer step
            # to update its internal loss bookkeeping.
            if "famo" in args.method:
                with torch.no_grad():
                    train_pred_famo = model(train_data, return_representation=False)
                    new_losses = torch.stack((
                        calc_loss(train_pred_famo[0], train_label, "semantic"),
                        calc_loss(train_pred_famo[1], train_depth, "depth"),
                        calc_loss(train_pred_famo[2], train_normal, "normal"),
                    ))
                    weight_method.method.update(new_losses.detach())

            for k in epoch_stats:
                epoch_stats[k] += stats[k] / train_batch

            loss_list.append(torch.tensor([l.item() for l in losses]))

            conf_mat.update(train_pred[0].argmax(1).flatten(), train_label.flatten())

            cost[0] = losses[0].item()
            cost[3] = losses[1].item()
            cost[4], cost[5] = depth_error(train_pred[1], train_depth)
            cost[6] = losses[2].item()
            cost[7], cost[8], cost[9], cost[10], cost[11] = normal_error(
                train_pred[2], train_normal)
            avg_cost[epoch, :12] += cost[:12] / train_batch

            epoch_iter.set_description(
                f"[{epoch+1}  {j+1}/{train_batch}] sem: {losses[0].item():.3f}, "
                f"dep: {losses[1].item():.3f}, nor: {losses[2].item():.3f}, "
                f"curv: {stats['curv_term']:.4f}, align: {stats['align_term']:.4f}, "
                f"sim: {stats['gradient_similarity']:.3f}")

        scheduler.step()
        avg_cost[epoch, 1:3] = conf_mat.get_metrics()

        model.eval()
        conf_mat = ConfMatrix(model.segnet.class_nb)
        with torch.no_grad():
            test_dataset = iter(test_loader)
            for k in range(test_batch):
                test_data, test_label, test_depth, test_normal = test_dataset.next()
                test_data, test_label = test_data.to(device), test_label.long().to(device)
                test_depth, test_normal = test_depth.to(device), test_normal.to(device)

                test_pred = model(test_data)
                test_loss = torch.stack((
                    calc_loss(test_pred[0], test_label, "semantic"),
                    calc_loss(test_pred[1], test_depth, "depth"),
                    calc_loss(test_pred[2], test_normal, "normal"),
                ))

                conf_mat.update(test_pred[0].argmax(1).flatten(), test_label.flatten())

                cost[12] = test_loss[0].item()
                cost[15] = test_loss[1].item()
                cost[16], cost[17] = depth_error(test_pred[1], test_depth)
                cost[18] = test_loss[2].item()
                cost[19], cost[20], cost[21], cost[22], cost[23] = normal_error(
                    test_pred[2], test_normal)
                avg_cost[epoch, 12:] += cost[12:] / test_batch

            avg_cost[epoch, 13:15] = conf_mat.get_metrics()

            test_delta_m = delta_fn(avg_cost[epoch, [13, 14, 16, 17, 19, 20, 21, 22, 23]])
            deltas[epoch] = test_delta_m

            print(
                f"LOSS FORMAT: SEMANTIC_LOSS MEAN_IOU PIX_ACC | DEPTH_LOSS ABS_ERR REL_ERR "
                f"| NORMAL_LOSS MEAN MED <11.25 <22.5 <30 | ∆m (test)")
            print(
                f"Epoch: {epoch:04d} | TRAIN: {avg_cost[epoch, 0]:.4f} {avg_cost[epoch, 1]:.4f} {avg_cost[epoch, 2]:.4f} "
                f"| {avg_cost[epoch, 3]:.4f} {avg_cost[epoch, 4]:.4f} {avg_cost[epoch, 5]:.4f} | {avg_cost[epoch, 6]:.4f} "
                f"{avg_cost[epoch, 7]:.4f} {avg_cost[epoch, 8]:.4f} {avg_cost[epoch, 9]:.4f} {avg_cost[epoch, 10]:.4f} {avg_cost[epoch, 11]:.4f} || "
                f"TEST: {avg_cost[epoch, 12]:.4f} {avg_cost[epoch, 13]:.4f} {avg_cost[epoch, 14]:.4f} | "
                f"{avg_cost[epoch, 15]:.4f} {avg_cost[epoch, 16]:.4f} {avg_cost[epoch, 17]:.4f} | {avg_cost[epoch, 18]:.4f} "
                f"{avg_cost[epoch, 19]:.4f} {avg_cost[epoch, 20]:.4f} {avg_cost[epoch, 21]:.4f} {avg_cost[epoch, 22]:.4f} {avg_cost[epoch, 23]:.4f} "
                f"| {test_delta_m:.3f}")

            if wandb.run is not None:
                wandb.log({"Train ERM Loss": epoch_stats["erm_loss"]}, step=epoch)
                wandb.log({"Train Total Loss": epoch_stats["total_loss"]}, step=epoch)
                wandb.log({"Train DECOR Curvature": epoch_stats["curv_term"]}, step=epoch)
                wandb.log({"Train DECOR Alignment": epoch_stats["align_term"]}, step=epoch)
                wandb.log({"Train Gradient Similarity": epoch_stats["gradient_similarity"]}, step=epoch)

                wandb.log({"Train Semantic Loss": avg_cost[epoch, 0]}, step=epoch)
                wandb.log({"Train Mean IoU": avg_cost[epoch, 1]}, step=epoch)
                wandb.log({"Train Pixel Accuracy": avg_cost[epoch, 2]}, step=epoch)
                wandb.log({"Train Depth Loss": avg_cost[epoch, 3]}, step=epoch)
                wandb.log({"Train Absolute Error": avg_cost[epoch, 4]}, step=epoch)
                wandb.log({"Train Relative Error": avg_cost[epoch, 5]}, step=epoch)
                wandb.log({"Train Normal Loss": avg_cost[epoch, 6]}, step=epoch)
                wandb.log({"Train Loss Mean": avg_cost[epoch, 7]}, step=epoch)
                wandb.log({"Train Loss Med": avg_cost[epoch, 8]}, step=epoch)
                wandb.log({"Train Loss <11.25": avg_cost[epoch, 9]}, step=epoch)
                wandb.log({"Train Loss <22.5": avg_cost[epoch, 10]}, step=epoch)
                wandb.log({"Train Loss <30": avg_cost[epoch, 11]}, step=epoch)

                wandb.log({"Test Semantic Loss": avg_cost[epoch, 12]}, step=epoch)
                wandb.log({"Test Mean IoU": avg_cost[epoch, 13]}, step=epoch)
                wandb.log({"Test Pixel Accuracy": avg_cost[epoch, 14]}, step=epoch)
                wandb.log({"Test Depth Loss": avg_cost[epoch, 15]}, step=epoch)
                wandb.log({"Test Absolute Error": avg_cost[epoch, 16]}, step=epoch)
                wandb.log({"Test Relative Error": avg_cost[epoch, 17]}, step=epoch)
                wandb.log({"Test Normal Loss": avg_cost[epoch, 18]}, step=epoch)
                wandb.log({"Test Loss Mean": avg_cost[epoch, 19]}, step=epoch)
                wandb.log({"Test Loss Med": avg_cost[epoch, 20]}, step=epoch)
                wandb.log({"Test Loss <11.25": avg_cost[epoch, 21]}, step=epoch)
                wandb.log({"Test Loss <22.5": avg_cost[epoch, 22]}, step=epoch)
                wandb.log({"Test Loss <30": avg_cost[epoch, 23]}, step=epoch)
                wandb.log({"Test ∆m": test_delta_m}, step=epoch)

            keys = [
                "Train Semantic Loss", "Train Mean IoU", "Train Pixel Accuracy",
                "Train Depth Loss", "Train Absolute Error", "Train Relative Error",
                "Train Normal Loss", "Train Loss Mean", "Train Loss Med",
                "Train Loss <11.25", "Train Loss <22.5", "Train Loss <30",
                "Test Semantic Loss", "Test Mean IoU", "Test Pixel Accuracy",
                "Test Depth Loss", "Test Absolute Error", "Test Relative Error",
                "Test Normal Loss", "Test Loss Mean", "Test Loss Med",
                "Test Loss <11.25", "Test Loss <22.5", "Test Loss <30",
            ]

            name = f"{args.method}_decor_lc{args.lambda_c}_la{args.lambda_a}_sd{args.seed}"
            torch.save(
                {
                    "delta_m": deltas,
                    "keys": keys,
                    "avg_cost": avg_cost,
                    "losses": loss_list,
                }, f"./save/{name}.stats")


if __name__ == "__main__":
    parser = ArgumentParser("NYUv2 DECOR", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=1e-4,
        n_epochs=200,
        batch_size=2,
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mtan",
        choices=["segnet", "mtan"],
        help="model type",
    )
    parser.add_argument("--apply-augmentation", type=str2bool, default=True,
                        help="data augmentations")
    parser.add_argument("--lambda_c", type=float, default=0.01,
                        help="Curvature term weight (lambda_c in DECOR).")
    parser.add_argument("--lambda_a", type=float, default=1.0,
                        help="Alignment term weight (lambda_a in DECOR).")
    parser.add_argument("--fisher_eps", type=float, default=1e-3,
                        help="Tikhonov shift multiplier for empirical-Fisher inverse.")
    parser.add_argument("--wandb_project", type=str, default=None,
                        help="Name of Weights & Biases Project.")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="Name of Weights & Biases Entity.")

    args = parser.parse_args()
    set_seed(args.seed)

    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=args)

    device = get_device(gpus=args.gpu)
    main(args=args, device=device)

    if wandb.run is not None:
        wandb.finish()
