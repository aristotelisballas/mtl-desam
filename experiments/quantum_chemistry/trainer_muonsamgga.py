from argparse import ArgumentParser
import os
import itertools

import numpy as np
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.datasets import QM9
from torch_geometric.loader import DataLoader
from tqdm import trange
try:
    import wandb
except ImportError:
    class DummyWandB:
        run = None
        def init(self, *args, **kwargs): pass
        def log(self, *args, **kwargs): pass
        def finish(self): pass
    wandb = DummyWandB()

from experiments.quantum_chemistry.models import Net
from experiments.quantum_chemistry.utils import (
    Complete,
    MyTransform,
    delta_fn,
    multiply_indx,
)
from experiments.quantum_chemistry.utils import target_idx as targets
from experiments.utils import (
    common_parser,
    extract_weight_method_parameters_from_args,
    get_device,
    set_logger,
    set_seed,
    str2bool,
    enable_running_stats,
    disable_running_stats,
)
from methods.weight_methods import WeightMethods

set_logger()

# Conditional compile decorator for PyTorch < 2.0 compatibility
def compile_if_available(func):
    if hasattr(torch, "compile"):
        return torch.compile(func)
    return func

# Helper function for Muon-style orthogonalization
@compile_if_available
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power of a matrix (orthogonalization).
    Source: https://github.com/KellerJordan/Muon
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16() / (G.norm() + eps)  # Ensure stability
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
    """
    Compute the average cosine similarity between domain/task gradients.
    """
    if len(grads_list) < 2:
        return torch.tensor(1.0, device=grads_list[0].device)

    pairwise_combinations = list(itertools.combinations(range(len(grads_list)), 2))
    all_sims = [cos_fn(grads_list[i], grads_list[j], dim=0) for i, j in pairwise_combinations]
    avg_sim = sum(all_sims) / len(pairwise_combinations)
    return avg_sim


@torch.no_grad()
def evaluate(model, loader, std, scale_target, device):
    model.eval()
    data_size = 0.0
    task_losses = 0.0
    for i, data in enumerate(loader):
        data = data.to(device)
        out = model(data)
        if scale_target:
            task_losses += F.l1_loss(out * std.to(device),
                                     data.y * std.to(device),
                                     reduction="none").sum(0)  # MAE
        else:
            task_losses += F.l1_loss(out, data.y,
                                     reduction="none").sum(0)  # MAE
        data_size += len(data.y)

    model.train()

    avg_task_losses = task_losses / data_size

    # Report meV instead of eV.
    avg_task_losses = avg_task_losses.detach().cpu().numpy()
    avg_task_losses[multiply_indx] *= 1000

    delta_m = delta_fn(avg_task_losses)
    return dict(
        avg_loss=avg_task_losses.mean(),
        avg_task_losses=avg_task_losses,
        delta_m=delta_m,
    )


def main(
    data_path: str,
    batch_size: int,
    device: torch.device,
    method: str,
    weight_method_params: dict,
    lr: float,
    method_params_lr: float,
    n_epochs: int,
    args,
    targets: list = None,
    scale_target: bool = True,
    main_task: int = None,
):
    dim = 64
    model = Net(n_tasks=len(targets), num_features=11, dim=dim).to(device)

    transform = T.Compose(
        [MyTransform(targets),
         Complete(), T.Distance(norm=False)])
    dataset = QM9(data_path, transform=transform).shuffle()

    # Split datasets.
    test_dataset = dataset[:10000]
    val_dataset = dataset[10000:20000]
    train_dataset = dataset[20000:]

    std = None
    if scale_target:
        mean = train_dataset.data.y[:, targets].mean(dim=0, keepdim=True)
        std = train_dataset.data.y[:, targets].std(dim=0, keepdim=True)

        dataset.data.y[:, targets] = (dataset.data.y[:, targets] - mean) / std

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=0)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0)
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0)

    n_tasks = len(targets)
    weight_method = WeightMethods(
        method,
        n_tasks=n_tasks,
        device=device,
        **weight_method_params[method],
    )

    optimizer = torch.optim.Adam([
        dict(params=model.parameters(), lr=lr),
        dict(params=weight_method.parameters(), lr=method_params_lr),
    ], )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="min",
                                                           factor=0.7,
                                                           patience=5,
                                                           min_lr=0.00001)

    epoch_iterator = trange(n_epochs)

    best_val = np.inf
    best_test = np.inf
    best_test_delta = np.inf
    best_val_delta = np.inf
    best_test_results = None

    avg_cost = np.zeros([n_epochs, 13 * 2], dtype=np.float32)
    deltas = np.zeros([n_epochs], dtype=np.float32)

    # some extra statistics we save during training
    loss_list = []

    for epoch in epoch_iterator:
        lr_current = scheduler.optimizer.param_groups[0]["lr"]
        for j, data in enumerate(train_loader):
            model.train()
            data = data.to(device)
            optimizer.zero_grad()

            enable_running_stats(model)

            # ==========================================================
            # Phase 1: Single Forward Pass for Task & Clean Gradients
            # ==========================================================
            out, features = model(data, return_representation=True)

            losses = F.mse_loss(out, data.y, reduction="none").mean(0)
            
            task_grads_flattened = []
            clean_grad_w = [torch.zeros_like(p) for p in model.parameters()]
            loss_clean_val = 0.0

            for i in range(n_tasks):
                loss_i = losses[i]
                weight_i = 1.0 / n_tasks
                loss_clean_val += loss_i.item() * weight_i

                retain = (i < n_tasks - 1)
                
                grad_i = torch.autograd.grad(loss_i, model.parameters(), retain_graph=retain, allow_unused=True)

                valid_grads = []
                for p, g in zip(model.parameters(), grad_i):
                    if g is not None:
                        valid_grads.append(g.flatten())
                    else:
                        valid_grads.append(torch.zeros_like(p).flatten())
                task_grads_flattened.append(torch.cat(valid_grads))

                for k, g in enumerate(grad_i):
                    if g is not None:
                        clean_grad_w[k] += g.detach() * weight_i

            # ==========================================================
            # Phase 2: Domain Similarity & Noise Scale Calculation
            # ==========================================================
            avg_sim = calculate_similarity(task_grads_flattened, F.cosine_similarity)
            alpha = args.gga_l_gamma * (1.0 - avg_sim)

            # ==========================================================
            # Phase 3: Compute Spectral Perturbations (Muon Logic)
            # ==========================================================
            eps = []
            for g in clean_grad_w:
                if g.norm() < 1e-12:
                    eps.append(None)
                    continue

                if g.ndim >= 2:
                    orig_shape = g.shape
                    view_2d = g.view(g.size(0), -1) if g.ndim == 4 else g
                    g_ortho = zeropower_via_newtonschulz5(view_2d, steps=5)
                    e = g_ortho.view(orig_shape) * args.rho
                    eps.append(e)
                else:
                    norm_val = g.norm(2)
                    if norm_val > 1e-12:
                        e = (g / norm_val) * args.rho
                    else:
                        e = None
                    eps.append(e)

            # ==========================================================
            # Phase 4: Apply Perturbation (Ascent Step)
            # ==========================================================
            with torch.no_grad():
                for p, v in zip(model.parameters(), eps):
                    if v is not None:
                        p.add_(v)

            # ==========================================================
            # Phase 5: Adversarial Forward & Backward Pass
            # ==========================================================
            disable_running_stats(model)
            model.zero_grad()
            optimizer.zero_grad()

            out_pert, features_pert = model(data, return_representation=True)
            losses_pert = F.mse_loss(out_pert, data.y, reduction="none").mean(0)

            # Backward through the perturbed weights to get task-weighted gradients
            loss_pert, extra_outputs = weight_method.backward(
                losses=losses_pert,
                shared_parameters=list(model.shared_parameters()),
                task_specific_parameters=list(model.task_specific_parameters()),
                last_shared_parameters=list(model.last_shared_parameters()),
                representation=features_pert,
            )

            # ==========================================================
            # Phase 6: Restore Weights & Inject GGA-L Noise (Descent Step)
            # ==========================================================
            with torch.no_grad():
                for p, v in zip(model.parameters(), eps):
                    if v is not None:
                        p.sub_(v)  # Restore clean weights

                    # Inject dynamic domain-conflict noise into the final gradient
                    if p.grad is not None:
                        noise = torch.randn_like(p.grad) * alpha
                        p.grad.add_(noise)

            # ==========================================================
            # Phase 7: Optimizer Step
            # ==========================================================
            optimizer.step()

            loss_list.append(losses.detach().cpu())

            if "famo" in args.method:
                with torch.no_grad():
                    out_famo = model(data, return_representation=False)
                    new_losses = F.mse_loss(out_famo, data.y,
                                            reduction="none").mean(0)
                    weight_method.method.update(new_losses.detach())

        val_loss_dict = evaluate(model,
                                 val_loader,
                                 std=std,
                                 scale_target=scale_target,
                                 device=device)
        test_loss_dict = evaluate(model,
                                  test_loader,
                                  std=std,
                                  scale_target=scale_target,
                                  device=device)
        val_loss = val_loss_dict["avg_loss"]
        val_delta = val_loss_dict["delta_m"]
        test_loss = test_loss_dict["avg_loss"]
        test_delta = test_loss_dict["delta_m"]

        if method == "stl":
            best_val_criteria = val_loss_dict["avg_task_losses"][
                main_task] <= best_val
        else:
            best_val_criteria = val_delta <= best_val_delta

        if best_val_criteria:
            best_val = val_loss
            best_test = test_loss
            best_test_results = test_loss_dict
            best_val_delta = val_delta
            best_test_delta = test_delta

        avg_cost[epoch, 0] = val_loss
        avg_cost[epoch, 1] = val_delta
        avg_cost[epoch, 2:2 + 11] = val_loss_dict["avg_task_losses"]
        avg_cost[epoch, 13] = test_loss
        avg_cost[epoch, 14:14 + 11] = test_loss_dict["avg_task_losses"]
        deltas[epoch] = test_delta

        sim_val = avg_sim.item() if isinstance(avg_sim, torch.Tensor) else avg_sim
        alpha_val = alpha.item() if isinstance(alpha, torch.Tensor) else alpha

        # for logger
        epoch_iterator.set_description(
            f"e {epoch} | lr {lr_current} | train {losses.mean().item():.3f} | val {val_loss:.3f} | "
            f"test {test_loss:.3f} | best d_m {best_test_delta:.3f} | sim {sim_val:.3f} | alpha {alpha_val:.3f}"
        )

        if wandb.run is not None:
            wandb.log({"Learning Rate": lr_current}, step=epoch)
            wandb.log({"Train Gradient Similarity": sim_val}, step=epoch)
            wandb.log({"Train Noise Alpha": alpha_val}, step=epoch)
            wandb.log({"Train Loss": losses.mean().item()}, step=epoch)
            wandb.log({"Val Loss": val_loss}, step=epoch)
            wandb.log({"Val Delta": val_delta}, step=epoch)
            wandb.log({"Test Loss": test_loss}, step=epoch)
            wandb.log({"Test Delta": test_delta}, step=epoch)
            wandb.log({"Best Test Loss": best_test}, step=epoch)
            wandb.log({"Best Test Delta": best_test_delta}, step=epoch)

        scheduler.step(val_loss_dict["avg_task_losses"][main_task] if method ==
                       "stl" else val_delta)

        name = f"{args.method}_rho{args.rho}_gamma{args.gga_l_gamma}_sd{args.seed}_muonsamgga"
        torch.save(
            {
                "avg_cost": avg_cost,
                "losses": loss_list,
                "delta_m": deltas,
            }, f"./save/{name}.stats")


if __name__ == "__main__":
    parser = ArgumentParser("QM9", parents=[common_parser])
    parser.set_defaults(
        data_path=os.path.join(os.getcwd(), "dataset"),
        lr=1e-3,
        n_epochs=300,
        batch_size=120,
        method="nashmtl",
    )
    parser.add_argument("--scale-y", default=False, type=str2bool)
    # MuonGGASAM settings
    parser.add_argument("--rho",
                        type=float,
                        default=0.003,
                        help="Rho for pertubation in MuonGGASAM.")
    parser.add_argument("--gga_l_gamma",
                        type=float,
                        default=0.0001,
                        help="Gamma scalar for dynamic domain noise (GGA-L).")
    
    parser.add_argument("--wandb_project",
                        type=str,
                        default=None,
                        help="Name of Weights & Biases Project.")
    parser.add_argument("--wandb_entity",
                        type=str,
                        default=None,
                        help="Name of Weights & Biases Entity.")
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    if args.wandb_project is not None:
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity,
                   config=args)

    weight_method_params = extract_weight_method_parameters_from_args(args)

    device = get_device(gpus=args.gpu)
    main(
        data_path=args.data_path,
        batch_size=args.batch_size,
        device=device,
        method=args.method,
        weight_method_params=weight_method_params,
        lr=args.lr,
        method_params_lr=args.method_params_lr,
        n_epochs=args.n_epochs,
        args=args,
        targets=targets,
        scale_target=args.scale_y,
        main_task=args.main_task,
    )

    if wandb.run is not None:
        wandb.finish()
