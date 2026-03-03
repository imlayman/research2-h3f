from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F

from h3f_recon.config import LossConfig


def _upper_triangle_mean(matrix: torch.Tensor) -> torch.Tensor:
    k = matrix.size(-1)
    if k < 2:
        return matrix.new_tensor(0.0)
    mask = torch.triu(torch.ones((k, k), device=matrix.device, dtype=torch.bool), diagonal=1)
    return matrix[:, mask].mean()


def compute_eikonal_loss(model: torch.nn.Module, points: torch.Tensor) -> torch.Tensor:
    if points.numel() == 0:
        return points.new_tensor(0.0)

    sample = points.detach().clone().requires_grad_(True)
    out = model(sample)
    grad = torch.autograd.grad(
        outputs=out["sdf"],
        inputs=sample,
        grad_outputs=torch.ones_like(out["sdf"]),
        create_graph=True,
        retain_graph=True,
    )[0]
    return ((grad.norm(dim=-1) - 1.0) ** 2).mean()


def compute_seam_loss(model: torch.nn.Module, points: torch.Tensor, seam_grad_weight: float) -> torch.Tensor:
    if points.numel() == 0:
        return points.new_tensor(0.0)

    require_grad = seam_grad_weight > 0.0
    sample = points.detach().clone().requires_grad_(require_grad)

    out = model(sample, return_intermediates=True)
    local_sdf = out["local_sdf"].squeeze(-1)
    k = local_sdf.size(1)
    if k < 2:
        return local_sdf.new_tensor(0.0)

    value_pair_diff = torch.abs(local_sdf.unsqueeze(2) - local_sdf.unsqueeze(1))
    value_loss = _upper_triangle_mean(value_pair_diff)

    if seam_grad_weight <= 0.0:
        return value_loss

    grads = []
    for i in range(k):
        grad_i = torch.autograd.grad(
            outputs=local_sdf[:, i].sum(),
            inputs=sample,
            create_graph=True,
            retain_graph=True,
        )[0]
        grads.append(grad_i)
    grad_stack = torch.stack(grads, dim=1)

    grad_pair_diff = torch.norm(grad_stack.unsqueeze(2) - grad_stack.unsqueeze(1), dim=-1)
    grad_loss = _upper_triangle_mean(grad_pair_diff)

    return value_loss + seam_grad_weight * grad_loss


def compute_blend_smoothness_loss(model: torch.nn.Module, points: torch.Tensor) -> torch.Tensor:
    if points.numel() == 0:
        return points.new_tensor(0.0)

    sample = points.detach().clone().requires_grad_(True)
    out = model(sample, return_intermediates=True)
    weights = out["weights"].squeeze(-1)
    k = weights.size(1)
    if k == 0:
        return weights.new_tensor(0.0)

    smoothness = weights.new_tensor(0.0)
    for i in range(k):
        grad_i = torch.autograd.grad(
            outputs=weights[:, i].sum(),
            inputs=sample,
            create_graph=True,
            retain_graph=True,
        )[0]
        smoothness = smoothness + grad_i.norm(dim=-1).mean()

    return smoothness / float(k)


def compute_training_losses(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_cfg: LossConfig,
    seam_sample_count: int,
    eikonal_sample_count: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    surface_points = batch["surface_points"].reshape(-1, 3)
    near_points = batch["near_points"].reshape(-1, 3)
    near_sdf = batch["near_sdf"].reshape(-1, 1)
    eikonal_points = batch["eikonal_points"].reshape(-1, 3)

    surface_out = model(surface_points)
    near_out = model(near_points, return_intermediates=True)

    losses: Dict[str, torch.Tensor] = {}
    losses["surface"] = surface_out["sdf"].abs().mean()
    losses["sign"] = F.smooth_l1_loss(near_out["sdf"], near_sdf)
    losses["coarse_global"] = F.l1_loss(near_out["sdf"], near_out["coarse_sdf"].detach())

    seam_count = min(seam_sample_count, near_points.size(0))
    seam_points = near_points[:seam_count]
    losses["seam"] = compute_seam_loss(model, seam_points, seam_grad_weight=loss_cfg.seam_grad)

    if loss_cfg.blend_smooth > 0.0:
        losses["blend_smooth"] = compute_blend_smoothness_loss(model, seam_points)
    else:
        losses["blend_smooth"] = near_points.new_tensor(0.0)

    eik_count = min(eikonal_sample_count, eikonal_points.size(0))
    losses["eikonal"] = compute_eikonal_loss(model, eikonal_points[:eik_count])

    total = (
        loss_cfg.surface * losses["surface"]
        + loss_cfg.sign * losses["sign"]
        + loss_cfg.eikonal * losses["eikonal"]
        + loss_cfg.seam * losses["seam"]
        + loss_cfg.coarse_global * losses["coarse_global"]
        + loss_cfg.blend_smooth * losses["blend_smooth"]
    )

    stats = {name: float(value.detach().item()) for name, value in losses.items()}
    stats["total"] = float(total.detach().item())
    return total, stats
