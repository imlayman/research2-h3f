from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from h3f_recon.config import LossConfig
from h3f_recon.data.block_index import BlockIndex


def _sample_surface_points(batch: Dict[str, torch.Tensor], sample_count: int) -> torch.Tensor:
    surface = batch["surface_points"].reshape(-1, 3)
    if surface.shape[0] == 0 or sample_count <= 0:
        return surface.new_empty((0, 3))
    idx = torch.randint(0, surface.shape[0], (sample_count,), device=surface.device)
    return surface[idx]


def _sample_near_surface_pairs(
    batch: Dict[str, torch.Tensor],
    sample_count: int,
    epsilon: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    surface = batch["surface_points"].reshape(-1, 3)
    if surface.shape[0] == 0 or sample_count <= 0:
        empty = surface.new_empty((0, 3))
        return empty, empty

    idx = torch.randint(0, surface.shape[0], (sample_count,), device=surface.device)
    base = surface[idx]

    normals: Optional[torch.Tensor] = None
    if "surface_normals" in batch:
        normals_raw = batch["surface_normals"].reshape(-1, 3)
        if normals_raw.shape[0] == surface.shape[0]:
            normals = normals_raw[idx]

    random_dirs = torch.randn_like(base)
    random_dirs = random_dirs / random_dirs.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    if normals is None:
        directions = random_dirs
    else:
        normal_norm = normals.norm(dim=-1, keepdim=True)
        valid = normal_norm > 1e-6
        normalized = normals / normal_norm.clamp_min(1e-6)
        directions = torch.where(valid, normalized, random_dirs)

    plus = base + float(epsilon) * directions
    minus = base - float(epsilon) * directions
    return plus, minus


def _sample_uniform_points(count: int, world_bound: float, device: torch.device) -> torch.Tensor:
    if count <= 0:
        return torch.empty((0, 3), device=device)
    return (torch.rand((count, 3), device=device) * 2.0 - 1.0) * float(world_bound)


def _build_context_from_batch(
    batch: Dict[str, torch.Tensor],
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    surface_points = batch.get("surface_points")
    surface_normals = batch.get("surface_normals")
    point_cloud = batch.get("point_cloud")

    context_points: Optional[torch.Tensor]
    context_normals: Optional[torch.Tensor]

    if torch.is_tensor(surface_points) and surface_points.numel() > 0:
        context_points = surface_points.reshape(-1, 3)
        if torch.is_tensor(surface_normals) and surface_normals.shape == surface_points.shape:
            context_normals = surface_normals.reshape(-1, 3)
        else:
            context_normals = None
        return context_points, context_normals

    if torch.is_tensor(point_cloud) and point_cloud.numel() > 0:
        context_points = point_cloud.reshape(-1, 3)
        return context_points, None

    return None, None


def _build_overlap_boxes(index: BlockIndex) -> List[np.ndarray]:
    boxes: List[np.ndarray] = []
    neighbor_offsets = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    for block in index.blocks:
        gx, gy, gz = block.grid_coord
        for ox, oy, oz in neighbor_offsets:
            neighbor_id = index.coord_to_block_id.get((gx + ox, gy + oy, gz + oz))
            if neighbor_id is None:
                continue
            neighbor = index.blocks[neighbor_id]

            low = np.maximum(block.overlap_min, neighbor.overlap_min)
            high = np.minimum(block.overlap_max, neighbor.overlap_max)
            if np.all(high > low):
                boxes.append(np.stack([low, high], axis=0).astype(np.float32))
    return boxes


def _sample_seam_points(
    batch: Dict[str, torch.Tensor],
    seam_sample_count: int,
    block_size: float,
    overlap_ratio: float,
) -> torch.Tensor:
    point_cloud = batch["point_cloud"]
    device = point_cloud.device

    if seam_sample_count <= 0 or point_cloud.ndim != 3:
        return torch.empty((0, 3), device=device)

    per_cloud = max(1, seam_sample_count // max(1, point_cloud.shape[0]))
    all_samples: List[torch.Tensor] = []

    for b in range(point_cloud.shape[0]):
        points_np = point_cloud[b].detach().cpu().numpy()
        if points_np.shape[0] < 2:
            continue

        index = BlockIndex(points=points_np, block_size=block_size, overlap_ratio=overlap_ratio)
        boxes = _build_overlap_boxes(index)
        if not boxes:
            continue

        boxes_np = np.stack(boxes, axis=0)
        boxes_t = torch.from_numpy(boxes_np).to(device=device)
        num_boxes = boxes_t.shape[0]

        choice = torch.randint(0, num_boxes, (per_cloud,), device=device)
        lows = boxes_t[choice, 0, :]
        highs = boxes_t[choice, 1, :]
        rand = torch.rand((per_cloud, 3), device=device)
        samples = lows + rand * (highs - lows)
        all_samples.append(samples)

    if not all_samples:
        return torch.empty((0, 3), device=device)

    seam_points = torch.cat(all_samples, dim=0)
    if seam_points.shape[0] > seam_sample_count:
        idx = torch.randint(0, seam_points.shape[0], (seam_sample_count,), device=device)
        seam_points = seam_points[idx]
    return seam_points


def _compute_seam_value_loss(
    model: torch.nn.Module,
    seam_points: torch.Tensor,
    context_points: Optional[torch.Tensor],
    context_normals: Optional[torch.Tensor],
) -> torch.Tensor:
    if seam_points.numel() == 0:
        return seam_points.new_tensor(0.0)

    out = model(
        seam_points,
        context_points=context_points,
        context_normals=context_normals,
        return_intermediates=True,
    )
    local_sdf = out.get("local_sdf")
    if local_sdf is None:
        return seam_points.new_tensor(0.0)

    local = local_sdf.squeeze(-1)  # [N, K]
    k = local.shape[1]
    if k < 2:
        return seam_points.new_tensor(0.0)

    diff = torch.abs(local.unsqueeze(2) - local.unsqueeze(1))
    mask = torch.triu(torch.ones((k, k), device=local.device, dtype=torch.bool), diagonal=1)
    return diff[:, mask].mean()


def _compute_seam_grad_loss(
    model: torch.nn.Module,
    seam_points: torch.Tensor,
    context_points: Optional[torch.Tensor],
    context_normals: Optional[torch.Tensor],
) -> torch.Tensor:
    if seam_points.numel() == 0:
        return seam_points.new_tensor(0.0)

    sample = seam_points.detach().clone().requires_grad_(True)
    out = model(
        sample,
        context_points=context_points,
        context_normals=context_normals,
        return_intermediates=True,
    )
    local_sdf = out.get("local_sdf")
    if local_sdf is None:
        return seam_points.new_tensor(0.0)

    local = local_sdf.squeeze(-1)  # [N, K]
    k = local.shape[1]
    if k < 2:
        return seam_points.new_tensor(0.0)

    grads = []
    for i in range(k):
        grad_i = torch.autograd.grad(
            outputs=local[:, i].sum(),
            inputs=sample,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        grads.append(grad_i)

    grad_stack = torch.stack(grads, dim=1)  # [N, K, 3]
    grad_diff = grad_stack.unsqueeze(2) - grad_stack.unsqueeze(1)  # [N, K, K, 3]
    grad_norm = grad_diff.norm(dim=-1)

    mask = torch.triu(torch.ones((k, k), device=grad_norm.device, dtype=torch.bool), diagonal=1)
    return grad_norm[:, mask].mean()


def _compute_blend_smooth_loss(
    model: torch.nn.Module,
    points: torch.Tensor,
    context_points: Optional[torch.Tensor],
    context_normals: Optional[torch.Tensor],
) -> torch.Tensor:
    if points.numel() == 0:
        return points.new_tensor(0.0)

    sample = points.detach().clone().requires_grad_(True)
    out = model(
        sample,
        context_points=context_points,
        context_normals=context_normals,
        return_intermediates=True,
    )
    weights = out.get("weights")
    if weights is None:
        return points.new_tensor(0.0)

    blend = weights.squeeze(-1)  # [N, K]
    if blend.ndim != 2 or blend.shape[1] == 0:
        return points.new_tensor(0.0)

    per_head = []
    for i in range(blend.shape[1]):
        grad_i = torch.autograd.grad(
            outputs=blend[:, i].sum(),
            inputs=sample,
            create_graph=True,
            retain_graph=True,
            allow_unused=False,
        )[0]
        per_head.append((grad_i.norm(dim=-1) ** 2).mean())

    return torch.stack(per_head).mean()


def _compute_eikonal_loss(
    model: torch.nn.Module,
    points: torch.Tensor,
    context_points: Optional[torch.Tensor],
    context_normals: Optional[torch.Tensor],
) -> torch.Tensor:
    if points.numel() == 0:
        return points.new_tensor(0.0)
    sample = points.detach().clone().requires_grad_(True)
    out = model(sample, context_points=context_points, context_normals=context_normals)
    grad = torch.autograd.grad(
        outputs=out["sdf"],
        inputs=sample,
        grad_outputs=torch.ones_like(out["sdf"]),
        create_graph=True,
        retain_graph=True,
    )[0]
    return ((grad.norm(dim=-1) - 1.0) ** 2).mean()


def compute_training_losses(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    loss_cfg: LossConfig,
    surf_sample_count: int,
    near_sample_count: int,
    near_epsilon: float,
    near_loss_type: str,
    seam_sample_count: int,
    seam_block_size: float,
    seam_overlap_ratio: float,
    eikonal_sample_count: int,
    world_bound: float,
    use_eikonal: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = batch["surface_points"].device

    surface_points = _sample_surface_points(batch=batch, sample_count=surf_sample_count)
    near_plus, near_minus = _sample_near_surface_pairs(
        batch=batch,
        sample_count=near_sample_count,
        epsilon=near_epsilon,
    )
    seam_points = _sample_seam_points(
        batch=batch,
        seam_sample_count=seam_sample_count,
        block_size=seam_block_size,
        overlap_ratio=seam_overlap_ratio,
    )
    context_points, context_normals = _build_context_from_batch(batch)

    losses: Dict[str, torch.Tensor] = {}

    if surface_points.numel() > 0:
        surf_out = model(surface_points, context_points=context_points, context_normals=context_normals)
        losses["surface"] = surf_out["sdf"].abs().mean()
    else:
        losses["surface"] = torch.tensor(0.0, device=device)

    near_loss_type = near_loss_type.lower().strip()
    if near_plus.numel() > 0:
        plus_out = model(near_plus, context_points=context_points, context_normals=context_normals)["sdf"]
        minus_out = model(near_minus, context_points=context_points, context_normals=context_normals)["sdf"]
        if near_loss_type == "sign":
            losses["near"] = F.softplus(-plus_out).mean() + F.softplus(minus_out).mean()
        else:
            target_plus = torch.full_like(plus_out, float(near_epsilon))
            target_minus = torch.full_like(minus_out, -float(near_epsilon))
            losses["near"] = F.smooth_l1_loss(plus_out, target_plus) + F.smooth_l1_loss(minus_out, target_minus)
    else:
        losses["near"] = torch.tensor(0.0, device=device)

    losses["seam_value"] = _compute_seam_value_loss(
        model=model,
        seam_points=seam_points,
        context_points=context_points,
        context_normals=context_normals,
    )
    losses["seam_grad"] = _compute_seam_grad_loss(
        model=model,
        seam_points=seam_points,
        context_points=context_points,
        context_normals=context_normals,
    )

    if seam_points.numel() > 0:
        smooth_points = seam_points
    elif near_plus.numel() > 0:
        smooth_points = torch.cat([near_plus, near_minus], dim=0)
    else:
        smooth_points = surface_points
    losses["blend_smooth"] = _compute_blend_smooth_loss(
        model=model,
        points=smooth_points,
        context_points=context_points,
        context_normals=context_normals,
    )

    if use_eikonal and eikonal_sample_count > 0:
        eik_points = _sample_uniform_points(
            count=eikonal_sample_count,
            world_bound=world_bound,
            device=device,
        )
        losses["eikonal"] = _compute_eikonal_loss(
            model=model,
            points=eik_points,
            context_points=context_points,
            context_normals=context_normals,
        )
    else:
        losses["eikonal"] = torch.tensor(0.0, device=device)

    total = (
        loss_cfg.surface * losses["surface"]
        + loss_cfg.sign * losses["near"]
        + loss_cfg.seam * losses["seam_value"]
        + loss_cfg.seam_grad * losses["seam_grad"]
        + loss_cfg.blend_smooth * losses["blend_smooth"]
        + loss_cfg.eikonal * losses["eikonal"]
    )

    stats = {name: float(value.detach().item()) for name, value in losses.items()}
    stats["total"] = float(total.detach().item())
    return total, stats
