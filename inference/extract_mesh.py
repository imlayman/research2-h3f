from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import torch

from h3f_recon.data.block_index import Block, BlockIndex


@dataclass
class BlockwiseExtractConfig:
    grid_resolution: int = 24
    query_batch_size: int = 65536
    top_k: int = 4
    iso_level: float = 0.0
    dedup_epsilon: float = 1e-5
    far_sdf: float = 1.0
    filter_core_faces: bool = True


def infer_auto_block_size(points: np.ndarray, target_blocks_per_axis: int = 8) -> float:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if pts.shape[0] == 0:
        raise ValueError("points cannot be empty")

    extent = pts.max(axis=0) - pts.min(axis=0)
    max_extent = float(np.max(extent))
    if max_extent <= 0.0:
        return 1.0
    target_blocks_per_axis = int(max(1, target_blocks_per_axis))
    return max_extent / float(target_blocks_per_axis)


def _regular_grid_in_core(block: Block, resolution: int) -> np.ndarray:
    if resolution < 2:
        raise ValueError("grid resolution must be >= 2 for marching cubes")
    axis_x = np.linspace(block.core_min[0], block.core_max[0], num=resolution, dtype=np.float32)
    axis_y = np.linspace(block.core_min[1], block.core_max[1], num=resolution, dtype=np.float32)
    axis_z = np.linspace(block.core_min[2], block.core_max[2], num=resolution, dtype=np.float32)
    gx, gy, gz = np.meshgrid(axis_x, axis_y, axis_z, indexing="ij")
    return np.stack([gx, gy, gz], axis=-1).reshape(-1, 3)


def _query_candidate_mask(
    block_index: BlockIndex,
    query_points: np.ndarray,
    top_k: int,
) -> Tuple[np.ndarray, np.ndarray]:
    queries = block_index.query_batch(query_points, top_k=top_k)
    counts = np.asarray([len(cands) for cands in queries], dtype=np.int32)
    mask = counts > 0
    return mask, counts


def _predict_sdf(
    model: torch.nn.Module,
    points: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    values = []
    with torch.no_grad():
        for start in range(0, points.shape[0], batch_size):
            chunk = torch.from_numpy(points[start : start + batch_size]).to(device=device, dtype=torch.float32)
            out = model(chunk)
            values.append(out["sdf"].detach().cpu().reshape(-1))
    return torch.cat(values, dim=0).numpy().astype(np.float32)


def _extract_single_block_mesh(
    model: torch.nn.Module,
    block: Block,
    block_index: BlockIndex,
    cfg: BlockwiseExtractConfig,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    from skimage.measure import marching_cubes

    grid_points = _regular_grid_in_core(block=block, resolution=cfg.grid_resolution)
    valid_mask, candidate_counts = _query_candidate_mask(
        block_index=block_index,
        query_points=grid_points,
        top_k=cfg.top_k,
    )

    sdf_flat = np.full((grid_points.shape[0],), float(cfg.far_sdf), dtype=np.float32)
    if np.any(valid_mask):
        # The model forward already outputs the fused SDF s(x).
        sdf_flat[valid_mask] = _predict_sdf(
            model=model,
            points=grid_points[valid_mask],
            device=device,
            batch_size=cfg.query_batch_size,
        )

    sdf_grid = sdf_flat.reshape(cfg.grid_resolution, cfg.grid_resolution, cfg.grid_resolution)
    sdf_min = float(np.min(sdf_grid))
    sdf_max = float(np.max(sdf_grid))

    stats = {
        "grid_points": int(grid_points.shape[0]),
        "valid_query_points": int(valid_mask.sum()),
        "max_candidates": int(candidate_counts.max()) if candidate_counts.size > 0 else 0,
        "triangles": 0,
    }

    if not (sdf_min <= cfg.iso_level <= sdf_max):
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64), stats

    spacing = (block.core_max - block.core_min) / float(cfg.grid_resolution - 1)
    try:
        vertices, faces, _, _ = marching_cubes(
            volume=sdf_grid,
            level=float(cfg.iso_level),
            spacing=(float(spacing[0]), float(spacing[1]), float(spacing[2])),
            allow_degenerate=False,
        )
    except ValueError:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64), stats

    vertices = vertices.astype(np.float32) + block.core_min[None, :]
    faces = faces.astype(np.int64)

    if cfg.filter_core_faces and faces.shape[0] > 0:
        tri_centers = vertices[faces].mean(axis=1)
        keep = np.all(
            (tri_centers >= (block.core_min[None, :] - 1e-6))
            & (tri_centers <= (block.core_max[None, :] + 1e-6)),
            axis=1,
        )
        faces = faces[keep]

    stats["triangles"] = int(faces.shape[0])
    if faces.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64), stats
    return vertices, faces, stats


def _merge_meshes(
    vertices: np.ndarray,
    faces: np.ndarray,
    dedup_epsilon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.int64)
    if dedup_epsilon <= 0.0:
        return vertices.astype(np.float32), faces.astype(np.int64)

    scale = 1.0 / float(dedup_epsilon)
    quantized = np.round(vertices * scale).astype(np.int64)

    key_to_new = {}
    remap = np.empty((vertices.shape[0],), dtype=np.int64)
    unique_vertices = []
    for i in range(vertices.shape[0]):
        key = (int(quantized[i, 0]), int(quantized[i, 1]), int(quantized[i, 2]))
        idx = key_to_new.get(key)
        if idx is None:
            idx = len(unique_vertices)
            key_to_new[key] = idx
            unique_vertices.append(vertices[i])
        remap[i] = idx

    merged_vertices = np.asarray(unique_vertices, dtype=np.float32)
    merged_faces = remap[faces.reshape(-1)].reshape(-1, 3).astype(np.int64)

    non_degenerate = (
        (merged_faces[:, 0] != merged_faces[:, 1])
        & (merged_faces[:, 0] != merged_faces[:, 2])
        & (merged_faces[:, 1] != merged_faces[:, 2])
    )
    merged_faces = merged_faces[non_degenerate]

    if merged_faces.shape[0] == 0:
        return merged_vertices, merged_faces

    normalized_faces = np.sort(merged_faces, axis=1)
    _, keep_indices = np.unique(normalized_faces, axis=0, return_index=True)
    merged_faces = merged_faces[np.sort(keep_indices)]
    return merged_vertices, merged_faces


def save_mesh_ply(path: str, vertices: np.ndarray, faces: np.ndarray) -> None:
    v = np.asarray(vertices, dtype=np.float32)
    f = np.asarray(faces, dtype=np.int64)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError("vertices must have shape [N, 3]")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError("faces must have shape [M, 3]")

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fw:
        fw.write("ply\n")
        fw.write("format ascii 1.0\n")
        fw.write(f"element vertex {v.shape[0]}\n")
        fw.write("property float x\n")
        fw.write("property float y\n")
        fw.write("property float z\n")
        fw.write(f"element face {f.shape[0]}\n")
        fw.write("property list uchar int vertex_indices\n")
        fw.write("end_header\n")

        for vx, vy, vz in v:
            fw.write(f"{vx:.6f} {vy:.6f} {vz:.6f}\n")
        for i0, i1, i2 in f:
            fw.write(f"3 {int(i0)} {int(i1)} {int(i2)}\n")


def extract_mesh_blockwise(
    model: torch.nn.Module,
    block_index: BlockIndex,
    cfg: BlockwiseExtractConfig,
    device: torch.device,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    if cfg.grid_resolution < 2:
        raise ValueError("grid_resolution must be >= 2")
    if cfg.query_batch_size <= 0:
        raise ValueError("query_batch_size must be > 0")
    if cfg.top_k <= 0:
        raise ValueError("top_k must be > 0")

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    summary = {
        "num_blocks": int(block_index.num_blocks),
        "grid_points": 0,
        "valid_query_points": 0,
        "blocks_with_mesh": 0,
        "triangles_before_merge": 0,
        "vertices_before_merge": 0,
    }

    for i, block in enumerate(block_index.blocks):
        v_block, f_block, stats = _extract_single_block_mesh(
            model=model,
            block=block,
            block_index=block_index,
            cfg=cfg,
            device=device,
        )

        summary["grid_points"] += stats["grid_points"]
        summary["valid_query_points"] += stats["valid_query_points"]

        if v_block.shape[0] == 0 or f_block.shape[0] == 0:
            if log_fn is not None and ((i + 1) % 20 == 0 or (i + 1) == block_index.num_blocks):
                log_fn(f"Block {i + 1}/{block_index.num_blocks}: no mesh extracted")
            continue

        f_block = f_block + vertex_offset
        vertex_offset += v_block.shape[0]

        all_vertices.append(v_block)
        all_faces.append(f_block)

        summary["blocks_with_mesh"] += 1
        summary["triangles_before_merge"] += int(f_block.shape[0])
        summary["vertices_before_merge"] += int(v_block.shape[0])

        if log_fn is not None and ((i + 1) % 10 == 0 or (i + 1) == block_index.num_blocks):
            log_fn(
                f"Block {i + 1}/{block_index.num_blocks}: "
                f"tris={stats['triangles']} valid_pts={stats['valid_query_points']}/{stats['grid_points']}"
            )

    if all_vertices and all_faces:
        merged_vertices = np.concatenate(all_vertices, axis=0).astype(np.float32)
        merged_faces = np.concatenate(all_faces, axis=0).astype(np.int64)
        merged_vertices, merged_faces = _merge_meshes(
            vertices=merged_vertices,
            faces=merged_faces,
            dedup_epsilon=float(cfg.dedup_epsilon),
        )
    else:
        merged_vertices = np.zeros((0, 3), dtype=np.float32)
        merged_faces = np.zeros((0, 3), dtype=np.int64)

    summary["vertices_after_merge"] = int(merged_vertices.shape[0])
    summary["triangles_after_merge"] = int(merged_faces.shape[0])
    return merged_vertices, merged_faces, summary
