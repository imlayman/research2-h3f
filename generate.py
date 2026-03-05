import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from h3f_recon.config import load_config, save_config
from h3f_recon.data.block_index import BlockIndex
from h3f_recon.data.dataset import discover_dataset_point_clouds, load_point_cloud, voxel_downsample
from h3f_recon.models import H3FRecon
from h3f_recon.utils import create_run_dir, load_checkpoint, resolve_device, set_seed, setup_logger
from inference.extract_mesh import (
    BlockwiseExtractConfig,
    extract_mesh_blockwise,
    infer_auto_block_size,
    save_mesh_ply,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch mesh generation on dataset split")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path")
    parser.add_argument("--run-name", type=str, default="", help="Override run name")
    parser.add_argument("--device", type=str, default="", help="Override device")

    parser.add_argument("--split", type=str, default="test", help="Dataset split to generate on")
    parser.add_argument("--dataset-root", type=str, default="", help="Override dataset root")
    parser.add_argument("--dataset-name", type=str, default="", help="Override dataset name")
    parser.add_argument("--categories", type=str, default="", help="Comma-separated categories")
    parser.add_argument("--start", type=int, default=0, help="Start index in split")
    parser.add_argument("--take", type=int, default=-1, help="Take first N samples after start, <=0 for all")

    parser.add_argument("--voxel-size", type=float, default=0.0, help="Optional voxel downsample size")
    parser.add_argument("--block-size", type=float, default=0.0, help="Block core size, auto if <= 0")
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=-1.0,
        help="Block overlap ratio, defaults to data.overlap_ratio",
    )
    parser.add_argument("--top-k", type=int, default=0, help="Top-k candidate blocks per query point")
    parser.add_argument("--grid-resolution", type=int, default=0, help="Grid resolution per block core")
    parser.add_argument("--query-batch-size", type=int, default=0, help="Model query batch size")
    parser.add_argument("--iso-level", type=float, default=None, help="Marching cubes level")
    parser.add_argument("--iso-threshold", type=float, default=None, help="Compatibility alias of --iso-level")
    parser.add_argument("--dedup-eps", type=float, default=0.0, help="Vertex dedup epsilon, <=0 keeps default")
    return parser.parse_args()


def _parse_categories(raw: str) -> List[str]:
    if not raw.strip():
        return []
    return [s.strip() for s in raw.split(",") if s.strip()]


def _resolve_iso_level(args: argparse.Namespace, cfg) -> float:
    if args.iso_level is not None:
        return float(args.iso_level)
    if args.iso_threshold is not None:
        return float(args.iso_threshold)
    if hasattr(cfg.infer, "iso_level"):
        return float(cfg.infer.iso_level)
    if hasattr(cfg.infer, "iso_threshold"):
        return float(cfg.infer.iso_threshold)
    return 0.0


def _scene_tag(scene_path: str, dataset_root: str) -> str:
    p = Path(scene_path)
    root = Path(dataset_root)
    try:
        rel = p.relative_to(root)
    except Exception:
        rel = p
    rel_no_suffix = rel.with_suffix("")
    return str(rel_no_suffix).replace(os.sep, "__")


def _scene_mesh_path(scene_path: str, dataset_root: str, mesh_root: Path) -> Path:
    p = Path(scene_path)
    root = Path(dataset_root)
    try:
        rel = p.relative_to(root)
    except Exception:
        rel = Path(p.name)
    rel_no_suffix = rel.with_suffix("")
    return mesh_root / rel_no_suffix.parent / f"{rel_no_suffix.name}.ply"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.checkpoint:
        cfg.infer.checkpoint = args.checkpoint
    if args.device:
        cfg.infer.device = args.device
    if not cfg.infer.checkpoint:
        raise ValueError("Please provide checkpoint via --checkpoint or infer.checkpoint in config")

    dataset_root = args.dataset_root if args.dataset_root else cfg.data.dataset_root
    dataset_name = args.dataset_name if args.dataset_name else cfg.data.dataset_name
    if not dataset_root:
        raise ValueError("Please provide dataset root via --dataset-root or data.dataset_root in config")
    if not dataset_name:
        raise ValueError("Please provide dataset name via --dataset-name or data.dataset_name in config")

    categories = _parse_categories(args.categories) if args.categories else list(cfg.data.dataset_categories)

    scene_paths = discover_dataset_point_clouds(
        dataset_root=dataset_root,
        dataset_name=dataset_name,
        split=args.split,
        categories=categories,
    )
    start = max(0, int(args.start))
    if start > 0:
        scene_paths = scene_paths[start:]
    if int(args.take) > 0:
        scene_paths = scene_paths[: int(args.take)]
    if not scene_paths:
        raise RuntimeError("No test scenes after start/take filtering")

    run_name = args.run_name if args.run_name else f"{cfg.infer.run_name}_{args.split}"
    run_dir = create_run_dir(cfg.output_root, run_name, prefix="generate")
    logger = setup_logger("h3f_generate", os.path.join(run_dir, "generate.log"))
    save_config(cfg, os.path.join(run_dir, "resolved_config.yaml"))

    set_seed(cfg.train.seed)
    device = resolve_device(cfg.infer.device)
    logger.info("Run dir: %s", run_dir)
    logger.info("Using device: %s", device)

    model = H3FRecon(cfg.model, cfg.data).to(device)
    checkpoint = load_checkpoint(cfg.infer.checkpoint, model, optimizer=None, map_location=device)
    logger.info(
        "Loaded checkpoint: %s (epoch=%s, step=%s)",
        cfg.infer.checkpoint,
        checkpoint.get("epoch"),
        checkpoint.get("step"),
    )
    model.eval()

    overlap_ratio = float(args.overlap_ratio if args.overlap_ratio >= 0.0 else cfg.data.overlap_ratio)
    top_k = int(args.top_k if args.top_k > 0 else max(1, int(getattr(cfg.model, "top_k", 1))))
    grid_resolution = int(args.grid_resolution if args.grid_resolution > 0 else cfg.infer.grid_resolution)
    query_batch_size = int(args.query_batch_size if args.query_batch_size > 0 else cfg.infer.query_batch_size)
    iso_level = _resolve_iso_level(args=args, cfg=cfg)
    dedup_eps = float(args.dedup_eps if args.dedup_eps > 0.0 else 1e-5)

    extract_cfg = BlockwiseExtractConfig(
        grid_resolution=grid_resolution,
        query_batch_size=query_batch_size,
        top_k=top_k,
        iso_level=iso_level,
        dedup_epsilon=dedup_eps,
        far_sdf=max(1.0, abs(iso_level) + 1.0),
        filter_core_faces=True,
    )

    mesh_dir = Path(run_dir) / "meshes"
    mesh_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        (
            "Generate setup | dataset=%s root=%s split=%s scenes=%d start=%d take=%d "
            "grid=%d top_k=%d query_bs=%d iso=%.6f dedup=%.1e"
        ),
        dataset_name,
        dataset_root,
        args.split,
        len(scene_paths),
        start,
        args.take,
        grid_resolution,
        top_k,
        query_batch_size,
        iso_level,
        dedup_eps,
    )

    rows = []
    total_vertices = 0
    total_faces = 0
    total_ok = 0

    for idx, scene_path in enumerate(scene_paths):
        t0 = time.time()
        scene_id = _scene_tag(scene_path=scene_path, dataset_root=dataset_root)
        mesh_path = _scene_mesh_path(scene_path=scene_path, dataset_root=dataset_root, mesh_root=mesh_dir)

        try:
            points, normals = load_point_cloud(file_path=scene_path, estimate_normals=False)
            if args.voxel_size > 0.0:
                points, normals = voxel_downsample(points=points, normals=normals, voxel_size=float(args.voxel_size))
            if points.shape[0] == 0:
                raise ValueError("empty point cloud")

            if args.block_size > 0.0:
                block_size = float(args.block_size)
            elif float(cfg.data.block_size) > 0.0:
                block_size = float(cfg.data.block_size)
            else:
                block_size = float(infer_auto_block_size(points))

            block_index = BlockIndex(points=points, block_size=block_size, overlap_ratio=overlap_ratio)
            context_points_t = torch.from_numpy(points).to(device=device, dtype=torch.float32)
            context_normals_t = (
                None
                if normals is None
                else torch.from_numpy(normals).to(device=device, dtype=torch.float32)
            )

            def _scene_log(msg: str, scene_idx: int = idx + 1, num_scenes: int = len(scene_paths)) -> None:
                logger.info("Scene %d/%d | %s", scene_idx, num_scenes, msg)

            vertices, faces, stats = extract_mesh_blockwise(
                model=model,
                block_index=block_index,
                cfg=extract_cfg,
                device=device,
                context_points=context_points_t,
                context_normals=context_normals_t,
                log_fn=_scene_log,
            )
            save_mesh_ply(str(mesh_path), vertices=vertices, faces=faces)

            elapsed = time.time() - t0
            v_num = int(vertices.shape[0])
            f_num = int(faces.shape[0])
            total_vertices += v_num
            total_faces += f_num
            total_ok += 1

            rows.append(
                {
                    "scene_id": scene_id,
                    "scene_path": scene_path,
                    "mesh_path": str(mesh_path),
                    "status": "ok",
                    "points": int(points.shape[0]),
                    "blocks": int(stats["num_blocks"]),
                    "vertices": v_num,
                    "faces": f_num,
                    "seconds": float(elapsed),
                    "error": "",
                }
            )
            logger.info(
                "Scene %d/%d done | id=%s points=%d vertices=%d faces=%d time=%.2fs",
                idx + 1,
                len(scene_paths),
                scene_id,
                points.shape[0],
                v_num,
                f_num,
                elapsed,
            )
        except Exception as exc:
            elapsed = time.time() - t0
            rows.append(
                {
                    "scene_id": scene_id,
                    "scene_path": scene_path,
                    "mesh_path": str(mesh_path),
                    "status": "failed",
                    "points": 0,
                    "blocks": 0,
                    "vertices": 0,
                    "faces": 0,
                    "seconds": float(elapsed),
                    "error": str(exc),
                }
            )
            logger.exception("Scene %d/%d failed | id=%s", idx + 1, len(scene_paths), scene_id)

    csv_path = Path(run_dir) / "summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_id",
                "scene_path",
                "mesh_path",
                "status",
                "points",
                "blocks",
                "vertices",
                "faces",
                "seconds",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "dataset_name": dataset_name,
        "dataset_root": dataset_root,
        "split": args.split,
        "num_scenes": len(scene_paths),
        "num_success": total_ok,
        "num_failed": len(scene_paths) - total_ok,
        "vertices_total": int(total_vertices),
        "faces_total": int(total_faces),
        "summary_csv": str(csv_path),
    }
    summary_path = Path(run_dir) / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(
        "Generate finished | success=%d failed=%d vertices_total=%d faces_total=%d",
        summary["num_success"],
        summary["num_failed"],
        summary["vertices_total"],
        summary["faces_total"],
    )
    logger.info("Summary CSV: %s", csv_path)
    logger.info("Summary JSON: %s", summary_path)


if __name__ == "__main__":
    main()
