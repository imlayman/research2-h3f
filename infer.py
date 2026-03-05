import argparse
import os

import numpy as np
import torch

from h3f_recon.config import load_config, save_config
from h3f_recon.data import DummyPointCloudDataset
from h3f_recon.data.block_index import BlockIndex
from h3f_recon.data.dataset import load_point_cloud, voxel_downsample
from h3f_recon.models import H3FRecon
from h3f_recon.utils import (
    create_run_dir,
    load_checkpoint,
    resolve_device,
    set_seed,
    setup_logger,
)
from inference.extract_mesh import (
    BlockwiseExtractConfig,
    extract_mesh_blockwise,
    infer_auto_block_size,
    save_mesh_ply,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blockwise mesh extraction for H3F-Recon MVP")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path")
    parser.add_argument("--run-name", type=str, default="", help="Override infer.run_name")
    parser.add_argument("--device", type=str, default="", help="Override infer.device")
    parser.add_argument("--point-cloud", type=str, default="", help="Input point cloud path (.ply/.xyz/.pcd/.npz)")
    parser.add_argument("--voxel-size", type=float, default=0.0, help="Optional voxel downsample size")
    parser.add_argument("--block-size", type=float, default=0.0, help="Block core size, auto if <= 0")
    parser.add_argument(
        "--overlap-ratio",
        type=float,
        default=-1.0,
        help="Block overlap ratio, default uses 0.25 if not specified",
    )
    parser.add_argument("--top-k", type=int, default=0, help="Top-k candidate blocks per query point")
    parser.add_argument("--grid-resolution", type=int, default=0, help="Grid resolution per block core")
    parser.add_argument("--query-batch-size", type=int, default=0, help="Model query batch size")
    parser.add_argument("--iso-level", type=float, default=None, help="Marching cubes level")
    parser.add_argument(
        "--iso-threshold",
        type=float,
        default=None,
        help="Compatibility alias of --iso-level",
    )
    parser.add_argument("--dedup-eps", type=float, default=0.0, help="Vertex dedup epsilon, <=0 keeps default")
    return parser.parse_args()


def _load_inference_points(args: argparse.Namespace, cfg) -> tuple[np.ndarray, np.ndarray | None, str]:
    if args.point_cloud:
        points, normals = load_point_cloud(file_path=args.point_cloud, estimate_normals=False)
        source = args.point_cloud
    elif cfg.data.mode == "dummy":
        dataset = DummyPointCloudDataset(cfg.data, split="val", base_seed=cfg.train.seed)
        sample = dataset[0]
        points = sample["point_cloud"].detach().cpu().numpy().astype(np.float32)
        normals = None
        source = "dummy_dataset[val:0]"
    else:
        raise ValueError("Please provide --point-cloud for non-dummy data mode")

    if args.voxel_size > 0.0:
        points, normals = voxel_downsample(points=points, normals=normals, voxel_size=float(args.voxel_size))
    if points.shape[0] == 0:
        raise ValueError("Input point cloud is empty after preprocessing")
    normals_out = None if normals is None else normals.astype(np.float32)
    return points.astype(np.float32), normals_out, source


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


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.run_name:
        cfg.infer.run_name = args.run_name
    if args.device:
        cfg.infer.device = args.device
    if args.checkpoint:
        cfg.infer.checkpoint = args.checkpoint

    if not cfg.infer.checkpoint:
        raise ValueError("Please provide checkpoint via --checkpoint or infer.checkpoint in config")

    run_dir = create_run_dir(cfg.output_root, cfg.infer.run_name, prefix="infer")
    logger = setup_logger("h3f_infer", os.path.join(run_dir, "infer.log"))
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
    points, normals, point_source = _load_inference_points(args=args, cfg=cfg)

    overlap_ratio = float(args.overlap_ratio if args.overlap_ratio >= 0.0 else 0.25)
    block_size = float(args.block_size if args.block_size > 0.0 else infer_auto_block_size(points))
    top_k = int(args.top_k if args.top_k > 0 else max(1, int(getattr(cfg.model, "top_k", 1))))
    grid_resolution = int(args.grid_resolution if args.grid_resolution > 0 else cfg.infer.grid_resolution)
    query_batch_size = int(args.query_batch_size if args.query_batch_size > 0 else cfg.infer.query_batch_size)
    iso_level = _resolve_iso_level(args=args, cfg=cfg)
    dedup_eps = float(args.dedup_eps if args.dedup_eps > 0.0 else 1e-5)

    logger.info("Point source: %s", point_source)
    logger.info("Input points: %d", points.shape[0])
    logger.info("Block setup: block_size=%.6f overlap=%.3f top_k=%d", block_size, overlap_ratio, top_k)
    logger.info(
        "Extraction setup: grid_resolution=%d query_batch_size=%d iso_level=%.6f dedup_eps=%.1e",
        grid_resolution,
        query_batch_size,
        iso_level,
        dedup_eps,
    )

    block_index = BlockIndex(points=points, block_size=block_size, overlap_ratio=overlap_ratio)
    logger.info("Built block index: %d blocks", block_index.num_blocks)

    context_points_t = torch.from_numpy(points).to(device=device, dtype=torch.float32)
    context_normals_t = (
        None
        if normals is None
        else torch.from_numpy(normals).to(device=device, dtype=torch.float32)
    )

    extract_cfg = BlockwiseExtractConfig(
        grid_resolution=grid_resolution,
        query_batch_size=query_batch_size,
        top_k=top_k,
        iso_level=iso_level,
        dedup_epsilon=dedup_eps,
        far_sdf=max(1.0, abs(iso_level) + 1.0),
        filter_core_faces=True,
    )

    vertices, faces, stats = extract_mesh_blockwise(
        model=model,
        block_index=block_index,
        cfg=extract_cfg,
        device=device,
        context_points=context_points_t,
        context_normals=context_normals_t,
        log_fn=logger.info,
    )

    ply_path = os.path.join(run_dir, f"{cfg.infer.output_name}.ply")
    save_mesh_ply(ply_path, vertices=vertices, faces=faces)
    np.save(os.path.join(run_dir, f"{cfg.infer.output_name}_vertices.npy"), vertices)
    np.save(os.path.join(run_dir, f"{cfg.infer.output_name}_faces.npy"), faces)

    logger.info(
        "Mesh summary: blocks=%d blocks_with_mesh=%d grid_points=%d valid_points=%d",
        stats["num_blocks"],
        stats["blocks_with_mesh"],
        stats["grid_points"],
        stats["valid_query_points"],
    )
    logger.info(
        "Mesh summary: vertices=%d triangles=%d (before merge: v=%d t=%d)",
        stats["vertices_after_merge"],
        stats["triangles_after_merge"],
        stats["vertices_before_merge"],
        stats["triangles_before_merge"],
    )
    logger.info("Saved mesh to: %s", ply_path)
    logger.info("Saved outputs to: %s", run_dir)


if __name__ == "__main__":
    main()
