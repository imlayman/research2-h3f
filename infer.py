import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from h3f_recon.config import load_config, save_config
from h3f_recon.models import H3FRecon
from h3f_recon.utils import (
    create_run_dir,
    load_checkpoint,
    resolve_device,
    save_point_cloud,
    set_seed,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference for H3F-Recon MVP")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, default="", help="Checkpoint path")
    parser.add_argument("--run-name", type=str, default="", help="Override infer.run_name")
    parser.add_argument("--device", type=str, default="", help="Override infer.device")
    parser.add_argument("--grid-resolution", type=int, default=0, help="Override infer.grid_resolution")
    parser.add_argument("--iso-threshold", type=float, default=0.0, help="Override infer.iso_threshold")
    parser.add_argument("--query-batch-size", type=int, default=0, help="Override infer.query_batch_size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.run_name:
        cfg.infer.run_name = args.run_name
    if args.device:
        cfg.infer.device = args.device
    if args.grid_resolution > 0:
        cfg.infer.grid_resolution = args.grid_resolution
    if args.iso_threshold > 0:
        cfg.infer.iso_threshold = args.iso_threshold
    if args.query_batch_size > 0:
        cfg.infer.query_batch_size = args.query_batch_size
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
    bound = float(cfg.data.world_bound)
    res = int(cfg.infer.grid_resolution)
    query_bs = int(cfg.infer.query_batch_size)

    axis = torch.linspace(-bound, bound, res)
    mesh = torch.meshgrid(axis, axis, axis, indexing="ij")
    grid_points = torch.stack(mesh, dim=-1).reshape(-1, 3)

    sdf_chunks = []
    unc_chunks = []
    with torch.no_grad():
        for i in tqdm(range(0, grid_points.size(0), query_bs), desc="Querying implicit field"):
            chunk = grid_points[i : i + query_bs].to(device)
            out = model(chunk)
            sdf_chunks.append(out["sdf"].cpu())
            unc_chunks.append(out["uncertainty"].cpu())

    sdf = torch.cat(sdf_chunks, dim=0).squeeze(-1)
    uncertainty = torch.cat(unc_chunks, dim=0).squeeze(-1)

    mask = sdf.abs() <= cfg.infer.iso_threshold
    surface_points = grid_points[mask]
    surface_uncertainty = uncertainty[mask]

    sdf_grid = sdf.reshape(res, res, res).numpy()
    np.save(os.path.join(run_dir, f"{cfg.infer.output_name}_sdf_grid.npy"), sdf_grid)
    np.save(os.path.join(run_dir, f"{cfg.infer.output_name}_points.npy"), surface_points.numpy())

    ply_path = os.path.join(run_dir, f"{cfg.infer.output_name}.ply")
    save_point_cloud(ply_path, surface_points.numpy(), surface_uncertainty.numpy())

    logger.info("Grid points: %d", grid_points.size(0))
    logger.info("Extracted near-surface points: %d", surface_points.size(0))
    logger.info("Saved outputs to: %s", run_dir)


if __name__ == "__main__":
    main()
