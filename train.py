import argparse
import os
import math

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from h3f_recon.config import load_config, save_config
from h3f_recon.data import (
    DummyPointCloudDataset,
    PointCloudCollectionTrainDataset,
    PointCloudPreprocessConfig,
    PointCloudTrainDataset,
)
from h3f_recon.engine import compute_training_losses
from h3f_recon.models import H3FRecon
from h3f_recon.utils import (
    create_run_dir,
    load_checkpoint,
    move_batch_to_device,
    resolve_device,
    save_checkpoint,
    set_seed,
    setup_logger,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train H3F-Recon MVP")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--run-name", type=str, default="", help="Override train.run_name")
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path for resume")
    parser.add_argument("--device", type=str, default="", help="Override train.device")
    parser.add_argument("--point-cloud", type=str, default="", help="Override data.point_cloud_path (.ply/.xyz/.pcd/.npz)")
    parser.add_argument("--max-steps", type=int, default=0, help="Train until this global step if > 0")
    parser.add_argument("--amp", action="store_true", help="Enable AMP (CUDA only)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.run_name:
        cfg.train.run_name = args.run_name
    if args.device:
        cfg.train.device = args.device
    if args.resume:
        cfg.train.resume = args.resume
    if args.point_cloud:
        cfg.data.point_cloud_path = args.point_cloud
        cfg.data.mode = "point_cloud"

    run_dir = create_run_dir(cfg.output_root, cfg.train.run_name, prefix="train")
    logger = setup_logger("h3f_train", os.path.join(run_dir, "train.log"))
    save_config(cfg, os.path.join(run_dir, "resolved_config.yaml"))

    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    logger.info("Run dir: %s", run_dir)
    logger.info("Using device: %s", device)

    val_dataset = None
    if cfg.data.mode == "dummy":
        train_dataset = DummyPointCloudDataset(cfg.data, split="train", base_seed=cfg.train.seed)
        val_dataset = DummyPointCloudDataset(cfg.data, split="val", base_seed=cfg.train.seed)
    elif cfg.data.mode in {"point_cloud", "real"}:
        if not cfg.data.point_cloud_path:
            raise ValueError("Please set data.point_cloud_path or pass --point-cloud for real point cloud training")

        preprocess_cfg = PointCloudPreprocessConfig(
            voxel_size=float(cfg.data.voxel_size),
            block_size=float(cfg.data.block_size),
            overlap_ratio=float(cfg.data.overlap_ratio),
            top_k=int(cfg.model.top_k),
            estimate_normals=bool(cfg.data.estimate_normals),
            normal_k=int(cfg.data.normal_k),
            fill_empty_normals=bool(cfg.data.fill_empty_normals),
        )
        train_dataset = PointCloudTrainDataset(
            point_cloud_path=cfg.data.point_cloud_path,
            preprocess_cfg=preprocess_cfg,
            surface_sample_count=cfg.data.surface_sample_count,
            points_per_shape=cfg.data.points_per_shape,
            virtual_length=cfg.data.num_train_samples,
            base_seed=cfg.train.seed,
        )
        val_dataset = PointCloudTrainDataset(
            point_cloud_path=cfg.data.point_cloud_path,
            preprocess_cfg=preprocess_cfg,
            surface_sample_count=cfg.data.surface_sample_count,
            points_per_shape=cfg.data.points_per_shape,
            virtual_length=cfg.data.num_val_samples,
            base_seed=cfg.train.seed + 100000,
        )
        logger.info(
            "Real point cloud mode | path=%s train_blocks=%d train_len=%d val_len=%d voxel=%.4f block=%.4f overlap=%.2f",
            cfg.data.point_cloud_path,
            train_dataset.num_blocks,
            len(train_dataset),
            len(val_dataset),
            cfg.data.voxel_size,
            cfg.data.block_size,
            cfg.data.overlap_ratio,
        )
    elif cfg.data.mode == "dataset":
        if not cfg.data.dataset_root:
            raise ValueError("Please set data.dataset_root for dataset mode")
        if not cfg.data.dataset_name:
            raise ValueError("Please set data.dataset_name for dataset mode")

        preprocess_cfg = PointCloudPreprocessConfig(
            voxel_size=float(cfg.data.voxel_size),
            block_size=float(cfg.data.block_size),
            overlap_ratio=float(cfg.data.overlap_ratio),
            top_k=int(cfg.model.top_k),
            estimate_normals=bool(cfg.data.estimate_normals),
            normal_k=int(cfg.data.normal_k),
            fill_empty_normals=bool(cfg.data.fill_empty_normals),
        )
        train_dataset = PointCloudCollectionTrainDataset(
            dataset_root=cfg.data.dataset_root,
            dataset_name=cfg.data.dataset_name,
            split=cfg.data.dataset_split,
            categories=cfg.data.dataset_categories,
            preprocess_cfg=preprocess_cfg,
            surface_sample_count=cfg.data.surface_sample_count,
            points_per_shape=cfg.data.points_per_shape,
            start=int(cfg.data.dataset_start),
            take=int(cfg.data.dataset_take),
            cache_size=int(cfg.data.dataset_cache_size),
            base_seed=cfg.train.seed,
        )
        val_dataset = PointCloudCollectionTrainDataset(
            dataset_root=cfg.data.dataset_root,
            dataset_name=cfg.data.dataset_name,
            split=cfg.data.dataset_val_split,
            categories=cfg.data.dataset_categories,
            preprocess_cfg=preprocess_cfg,
            surface_sample_count=cfg.data.surface_sample_count,
            points_per_shape=cfg.data.points_per_shape,
            start=int(cfg.data.dataset_val_start),
            take=int(cfg.data.dataset_val_take),
            cache_size=int(cfg.data.dataset_cache_size),
            base_seed=cfg.train.seed + 100000,
        )
        logger.info(
            (
                "Dataset mode | name=%s root=%s train_split=%s train_scenes=%d train_start=%d train_take=%d "
                "val_split=%s val_scenes=%d val_start=%d val_take=%d "
                "voxel=%.4f block=%.4f overlap=%.2f"
            ),
            cfg.data.dataset_name,
            cfg.data.dataset_root,
            cfg.data.dataset_split,
            train_dataset.num_scenes,
            cfg.data.dataset_start,
            cfg.data.dataset_take,
            cfg.data.dataset_val_split,
            val_dataset.num_scenes,
            cfg.data.dataset_val_start,
            cfg.data.dataset_val_take,
            cfg.data.voxel_size,
            cfg.data.block_size,
            cfg.data.overlap_ratio,
        )
    else:
        raise ValueError("Unsupported data.mode: expected 'dummy', 'point_cloud', or 'dataset'")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        drop_last=False,
        pin_memory=(device.type == "cuda"),
    )

    model = H3FRecon(cfg.model, cfg.data).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    # Sampling/loss knobs with backward-compatible defaults.
    surf_sample_count = int(getattr(cfg.train, "surf_sample_count", cfg.data.surface_sample_count))
    near_sample_count = int(getattr(cfg.train, "near_sample_count", cfg.data.near_surface_sample_count))
    near_epsilon = float(getattr(cfg.train, "near_epsilon", cfg.data.near_surface_offset))
    near_loss_type = str(getattr(cfg.train, "near_loss_type", "tsdf")).lower()
    seam_block_size = float(getattr(cfg.train, "seam_block_size", 0.5))
    seam_overlap_ratio = float(getattr(cfg.train, "seam_overlap_ratio", 0.25))
    use_eikonal = bool(getattr(cfg.train, "use_eikonal", True))

    target_steps = args.max_steps if args.max_steps > 0 else None

    amp_enabled = bool(getattr(cfg.train, "use_amp", False) or args.amp)
    if device.type != "cuda":
        amp_enabled = False
    scaler = GradScaler(enabled=amp_enabled)

    logger.info(
        (
            "train setup | surf=%d near=%d eps=%.4f near_loss=%s seam=%d seam_block=%.3f "
            "overlap=%.2f eik=%d amp=%s val_every=%d val_max_batches=%d"
        ),
        surf_sample_count,
        near_sample_count,
        near_epsilon,
        near_loss_type,
        cfg.train.seam_sample_count,
        seam_block_size,
        seam_overlap_ratio,
        cfg.train.eikonal_sample_count,
        amp_enabled,
        int(cfg.train.val_every),
        int(cfg.train.val_max_batches),
    )

    start_epoch = 0
    global_step = 0
    if cfg.train.resume:
        checkpoint = load_checkpoint(cfg.train.resume, model, optimizer, map_location=device)
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("step", 0))
        logger.info("Resumed from %s (epoch=%d, step=%d)", cfg.train.resume, start_epoch, global_step)

    epoch = start_epoch
    stop_flag = False
    while (target_steps is None and epoch < cfg.train.epochs) or (target_steps is not None and global_step < target_steps):
        model.train()
        running_loss = 0.0
        steps_in_epoch = 0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False)
        for batch in progress:
            if target_steps is not None and global_step >= target_steps:
                stop_flag = True
                break

            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=amp_enabled):
                total_loss, stats = compute_training_losses(
                    model=model,
                    batch=batch,
                    loss_cfg=cfg.train.loss,
                    surf_sample_count=surf_sample_count,
                    near_sample_count=near_sample_count,
                    near_epsilon=near_epsilon,
                    near_loss_type=near_loss_type,
                    seam_sample_count=cfg.train.seam_sample_count,
                    seam_block_size=seam_block_size,
                    seam_overlap_ratio=seam_overlap_ratio,
                    eikonal_sample_count=cfg.train.eikonal_sample_count,
                    world_bound=cfg.data.world_bound,
                    use_eikonal=use_eikonal,
                )

            if not torch.isfinite(total_loss):
                bad_terms = [name for name, value in stats.items() if not math.isfinite(float(value))]
                logger.warning(
                    "step=%d non-finite loss detected (total=%s, bad_terms=%s), skip optimizer step",
                    global_step + 1,
                    stats.get("total"),
                    ",".join(bad_terms) if bad_terms else "unknown",
                )
                continue

            if amp_enabled:
                scaler.scale(total_loss).backward()
                if cfg.train.grad_clip > 0.0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                if cfg.train.grad_clip > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
                optimizer.step()

            global_step += 1
            steps_in_epoch += 1
            running_loss += stats["total"]
            progress.set_postfix(loss=f"{stats['total']:.4f}")

            if global_step % cfg.train.log_interval == 0:
                logger.info(
                    "step=%d total=%.4f surf=%.4f near=%.4f seam=%.4f seam_g=%.4f blend=%.4f eik=%.4f",
                    global_step,
                    stats["total"],
                    stats["surface"],
                    stats["near"],
                    stats["seam_value"],
                    stats["seam_grad"],
                    stats["blend_smooth"],
                    stats["eikonal"],
                )

        epoch_loss = running_loss / max(1, steps_in_epoch)
        logger.info("epoch=%d avg_total=%.4f", epoch + 1, epoch_loss)

        should_validate = int(cfg.train.val_every) > 0 and (
            (epoch + 1) % int(cfg.train.val_every) == 0 or stop_flag
        )
        if should_validate:
            model.eval()
            val_running = 0.0
            val_steps = 0
            max_val_batches = int(cfg.train.val_max_batches)
            val_progress = tqdm(val_loader, desc=f"Val {epoch + 1}", leave=False)
            with torch.enable_grad():
                for val_idx, val_batch in enumerate(val_progress):
                    if max_val_batches > 0 and val_idx >= max_val_batches:
                        break

                    val_batch = move_batch_to_device(val_batch, device)
                    with autocast(enabled=amp_enabled):
                        _, val_stats = compute_training_losses(
                            model=model,
                            batch=val_batch,
                            loss_cfg=cfg.train.loss,
                            surf_sample_count=surf_sample_count,
                            near_sample_count=near_sample_count,
                            near_epsilon=near_epsilon,
                            near_loss_type=near_loss_type,
                            seam_sample_count=cfg.train.seam_sample_count,
                            seam_block_size=seam_block_size,
                            seam_overlap_ratio=seam_overlap_ratio,
                            eikonal_sample_count=cfg.train.eikonal_sample_count,
                            world_bound=cfg.data.world_bound,
                            use_eikonal=use_eikonal,
                        )
                    val_running += val_stats["total"]
                    val_steps += 1
                    val_progress.set_postfix(loss=f"{val_stats['total']:.4f}")

            if val_steps > 0:
                val_epoch_loss = val_running / float(val_steps)
                logger.info("epoch=%d val_total=%.4f val_steps=%d", epoch + 1, val_epoch_loss, val_steps)
            else:
                logger.info("epoch=%d validation skipped (no val batches)", epoch + 1)
            model.train()

        if (epoch + 1) % cfg.train.save_every == 0 or stop_flag:
            ckpt_epoch_path = os.path.join(run_dir, f"ckpt_epoch_{epoch + 1:04d}.pt")
            ckpt_latest_path = os.path.join(run_dir, "ckpt_latest.pt")
            save_checkpoint(
                path=ckpt_epoch_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                extra={"run_dir": run_dir},
            )
            save_checkpoint(
                path=ckpt_latest_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
                extra={"run_dir": run_dir},
            )
            logger.info("Saved checkpoint: %s", ckpt_latest_path)

        epoch += 1
        if stop_flag:
            break

    logger.info("Training finished. Final step=%d", global_step)


if __name__ == "__main__":
    main()
