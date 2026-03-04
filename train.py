import argparse
import os

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from h3f_recon.config import load_config, save_config
from h3f_recon.data import DummyPointCloudDataset
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

    run_dir = create_run_dir(cfg.output_root, cfg.train.run_name, prefix="train")
    logger = setup_logger("h3f_train", os.path.join(run_dir, "train.log"))
    save_config(cfg, os.path.join(run_dir, "resolved_config.yaml"))

    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)
    logger.info("Run dir: %s", run_dir)
    logger.info("Using device: %s", device)

    if cfg.data.mode != "dummy":
        raise ValueError("This MVP trainer currently supports only data.mode=dummy")

    train_dataset = DummyPointCloudDataset(cfg.data, split="train", base_seed=cfg.train.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
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
        "train setup | surf=%d near=%d eps=%.4f near_loss=%s seam=%d seam_block=%.3f overlap=%.2f eik=%d amp=%s",
        surf_sample_count,
        near_sample_count,
        near_epsilon,
        near_loss_type,
        cfg.train.seam_sample_count,
        seam_block_size,
        seam_overlap_ratio,
        cfg.train.eikonal_sample_count,
        amp_enabled,
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
