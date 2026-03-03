from __future__ import annotations

from typing import Dict

import torch
from torch.utils.data import Dataset

from h3f_recon.config import DataConfig


class DummyPointCloudDataset(Dataset):
    def __init__(self, cfg: DataConfig, split: str, base_seed: int = 42) -> None:
        super().__init__()
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        self.cfg = cfg
        self.split = split
        self.base_seed = base_seed
        self.length = cfg.num_train_samples if split == "train" else cfg.num_val_samples

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        seed_shift = 0 if self.split == "train" else 100000
        g = torch.Generator().manual_seed(self.base_seed + seed_shift + index)

        center = torch.rand((3,), generator=g) * 0.6 - 0.3
        radius = torch.rand((1,), generator=g) * 0.25 + 0.35

        dirs = torch.randn((self.cfg.points_per_shape, 3), generator=g)
        dirs = dirs / (dirs.norm(dim=-1, keepdim=True) + 1e-6)

        clean_surface = center.unsqueeze(0) + radius * dirs
        noise = torch.randn(clean_surface.shape, generator=g) * self.cfg.noise_std
        point_cloud = clean_surface + noise

        surf_idx = torch.randint(
            low=0,
            high=self.cfg.points_per_shape,
            size=(self.cfg.surface_sample_count,),
            generator=g,
        )
        surface_points = point_cloud[surf_idx]
        surface_normals = dirs[surf_idx]

        near_idx = torch.randint(
            low=0,
            high=self.cfg.points_per_shape,
            size=(self.cfg.near_surface_sample_count,),
            generator=g,
        )
        near_dirs = dirs[near_idx]
        near_base = center.unsqueeze(0) + radius * near_dirs
        offsets = (
            torch.rand((self.cfg.near_surface_sample_count, 1), generator=g) * 2.0 - 1.0
        ) * self.cfg.near_surface_offset
        near_points = near_base + offsets * near_dirs
        near_sdf = offsets

        eikonal_points = (
            torch.rand((self.cfg.eikonal_sample_count, 3), generator=g) * 2.0 - 1.0
        ) * self.cfg.world_bound

        return {
            "point_cloud": point_cloud.float(),
            "surface_points": surface_points.float(),
            "surface_normals": surface_normals.float(),
            "near_points": near_points.float(),
            "near_sdf": near_sdf.float(),
            "eikonal_points": eikonal_points.float(),
        }
