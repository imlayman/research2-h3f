from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn


class SparseFeatureHierarchy(nn.Module):
    def __init__(
        self,
        resolutions: List[int],
        cells_per_level: List[int],
        feature_dims: List[int],
        world_bound: float,
        max_candidates: int,
    ) -> None:
        super().__init__()
        if not (len(resolutions) == len(cells_per_level) == len(feature_dims)):
            raise ValueError("resolutions/cells_per_level/feature_dims must have same length")

        self.resolutions = resolutions
        self.cells_per_level = cells_per_level
        self.feature_dims = feature_dims
        self.world_bound = world_bound
        self.max_candidates = max(1, min(max_candidates, 8))

        self.level_features = nn.ParameterList()
        self.level_quality = nn.ParameterList()
        self.level_translation = nn.ParameterList()

        for cells, dim in zip(cells_per_level, feature_dims):
            self.level_features.append(nn.Parameter(torch.randn(cells, dim) * 0.01))
            self.level_quality.append(nn.Parameter(torch.zeros(cells, 1)))
            self.level_translation.append(nn.Parameter(torch.zeros(cells, 3)))

        offsets = torch.tensor(
            [
                [0, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, -1],
                [1, 1, 1],
            ],
            dtype=torch.long,
        )
        self.register_buffer("candidate_offsets", offsets, persistent=False)

    @staticmethod
    def _hash_coords(coords: torch.Tensor, table_size: int) -> torch.Tensor:
        primes = torch.tensor([73856093, 19349663, 83492791], device=coords.device, dtype=torch.long)
        hashed = ((coords.long() * primes).sum(dim=-1)) % table_size
        return hashed

    def _world_to_grid(self, points: torch.Tensor, resolution: int) -> torch.Tensor:
        normalized = (points + self.world_bound) / (2.0 * self.world_bound)
        coords = torch.floor(normalized * resolution).long()
        return coords.clamp(0, resolution - 1)

    @staticmethod
    def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        expand_idx = indices.unsqueeze(-1).expand(-1, -1, values.size(-1))
        return torch.gather(values, 1, expand_idx)

    def forward(self, points: torch.Tensor, top_k: int) -> Dict[str, torch.Tensor]:
        if points.ndim != 2 or points.size(-1) != 3:
            raise ValueError("points must be [N, 3]")

        finest_res = self.resolutions[-1]
        base_coords = self._world_to_grid(points, finest_res)

        offsets = self.candidate_offsets[: self.max_candidates].to(points.device)
        candidate_coords = base_coords.unsqueeze(1) + offsets.unsqueeze(0)
        candidate_coords = candidate_coords.clamp(0, finest_res - 1)

        level_features = []
        finest_quality = None
        finest_translation = None

        for level, (res, cells) in enumerate(zip(self.resolutions, self.cells_per_level)):
            scaled_coords = torch.div(candidate_coords * res, finest_res, rounding_mode="floor")
            flat_index = self._hash_coords(scaled_coords, cells)

            feat = self.level_features[level][flat_index]
            level_features.append(feat)

            if level == len(self.resolutions) - 1:
                finest_quality = torch.sigmoid(self.level_quality[level][flat_index])
                finest_translation = 0.25 * torch.tanh(self.level_translation[level][flat_index])

        if finest_quality is None or finest_translation is None:
            raise RuntimeError("Failed to initialize finest-level tensors")

        multi_scale_features = torch.cat(level_features, dim=-1)
        k = min(max(1, top_k), multi_scale_features.size(1))

        quality_scores = finest_quality.squeeze(-1)
        _, best_idx = torch.topk(quality_scores, k=k, dim=1)

        selected_features = self._gather(multi_scale_features, best_idx)
        selected_quality = torch.gather(quality_scores, 1, best_idx).unsqueeze(-1)
        selected_translation = self._gather(finest_translation, best_idx)

        selected_coords = self._gather(candidate_coords.float(), best_idx)
        centers = (selected_coords + 0.5) / float(finest_res) * (2.0 * self.world_bound) - self.world_bound
        cell_size = (2.0 * self.world_bound) / float(finest_res)

        local_coords = (points.unsqueeze(1) - centers) / cell_size
        local_coords = local_coords + selected_translation

        return {
            "features": selected_features,
            "quality": selected_quality,
            "local_coords": local_coords,
        }
