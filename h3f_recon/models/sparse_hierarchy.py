from __future__ import annotations

from typing import Dict, List, Optional

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
        use_active_refine: bool = True,
        active_min_points: int = 8,
        active_complexity_quantile: float = 0.6,
        active_min_complexity: float = 1e-4,
    ) -> None:
        super().__init__()
        if not (len(resolutions) == len(cells_per_level) == len(feature_dims)):
            raise ValueError("resolutions/cells_per_level/feature_dims must have same length")

        self.resolutions = resolutions
        self.cells_per_level = cells_per_level
        self.feature_dims = feature_dims
        self.world_bound = world_bound
        self.max_candidates = max(1, min(max_candidates, 8))
        self.use_active_refine = bool(use_active_refine)
        self.active_min_points = int(max(1, active_min_points))
        self.active_complexity_quantile = float(min(max(active_complexity_quantile, 0.0), 1.0))
        self.active_min_complexity = float(max(0.0, active_min_complexity))

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

    @staticmethod
    def _coord_keys(coords: torch.Tensor) -> torch.Tensor:
        primes = torch.tensor([73856093, 19349663, 83492791], device=coords.device, dtype=torch.long)
        return (coords.long() * primes).sum(dim=-1)

    def _world_to_grid(self, points: torch.Tensor, resolution: int) -> torch.Tensor:
        normalized = (points + self.world_bound) / (2.0 * self.world_bound)
        coords = torch.floor(normalized * resolution).long()
        return coords.clamp(0, resolution - 1)

    @staticmethod
    def _gather(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
        expand_idx = indices.unsqueeze(-1).expand(-1, -1, values.size(-1))
        return torch.gather(values, 1, expand_idx)

    def _compute_level_stats(
        self,
        context_points: torch.Tensor,
        resolution: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        coords = self._world_to_grid(context_points, resolution)
        unique, inverse, counts = torch.unique(coords, dim=0, return_inverse=True, return_counts=True)

        if unique.numel() == 0:
            empty = torch.zeros((0,), device=context_points.device, dtype=context_points.dtype)
            return unique, counts, empty

        num_unique = unique.size(0)
        point_sum = torch.zeros((num_unique, 3), device=context_points.device, dtype=context_points.dtype)
        point_sum.index_add_(0, inverse, context_points)
        means = point_sum / counts.to(context_points.dtype).unsqueeze(1)

        centered = context_points - means[inverse]
        sq_dist = centered.pow(2).sum(dim=1)
        var_acc = torch.zeros((num_unique,), device=context_points.device, dtype=context_points.dtype)
        var_acc.index_add_(0, inverse, sq_dist)
        complexity = var_acc / counts.to(context_points.dtype).clamp_min(1)
        return unique, counts, complexity

    def _build_active_tables(self, context_points: Optional[torch.Tensor]) -> list[Optional[torch.Tensor]]:
        if not self.use_active_refine or context_points is None or context_points.numel() == 0:
            return [None for _ in self.resolutions]
        if context_points.ndim != 2 or context_points.shape[1] != 3:
            return [None for _ in self.resolutions]

        tables: list[Optional[torch.Tensor]] = []
        prev_active_keys: Optional[torch.Tensor] = None
        prev_res: Optional[int] = None

        for level, (res, cells) in enumerate(zip(self.resolutions, self.cells_per_level)):
            coords, counts, complexity = self._compute_level_stats(context_points=context_points, resolution=res)
            table = torch.zeros((cells,), device=context_points.device, dtype=torch.bool)

            if coords.numel() == 0:
                tables.append(table)
                prev_active_keys = None
                prev_res = res
                continue

            if level == 0:
                active_mask = counts >= self.active_min_points
                if not torch.any(active_mask):
                    active_mask = counts > 0
            else:
                assert prev_res is not None
                if prev_active_keys is None or prev_active_keys.numel() == 0:
                    active_mask = counts >= self.active_min_points
                    if not torch.any(active_mask):
                        active_mask = counts > 0
                else:
                    parent_coords = torch.div(coords * prev_res, res, rounding_mode="floor")
                    parent_keys = self._coord_keys(parent_coords)
                    parent_active = torch.isin(parent_keys, prev_active_keys)

                    base_mask = parent_active & (counts > 0)
                    if torch.any(base_mask):
                        valid_complexity = complexity[base_mask]
                        q = torch.quantile(valid_complexity, self.active_complexity_quantile)
                        complexity_thr = torch.maximum(
                            q,
                            torch.tensor(self.active_min_complexity, device=complexity.device, dtype=complexity.dtype),
                        )
                        active_mask = base_mask & (complexity >= complexity_thr)
                        if not torch.any(active_mask):
                            active_mask = base_mask
                    else:
                        active_mask = counts >= self.active_min_points
                        if not torch.any(active_mask):
                            active_mask = counts > 0

            active_coords = coords[active_mask]
            if active_coords.numel() == 0:
                active_coords = coords

            table[self._hash_coords(active_coords, cells)] = True
            tables.append(table)
            prev_active_keys = self._coord_keys(active_coords)
            prev_res = res

        return tables

    def forward(
        self,
        points: torch.Tensor,
        top_k: int,
        context_points: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if points.ndim != 2 or points.size(-1) != 3:
            raise ValueError("points must be [N, 3]")

        active_tables = self._build_active_tables(context_points)
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
            level_active = active_tables[level]
            if level_active is not None:
                active_mask = level_active[flat_index].unsqueeze(-1).to(feat.dtype)
                feat = feat * active_mask
            level_features.append(feat)

            if level == len(self.resolutions) - 1:
                finest_quality = torch.sigmoid(self.level_quality[level][flat_index])
                finest_translation = 0.25 * torch.tanh(self.level_translation[level][flat_index])
                if level_active is not None:
                    quality_mask = level_active[flat_index].unsqueeze(-1).to(finest_quality.dtype)
                    finest_quality = finest_quality * quality_mask
                    finest_translation = finest_translation * quality_mask

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
