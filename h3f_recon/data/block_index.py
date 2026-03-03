from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Block:
    block_id: int
    grid_coord: Tuple[int, int, int]
    core_min: np.ndarray
    core_max: np.ndarray
    overlap_min: np.ndarray
    overlap_max: np.ndarray
    num_points: int

    @property
    def center(self) -> np.ndarray:
        return 0.5 * (self.core_min + self.core_max)

    def contains(self, point: np.ndarray, use_overlap: bool = True) -> bool:
        p = np.asarray(point, dtype=np.float32).reshape(3)
        lower = self.overlap_min if use_overlap else self.core_min
        upper = self.overlap_max if use_overlap else self.core_max
        return bool(np.all(p >= lower) and np.all(p <= upper))


class BlockIndex:
    def __init__(self, points: np.ndarray, block_size: float, overlap_ratio: float) -> None:
        points_np = np.asarray(points, dtype=np.float32)
        if points_np.ndim != 2 or points_np.shape[1] != 3:
            raise ValueError("points must have shape [N, 3]")
        if points_np.shape[0] == 0:
            raise ValueError("points cannot be empty")
        if block_size <= 0.0:
            raise ValueError("block_size must be > 0")
        if overlap_ratio < 0.0:
            raise ValueError("overlap_ratio must be >= 0")

        self.points = points_np
        self.block_size = float(block_size)
        self.overlap_ratio = float(overlap_ratio)
        self.overlap_delta = self.block_size * self.overlap_ratio

        self.global_min = self.points.min(axis=0)

        point_coords = self._points_to_grid(self.points)
        unique_coords, inverse, counts = np.unique(
            point_coords,
            axis=0,
            return_inverse=True,
            return_counts=True,
        )

        self.blocks: List[Block] = []
        self.coord_to_block_id: Dict[Tuple[int, int, int], int] = {}
        for block_id, (coord_np, count) in enumerate(zip(unique_coords, counts)):
            coord = (int(coord_np[0]), int(coord_np[1]), int(coord_np[2]))
            core_min = self.global_min + np.asarray(coord, dtype=np.float32) * self.block_size
            core_max = core_min + self.block_size
            overlap_min = core_min - self.overlap_delta
            overlap_max = core_max + self.overlap_delta

            self.blocks.append(
                Block(
                    block_id=block_id,
                    grid_coord=coord,
                    core_min=core_min.astype(np.float32),
                    core_max=core_max.astype(np.float32),
                    overlap_min=overlap_min.astype(np.float32),
                    overlap_max=overlap_max.astype(np.float32),
                    num_points=int(count),
                )
            )
            self.coord_to_block_id[coord] = block_id

        self._core_point_indices: List[List[int]] = [[] for _ in range(len(self.blocks))]
        for point_idx, bid in enumerate(inverse.tolist()):
            self._core_point_indices[int(bid)].append(point_idx)

        self._neighbor_radius = int(np.ceil(self.overlap_ratio))
        self._neighbor_offsets = list(
            product(
                range(-self._neighbor_radius, self._neighbor_radius + 1),
                range(-self._neighbor_radius, self._neighbor_radius + 1),
                range(-self._neighbor_radius, self._neighbor_radius + 1),
            )
        )

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def _points_to_grid(self, points: np.ndarray) -> np.ndarray:
        coords = np.floor((points - self.global_min[None, :]) / self.block_size).astype(np.int64)
        return coords

    def point_to_grid_coord(self, point: np.ndarray) -> Tuple[int, int, int]:
        point_np = np.asarray(point, dtype=np.float32).reshape(1, 3)
        coord = self._points_to_grid(point_np)[0]
        return (int(coord[0]), int(coord[1]), int(coord[2]))

    def find_core_block_id(self, point: np.ndarray) -> Optional[int]:
        return self.coord_to_block_id.get(self.point_to_grid_coord(point))

    def get_core_point_indices(self, block_id: int) -> np.ndarray:
        if block_id < 0 or block_id >= self.num_blocks:
            raise IndexError("block_id out of range")
        return np.asarray(self._core_point_indices[block_id], dtype=np.int64)

    def all_covering_block_ids(self, point: np.ndarray) -> List[int]:
        p = np.asarray(point, dtype=np.float32).reshape(3)
        covering: List[int] = []
        for block in self.blocks:
            if block.contains(p, use_overlap=True):
                covering.append(block.block_id)
        return covering

    def query(self, point: np.ndarray, top_k: Optional[int] = None) -> List[Block]:
        if top_k is not None and top_k <= 0:
            return []

        p = np.asarray(point, dtype=np.float32).reshape(3)
        base_coord = self.point_to_grid_coord(p)

        candidates = []
        for dx, dy, dz in self._neighbor_offsets:
            coord = (base_coord[0] + dx, base_coord[1] + dy, base_coord[2] + dz)
            block_id = self.coord_to_block_id.get(coord)
            if block_id is None:
                continue

            block = self.blocks[block_id]
            if not block.contains(p, use_overlap=True):
                continue

            priority = 0 if block.contains(p, use_overlap=False) else 1
            distance = float(np.linalg.norm(p - block.center))
            candidates.append((priority, distance, block.block_id, block))

        candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        if top_k is not None:
            candidates = candidates[:top_k]

        return [item[3] for item in candidates]

    def query_block_ids(self, point: np.ndarray, top_k: Optional[int] = None) -> List[int]:
        return [block.block_id for block in self.query(point=point, top_k=top_k)]

    def query_batch(self, points: np.ndarray, top_k: Optional[int] = None) -> List[List[Block]]:
        points_np = np.asarray(points, dtype=np.float32)
        if points_np.ndim != 2 or points_np.shape[1] != 3:
            raise ValueError("points must have shape [N, 3]")
        return [self.query(point, top_k=top_k) for point in points_np]
