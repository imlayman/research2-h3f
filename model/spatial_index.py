from __future__ import annotations

from typing import Iterable, List, Tuple

import torch


class TwoLevelSpatialIndex:
    def __init__(
        self,
        cell_size_coarse: float,
        cell_size_fine: float,
        origin: Iterable[float] = (0.0, 0.0, 0.0),
    ) -> None:
        if cell_size_coarse <= 0.0 or cell_size_fine <= 0.0:
            raise ValueError("cell sizes must be positive")

        origin_list = list(origin)
        if len(origin_list) != 3:
            raise ValueError("origin must contain three values")

        self.cell_size_coarse = float(cell_size_coarse)
        self.cell_size_fine = float(cell_size_fine)
        self.origin = torch.tensor(origin_list, dtype=torch.float32)

    def _resolve_cell_size(self, level: str) -> float:
        if level == "coarse":
            return self.cell_size_coarse
        if level == "fine":
            return self.cell_size_fine
        raise ValueError("level must be 'coarse' or 'fine'")

    def world_to_cell(self, level: str, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("x must have shape [B, 3]")

        cell_size = self._resolve_cell_size(level)
        origin = self.origin.to(device=x.device, dtype=x.dtype)
        return torch.floor((x - origin) / cell_size).to(torch.long)

    @staticmethod
    def cell_to_key(cell_coord: Iterable[int]) -> Tuple[int, int, int]:
        values = list(cell_coord)
        if len(values) != 3:
            raise ValueError("cell_coord must contain three values")
        return (int(values[0]), int(values[1]), int(values[2]))

    def world_to_keys(self, level: str, x: torch.Tensor) -> List[Tuple[int, int, int]]:
        cell_coords = self.world_to_cell(level=level, x=x).detach().cpu().tolist()
        return [self.cell_to_key(coord) for coord in cell_coords]

