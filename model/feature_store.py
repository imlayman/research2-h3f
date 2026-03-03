from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Iterable, List, Tuple

import torch
import torch.nn as nn

from model.spatial_index import TwoLevelSpatialIndex


CellKey = Tuple[int, int, int]


class SparseFeatureStore(nn.Module):
    def __init__(
        self,
        cell_size_coarse: float,
        cell_size_fine: float,
        feat_dim_coarse: int,
        feat_dim_fine: int,
        use_half: bool = False,
        output_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if feat_dim_coarse <= 0 or feat_dim_fine <= 0:
            raise ValueError("feature dims must be positive")

        self.spatial_index = TwoLevelSpatialIndex(
            cell_size_coarse=cell_size_coarse,
            cell_size_fine=cell_size_fine,
        )
        self.feat_dim_coarse = int(feat_dim_coarse)
        self.feat_dim_fine = int(feat_dim_fine)
        self.use_half = bool(use_half)
        self.output_dtype = output_dtype

        self.coarse_map: Dict[CellKey, int] = {}
        self.fine_map: Dict[CellKey, int] = {}

        self.coarse_embedding = self._create_embedding(1, self.feat_dim_coarse)
        self.fine_embedding = self._create_embedding(1, self.feat_dim_fine)

    @property
    def total_dim(self) -> int:
        return self.feat_dim_coarse + self.feat_dim_fine

    def _create_embedding(self, num_embeddings: int, feat_dim: int) -> nn.Embedding:
        emb = nn.Embedding(max(1, int(num_embeddings)), int(feat_dim))
        nn.init.normal_(emb.weight, mean=0.0, std=0.01)
        if self.use_half:
            emb.weight.data = emb.weight.data.half()
        return emb

    @staticmethod
    def _to_unique_keys(cell_keys: Iterable[Iterable[int]]) -> List[CellKey]:
        ordered = OrderedDict()
        for key in cell_keys:
            key_tuple = TwoLevelSpatialIndex.cell_to_key(key)
            if key_tuple not in ordered:
                ordered[key_tuple] = None
        return list(ordered.keys())

    def set_active_cells(self, level: str, cell_keys: Iterable[Iterable[int]]) -> None:
        keys = self._to_unique_keys(cell_keys)
        num = max(1, len(keys))

        if level == "coarse":
            self.coarse_map = {k: i for i, k in enumerate(keys)}
            self.coarse_embedding = self._create_embedding(num, self.feat_dim_coarse)
            return
        if level == "fine":
            self.fine_map = {k: i for i, k in enumerate(keys)}
            self.fine_embedding = self._create_embedding(num, self.feat_dim_fine)
            return
        raise ValueError("level must be 'coarse' or 'fine'")

    def set_cell_feature(self, level: str, cell_key: Iterable[int], feature: torch.Tensor) -> None:
        key = TwoLevelSpatialIndex.cell_to_key(cell_key)
        feat = feature.detach().reshape(-1)

        if level == "coarse":
            index = self.coarse_map.get(key)
            if index is None:
                raise KeyError(f"coarse cell not active: {key}")
            expected_dim = self.feat_dim_coarse
            emb = self.coarse_embedding
        elif level == "fine":
            index = self.fine_map.get(key)
            if index is None:
                raise KeyError(f"fine cell not active: {key}")
            expected_dim = self.feat_dim_fine
            emb = self.fine_embedding
        else:
            raise ValueError("level must be 'coarse' or 'fine'")

        if feat.numel() != expected_dim:
            raise ValueError(f"feature dim mismatch: expected {expected_dim}, got {feat.numel()}")

        feat = feat.to(device=emb.weight.device, dtype=emb.weight.dtype)
        with torch.no_grad():
            emb.weight[index].copy_(feat)

    def _query_one_level(
        self,
        level: str,
        x: torch.Tensor,
    ) -> torch.Tensor:
        batch = x.shape[0]
        if level == "coarse":
            key_map = self.coarse_map
            emb = self.coarse_embedding
            feat_dim = self.feat_dim_coarse
        elif level == "fine":
            key_map = self.fine_map
            emb = self.fine_embedding
            feat_dim = self.feat_dim_fine
        else:
            raise ValueError("level must be 'coarse' or 'fine'")

        out = torch.zeros((batch, feat_dim), device=x.device, dtype=self.output_dtype)
        if not key_map:
            return out

        keys = self.spatial_index.world_to_keys(level=level, x=x)
        active_pos: List[int] = []
        active_indices: List[int] = []
        for i, key in enumerate(keys):
            index = key_map.get(key)
            if index is not None:
                active_pos.append(i)
                active_indices.append(index)

        if not active_indices:
            return out

        emb_indices = torch.tensor(active_indices, dtype=torch.long, device=emb.weight.device)
        feat = emb(emb_indices)
        # Keep retrieval stable when embedding weights are fp16.
        feat = feat.to(dtype=torch.float32)
        feat = feat.to(device=x.device, dtype=self.output_dtype)

        pos_tensor = torch.tensor(active_pos, dtype=torch.long, device=x.device)
        out[pos_tensor] = feat
        return out

    def query_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError("x must have shape [B, 3]")

        coarse_feat = self._query_one_level(level="coarse", x=x)
        fine_feat = self._query_one_level(level="fine", x=x)
        return torch.cat([coarse_feat, fine_feat], dim=-1)

