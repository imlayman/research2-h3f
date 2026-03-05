from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PointGeometryEncoder(nn.Module):
    """Point-level local geometry encoding from KNN context."""

    def __init__(
        self,
        k_neighbors: int = 16,
        out_dim: int = 16,
        hidden_dim: int = 32,
        max_context_points: int = 4096,
    ) -> None:
        super().__init__()
        if k_neighbors <= 0:
            raise ValueError("k_neighbors must be > 0")
        if out_dim <= 0:
            raise ValueError("out_dim must be > 0")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be > 0")
        if max_context_points <= 0:
            raise ValueError("max_context_points must be > 0")

        self.k_neighbors = int(k_neighbors)
        self.max_context_points = int(max_context_points)
        self.out_dim = int(out_dim)

        # [mean_d, std_d, inv_d, curvature, anisotropy, normal_consistency, plane_residual, max_d]
        self.raw_dim = 8
        self.net = nn.Sequential(
            nn.Linear(self.raw_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
            nn.SiLU(),
        )

    def _subsample_context(
        self,
        context_points: torch.Tensor,
        context_normals: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if context_points.size(0) <= self.max_context_points:
            return context_points, context_normals

        idx = torch.randperm(context_points.size(0), device=context_points.device)[: self.max_context_points]
        points = context_points[idx]
        normals = None if context_normals is None else context_normals[idx]
        return points, normals

    @staticmethod
    def _gather_neighbors(values: torch.Tensor, knn_idx: torch.Tensor) -> torch.Tensor:
        n, _, c = values.shape
        k = knn_idx.size(1)
        idx_expand = knn_idx.unsqueeze(-1).expand(n, k, c)
        return torch.gather(values, dim=1, index=idx_expand)

    @staticmethod
    def _safe_eigvalsh(cov: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
        cov_f32 = cov.float()
        if cov_f32.size(0) == 0:
            return cov_f32.new_empty((0, 3))

        eigvals_parts = []
        for start in range(0, cov_f32.size(0), max(1, int(chunk_size))):
            part = cov_f32[start : start + max(1, int(chunk_size))]
            try:
                eig_part = torch.linalg.eigvalsh(part)
            except RuntimeError as exc:
                # Some CUDA environments fail on large batched eigvalsh; fall back to CPU for robustness.
                if part.is_cuda and "cusolver" in str(exc).lower():
                    eig_part = torch.linalg.eigvalsh(part.cpu()).to(device=part.device)
                else:
                    raise
            eigvals_parts.append(eig_part)
        return torch.cat(eigvals_parts, dim=0)

    def forward(
        self,
        query_points: torch.Tensor,
        context_points: Optional[torch.Tensor] = None,
        context_normals: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if query_points.ndim != 2 or query_points.shape[1] != 3:
            raise ValueError("query_points must have shape [N, 3]")

        n = query_points.size(0)
        if n == 0:
            return torch.zeros((0, self.out_dim), device=query_points.device, dtype=query_points.dtype)

        if context_points is None or context_points.numel() == 0:
            raw = torch.zeros((n, self.raw_dim), device=query_points.device, dtype=query_points.dtype)
            return self.net(raw)

        if context_points.ndim != 2 or context_points.shape[1] != 3:
            raise ValueError("context_points must have shape [M, 3]")
        if context_normals is not None and (context_normals.ndim != 2 or context_normals.shape != context_points.shape):
            context_normals = None

        context_points, context_normals = self._subsample_context(context_points, context_normals)

        m = context_points.size(0)
        if m == 0:
            raw = torch.zeros((n, self.raw_dim), device=query_points.device, dtype=query_points.dtype)
            return self.net(raw)

        k = min(self.k_neighbors, m)
        dist = torch.cdist(query_points, context_points)  # [N, M]
        knn_dist, knn_idx = torch.topk(dist, k=k, dim=1, largest=False, sorted=False)

        expanded_context = context_points.unsqueeze(0).expand(n, m, 3)
        neighbors = self._gather_neighbors(expanded_context, knn_idx)  # [N, K, 3]
        rel = neighbors - query_points.unsqueeze(1)

        mean_d = knn_dist.mean(dim=1, keepdim=True)
        std_d = knn_dist.std(dim=1, unbiased=False, keepdim=True)
        inv_d = 1.0 / (mean_d + 1e-6)
        max_d = knn_dist.max(dim=1, keepdim=True)[0]

        centered = rel - rel.mean(dim=1, keepdim=True)
        cov = torch.matmul(centered.transpose(1, 2), centered) / float(max(k, 1))
        # eigvalsh is not available for float16 and can be unstable for very large CUDA batches.
        eigvals = self._safe_eigvalsh(cov).clamp_min(1e-8).to(cov.dtype)
        curvature = eigvals[:, 0:1] / (eigvals.sum(dim=1, keepdim=True) + 1e-6)
        anisotropy = eigvals[:, 2:3] / (eigvals[:, 1:2] + 1e-6)

        if context_normals is not None:
            expanded_normals = context_normals.unsqueeze(0).expand(n, m, 3)
            neighbor_normals = self._gather_neighbors(expanded_normals, knn_idx)
            normal_consistency = neighbor_normals.mean(dim=1).norm(dim=1, keepdim=True)
            plane_residual = torch.abs((rel * neighbor_normals).sum(dim=-1)).mean(dim=1, keepdim=True)
        else:
            normal_consistency = torch.zeros_like(mean_d)
            plane_residual = torch.zeros_like(mean_d)

        raw = torch.cat(
            [
                mean_d,
                std_d,
                inv_d,
                curvature,
                anisotropy,
                normal_consistency,
                plane_residual,
                max_d,
            ],
            dim=1,
        )
        return self.net(raw)
