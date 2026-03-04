from __future__ import annotations

import torch
import torch.nn as nn

from model.posenc import FourierPositionalEncoding


class SharedImplicitDecoder(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_frequencies: int = 6,
        hidden_dim: int = 96,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        if feat_dim <= 0:
            raise ValueError("feat_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.feat_dim = int(feat_dim)
        self.posenc = FourierPositionalEncoding(num_frequencies=num_frequencies, include_input=True)

        in_dim = self.posenc.out_dim + self.feat_dim
        layers = []
        current_dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_rel: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        if x_rel.ndim != 3 or x_rel.shape[-1] != 3:
            raise ValueError("x_rel must have shape [B, K, 3]")
        if feat.ndim != 3:
            raise ValueError("feat must have shape [B, K, C]")
        if x_rel.shape[:2] != feat.shape[:2]:
            raise ValueError("x_rel and feat must align on [B, K]")
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(f"feat last dim must be {self.feat_dim}")

        encoded = self.posenc(x_rel)
        decoder_input = torch.cat([encoded, feat.to(torch.float32)], dim=-1)

        b, k, d = decoder_input.shape
        flat = decoder_input.reshape(b * k, d)
        sdf = self.net(flat).reshape(b, k, 1)
        return sdf

