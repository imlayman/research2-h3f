from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from model.decoder import SharedImplicitDecoder
from model.posenc import FourierPositionalEncoding


class BlendNetwork(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        num_frequencies: int = 6,
        hidden_dim: int = 64,
        num_layers: int = 3,
        use_confidence: bool = True,
    ) -> None:
        super().__init__()
        if feat_dim <= 0:
            raise ValueError("feat_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        self.feat_dim = int(feat_dim)
        self.use_confidence = bool(use_confidence)
        self.posenc = FourierPositionalEncoding(num_frequencies=num_frequencies, include_input=True)

        in_dim = self.posenc.out_dim + self.feat_dim + (1 if self.use_confidence else 0)
        layers = []
        current_dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        x_rel: torch.Tensor,
        feat: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if x_rel.ndim != 3 or x_rel.shape[-1] != 3:
            raise ValueError("x_rel must have shape [B, K, 3]")
        if feat.ndim != 3:
            raise ValueError("feat must have shape [B, K, C]")
        if x_rel.shape[:2] != feat.shape[:2]:
            raise ValueError("x_rel and feat must align on [B, K]")
        if feat.shape[-1] != self.feat_dim:
            raise ValueError(f"feat last dim must be {self.feat_dim}")

        b, k, _ = x_rel.shape
        encoded = self.posenc(x_rel)
        parts = [encoded, feat.to(torch.float32)]

        if self.use_confidence:
            if confidence is None:
                confidence = torch.ones((b, k, 1), device=x_rel.device, dtype=torch.float32)
            else:
                if confidence.shape != (b, k, 1):
                    raise ValueError("confidence must have shape [B, K, 1]")
                confidence = confidence.to(torch.float32)
            parts.append(confidence)

        blend_input = torch.cat(parts, dim=-1)
        flat = blend_input.reshape(b * k, blend_input.shape[-1])
        logits = self.net(flat).reshape(b, k, 1)

        weights = torch.softmax(logits, dim=1)
        return logits, weights

    @staticmethod
    def fuse_sdf(sdf_i: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        if sdf_i.shape != weights.shape:
            raise ValueError("sdf_i and weights must have same shape [B, K, 1]")
        return (sdf_i * weights).sum(dim=1)


class ImplicitFieldWithBlending(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        posenc_frequencies: int = 6,
        decoder_hidden_dim: int = 96,
        decoder_layers: int = 4,
        blend_hidden_dim: int = 64,
        blend_layers: int = 3,
        use_confidence: bool = True,
    ) -> None:
        super().__init__()
        self.decoder = SharedImplicitDecoder(
            feat_dim=feat_dim,
            num_frequencies=posenc_frequencies,
            hidden_dim=decoder_hidden_dim,
            num_layers=decoder_layers,
        )
        self.blend = BlendNetwork(
            feat_dim=feat_dim,
            num_frequencies=posenc_frequencies,
            hidden_dim=blend_hidden_dim,
            num_layers=blend_layers,
            use_confidence=use_confidence,
        )

    def forward(
        self,
        x_world: torch.Tensor,
        x_rel: torch.Tensor,
        feat: torch.Tensor,
        confidence: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if x_world.ndim != 2 or x_world.shape[-1] != 3:
            raise ValueError("x_world must have shape [B, 3]")
        if x_rel.shape[0] != x_world.shape[0]:
            raise ValueError("x_world and x_rel must align on batch size")

        sdf_i = self.decoder(x_rel=x_rel, feat=feat)
        logits, weights = self.blend(x_rel=x_rel, feat=feat, confidence=confidence)
        sdf = self.blend.fuse_sdf(sdf_i=sdf_i, weights=weights)
        return {"sdf_i": sdf_i, "logits": logits, "weights": weights, "sdf": sdf}

