from __future__ import annotations

import math

import torch
import torch.nn as nn


class FourierPositionalEncoding(nn.Module):
    def __init__(
        self,
        num_frequencies: int = 6,
        include_input: bool = True,
        frequency_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if num_frequencies < 0:
            raise ValueError("num_frequencies must be >= 0")
        if frequency_scale <= 0.0:
            raise ValueError("frequency_scale must be > 0")

        self.num_frequencies = int(num_frequencies)
        self.include_input = bool(include_input)
        self.frequency_scale = float(frequency_scale)

        if self.num_frequencies > 0:
            bands = (2.0 ** torch.arange(self.num_frequencies, dtype=torch.float32)) * (
                math.pi * self.frequency_scale
            )
        else:
            bands = torch.empty((0,), dtype=torch.float32)
        self.register_buffer("freq_bands", bands, persistent=False)

    @property
    def out_dim(self) -> int:
        base = 3 if self.include_input else 0
        return base + 3 * 2 * self.num_frequencies

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 3:
            raise ValueError("x must have last dim = 3")

        original_shape = x.shape[:-1]
        flat = x.reshape(-1, 3).to(torch.float32)

        parts = []
        if self.include_input:
            parts.append(flat)

        if self.num_frequencies > 0:
            projected = flat.unsqueeze(1) * self.freq_bands.view(1, -1, 1)
            parts.append(torch.sin(projected).reshape(flat.shape[0], -1))
            parts.append(torch.cos(projected).reshape(flat.shape[0], -1))

        encoded = torch.cat(parts, dim=-1) if parts else torch.empty((flat.shape[0], 0), device=flat.device)
        return encoded.reshape(*original_shape, -1)

