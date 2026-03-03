import math

import torch
import torch.nn as nn


class FourierPositionalEncoding(nn.Module):
    def __init__(self, num_frequencies: int) -> None:
        super().__init__()
        if num_frequencies < 0:
            raise ValueError("num_frequencies must be >= 0")
        self.num_frequencies = num_frequencies
        if num_frequencies > 0:
            bands = (2.0 ** torch.arange(num_frequencies).float()) * math.pi
        else:
            bands = torch.empty(0)
        self.register_buffer("freq_bands", bands, persistent=False)

    @property
    def out_dim(self) -> int:
        return 3 * (2 * self.num_frequencies + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != 3:
            raise ValueError("Input must have last dimension 3")

        original_shape = x.shape[:-1]
        flat = x.reshape(-1, 3)

        if self.num_frequencies == 0:
            encoded = flat
        else:
            freq = self.freq_bands.view(1, -1, 1)
            projected = flat.unsqueeze(1) * freq
            sin_feat = torch.sin(projected).reshape(flat.shape[0], -1)
            cos_feat = torch.cos(projected).reshape(flat.shape[0], -1)
            encoded = torch.cat([flat, sin_feat, cos_feat], dim=-1)

        return encoded.reshape(*original_shape, -1)
