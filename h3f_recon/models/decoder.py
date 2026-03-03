import torch
import torch.nn as nn


class MLPBackbone(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        layers = []
        current_dim = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.SiLU())
            current_dim = hidden_dim

        self.net = nn.Sequential(*layers)
        self.out_dim = current_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LocalFusionDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        self.backbone = MLPBackbone(in_dim, hidden_dim, num_layers)
        self.sdf_head = nn.Linear(self.backbone.out_dim, 1)
        self.blend_head = nn.Linear(self.backbone.out_dim, 1)
        self.uncertainty_head = nn.Linear(self.backbone.out_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if x.ndim != 3:
            raise ValueError("Local decoder input must be [N, K, D]")

        n, k, d = x.shape
        flat = x.reshape(n * k, d)
        hidden = self.backbone(flat)

        sdf = self.sdf_head(hidden).reshape(n, k, 1)
        blend = self.blend_head(hidden).reshape(n, k, 1)
        uncertainty = self.uncertainty_head(hidden).reshape(n, k, 1)
        return sdf, blend, uncertainty


class CoarseField(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("num_layers must be >= 2")

        layers = []
        in_dim = 3
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        return self.net(points)
