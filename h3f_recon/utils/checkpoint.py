import os
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    step: int,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sparse_feature_state: Dict[str, Any] = {}
    if hasattr(model, "sparse_hierarchy"):
        sparse_feature_state = model.sparse_hierarchy.state_dict()

    checkpoint = {
        "epoch": epoch,
        "step": step,
        "model_state": model.state_dict(),
        "sparse_feature_state": sparse_feature_state,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra or {},
    }
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state"], strict=False)
    if "sparse_feature_state" in checkpoint and hasattr(model, "sparse_hierarchy"):
        model.sparse_hierarchy.load_state_dict(checkpoint["sparse_feature_state"], strict=False)

    if optimizer is not None and checkpoint.get("optimizer_state") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])

    return checkpoint
