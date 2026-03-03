from __future__ import annotations

import os
import random
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def create_run_dir(root: str, run_name: str, prefix: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(root, f"{prefix}_{run_name}_{timestamp}")
    os.makedirs(path, exist_ok=False)
    return path


def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    output: Dict[str, Any] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            output[key] = value.to(device)
        else:
            output[key] = value
    return output
