from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List

import yaml


@dataclass
class DataConfig:
    mode: str = "dummy"
    num_train_samples: int = 128
    num_val_samples: int = 16
    batch_size: int = 4
    num_workers: int = 0
    points_per_shape: int = 2048
    surface_sample_count: int = 512
    near_surface_sample_count: int = 512
    eikonal_sample_count: int = 256
    noise_std: float = 0.01
    near_surface_offset: float = 0.03
    world_bound: float = 1.0


@dataclass
class ModelConfig:
    resolutions: List[int] = field(default_factory=lambda: [16, 32, 64])
    cells_per_level: List[int] = field(default_factory=lambda: [2048, 4096, 8192])
    feature_dims: List[int] = field(default_factory=lambda: [8, 8, 8])
    top_k: int = 3
    num_frequencies: int = 6
    hidden_dim: int = 96
    decoder_layers: int = 4
    coarse_hidden_dim: int = 64
    coarse_layers: int = 3
    max_candidates: int = 8


@dataclass
class LossConfig:
    surface: float = 1.0
    sign: float = 1.0
    eikonal: float = 0.1
    seam: float = 0.2
    seam_grad: float = 0.05
    coarse_global: float = 0.1
    blend_smooth: float = 0.01


@dataclass
class TrainConfig:
    seed: int = 42
    run_name: str = "h3f_mvp"
    device: str = "auto"
    epochs: int = 5
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_interval: int = 10
    save_every: int = 1
    resume: str = ""
    seam_sample_count: int = 128
    eikonal_sample_count: int = 128
    loss: LossConfig = field(default_factory=LossConfig)


@dataclass
class InferConfig:
    checkpoint: str = ""
    run_name: str = "h3f_mvp_infer"
    device: str = "auto"
    grid_resolution: int = 48
    query_batch_size: int = 65536
    iso_threshold: float = 0.02
    output_name: str = "recon_surface"


@dataclass
class ExperimentConfig:
    output_root: str = "runs"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    infer: InferConfig = field(default_factory=InferConfig)


def _merge_dataclass(target: Any, updates: Dict[str, Any], prefix: str = "") -> None:
    for key, value in updates.items():
        if not hasattr(target, key):
            location = f"{prefix}.{key}" if prefix else key
            raise KeyError(f"Unknown config key: {location}")

        current = getattr(target, key)
        location = f"{prefix}.{key}" if prefix else key
        if is_dataclass(current):
            if not isinstance(value, dict):
                raise TypeError(f"Expected dict for nested config: {location}")
            _merge_dataclass(current, value, prefix=location)
        else:
            setattr(target, key, value)


def load_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    cfg = ExperimentConfig()
    _merge_dataclass(cfg, loaded)
    return cfg


def config_to_dict(cfg: ExperimentConfig) -> Dict[str, Any]:
    return asdict(cfg)


def save_config(cfg: ExperimentConfig, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False, allow_unicode=True)
