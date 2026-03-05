from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, List

import yaml


@dataclass
class DataConfig:
    mode: str = "dummy"
    point_cloud_path: str = ""
    dataset_root: str = ""
    dataset_name: str = ""
    dataset_split: str = "train"
    dataset_val_split: str = "val"
    dataset_categories: List[str] = field(default_factory=list)
    dataset_start: int = 0
    dataset_take: int = -1
    dataset_val_start: int = 0
    dataset_val_take: int = -1
    dataset_cache_size: int = 8
    voxel_size: float = 0.0
    block_size: float = 0.5
    overlap_ratio: float = 0.25
    estimate_normals: bool = False
    normal_k: int = 16
    fill_empty_normals: bool = True
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
    use_point_geo: bool = True
    point_geo_k: int = 16
    point_geo_dim: int = 16
    point_geo_hidden_dim: int = 32
    point_geo_max_context: int = 4096
    use_active_refine: bool = True
    active_min_points: int = 8
    active_complexity_quantile: float = 0.6
    active_min_complexity: float = 1e-4
    hidden_dim: int = 96
    decoder_layers: int = 4
    max_candidates: int = 8


@dataclass
class LossConfig:
    surface: float = 1.0
    sign: float = 1.0
    eikonal: float = 0.1
    seam: float = 0.2
    seam_grad: float = 0.05
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
    val_every: int = 1
    val_max_batches: int = 0
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
