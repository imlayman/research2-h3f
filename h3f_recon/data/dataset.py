from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from torch.utils.data import Dataset

from h3f_recon.data.block_index import BlockIndex


@dataclass
class PointCloudPreprocessConfig:
    voxel_size: float = 0.0
    block_size: float = 1.0
    overlap_ratio: float = 0.25
    top_k: int = 4
    estimate_normals: bool = False
    normal_k: int = 16
    fill_empty_normals: bool = True


def _normalize_vectors(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, eps, None)


def estimate_normals_knn(points: np.ndarray, k: int = 16) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")

    n = pts.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32)
    if n < 3:
        return np.zeros((n, 3), dtype=np.float32)

    k = int(max(3, min(k, n - 1)))
    normals = np.zeros((n, 3), dtype=np.float32)

    for i in range(n):
        diff = pts - pts[i : i + 1]
        dist2 = np.einsum("ij,ij->i", diff, diff)
        nn_idx = np.argpartition(dist2, kth=k)[: k + 1]
        nn_idx = nn_idx[nn_idx != i]
        if nn_idx.shape[0] < 3:
            continue

        neighbors = pts[nn_idx] - pts[i : i + 1]
        cov = neighbors.T @ neighbors / float(neighbors.shape[0])
        _, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0].astype(np.float32)
        if normal[2] < 0.0:
            normal = -normal
        normals[i] = normal

    return _normalize_vectors(normals).astype(np.float32)


def voxel_downsample(
    points: np.ndarray,
    normals: Optional[np.ndarray],
    voxel_size: float,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    pts = np.asarray(points, dtype=np.float32)
    nrm = None if normals is None else np.asarray(normals, dtype=np.float32)

    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    if nrm is not None and (nrm.ndim != 2 or nrm.shape != pts.shape):
        raise ValueError("normals must have shape [N, 3] and align with points")

    if voxel_size <= 0.0 or pts.shape[0] == 0:
        return pts.copy(), None if nrm is None else nrm.copy()

    origin = pts.min(axis=0)
    voxel_idx = np.floor((pts - origin[None, :]) / float(voxel_size)).astype(np.int64)
    _, inverse = np.unique(voxel_idx, axis=0, return_inverse=True)
    num_voxels = int(inverse.max()) + 1

    point_acc = np.zeros((num_voxels, 3), dtype=np.float64)
    counts = np.zeros((num_voxels,), dtype=np.int64)
    np.add.at(point_acc, inverse, pts)
    np.add.at(counts, inverse, 1)
    points_ds = (point_acc / counts[:, None]).astype(np.float32)

    normals_ds = None
    if nrm is not None:
        normal_acc = np.zeros((num_voxels, 3), dtype=np.float64)
        np.add.at(normal_acc, inverse, nrm)
        normals_ds = _normalize_vectors(normal_acc.astype(np.float32)).astype(np.float32)

    return points_ds, normals_ds


def _load_ascii_ply(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        first = f.readline().strip()
        if first != "ply":
            raise ValueError("Invalid PLY file: missing 'ply' header")

        format_line = ""
        vertex_count: Optional[int] = None
        vertex_props = []
        in_vertex = False

        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PLY file: header not terminated")
            s = line.strip()

            if s.startswith("format"):
                format_line = s
            elif s.startswith("element"):
                parts = s.split()
                if len(parts) >= 3 and parts[1] == "vertex":
                    vertex_count = int(parts[2])
                    vertex_props = []
                    in_vertex = True
                else:
                    in_vertex = False
            elif s.startswith("property") and in_vertex:
                parts = s.split()
                vertex_props.append(parts[-1])
            elif s == "end_header":
                break

        if "ascii" not in format_line.lower():
            raise ValueError("Only ASCII PLY is supported without open3d")
        if vertex_count is None or vertex_count <= 0:
            raise ValueError("Invalid PLY file: missing or invalid vertex count")

        rows = []
        for _ in range(vertex_count):
            line = f.readline()
            if not line:
                break
            parts = line.strip().split()
            if not parts:
                continue
            rows.append([float(x) for x in parts])

    if len(rows) != vertex_count:
        raise ValueError("Invalid PLY file: vertex count does not match data rows")

    data = np.asarray(rows, dtype=np.float32)

    try:
        ix = vertex_props.index("x")
        iy = vertex_props.index("y")
        iz = vertex_props.index("z")
    except ValueError as exc:
        raise ValueError("PLY file must contain x/y/z properties") from exc

    points = data[:, [ix, iy, iz]].astype(np.float32)

    normals = None
    normal_sets = [("nx", "ny", "nz"), ("normal_x", "normal_y", "normal_z")]
    for names in normal_sets:
        if all(name in vertex_props for name in names):
            inx = vertex_props.index(names[0])
            iny = vertex_props.index(names[1])
            inz = vertex_props.index(names[2])
            normals = data[:, [inx, iny, inz]].astype(np.float32)
            normals = _normalize_vectors(normals)
            break

    return points, normals


def _load_ascii_xyz(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    data = np.loadtxt(path, dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 3:
        raise ValueError("XYZ file must have at least 3 columns")

    points = data[:, :3].astype(np.float32)
    normals = None
    if data.shape[1] >= 6:
        normals = _normalize_vectors(data[:, 3:6].astype(np.float32))

    return points, normals


def _load_ascii_pcd(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        fields = None
        data_mode = None

        while True:
            line = f.readline()
            if not line:
                raise ValueError("Invalid PCD file: missing DATA section")
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            up = s.upper()
            if up.startswith("FIELDS"):
                fields = s.split()[1:]
            elif up.startswith("DATA"):
                data_mode = s.split()[1].lower()
                break

        if fields is None:
            raise ValueError("Invalid PCD file: missing FIELDS")
        if data_mode != "ascii":
            raise ValueError("Only ASCII PCD is supported without open3d")

        rows = []
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            rows.append([float(x) for x in s.split()])

    if not rows:
        raise ValueError("PCD contains no points")

    data = np.asarray(rows, dtype=np.float32)

    try:
        ix = fields.index("x")
        iy = fields.index("y")
        iz = fields.index("z")
    except ValueError as exc:
        raise ValueError("PCD file must contain x/y/z fields") from exc

    points = data[:, [ix, iy, iz]].astype(np.float32)

    normals = None
    normal_sets = [("normal_x", "normal_y", "normal_z"), ("nx", "ny", "nz")]
    for names in normal_sets:
        if all(name in fields for name in names):
            inx = fields.index(names[0])
            iny = fields.index(names[1])
            inz = fields.index(names[2])
            normals = data[:, [inx, iny, inz]].astype(np.float32)
            normals = _normalize_vectors(normals)
            break

    return points, normals


def _load_npz_point_cloud(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with np.load(path, allow_pickle=True) as data:
        # Guard against passing occupancy supervision files (e.g. points.npz)
        # where "points" are query samples instead of input point-cloud observations.
        if "occupancies" in data:
            raise ValueError(
                f"NPZ file looks like occupancy samples (has 'occupancies'): {path}. "
                "Please use pointcloud.npz / pointcloud_XX.npz for point cloud input."
            )

        point_keys = ("points", "xyz", "pointcloud")
        points = None
        for key in point_keys:
            if key in data:
                points = np.asarray(data[key], dtype=np.float32)
                break
        if points is None:
            raise ValueError(
                "NPZ point cloud must contain one of keys: " + ", ".join(point_keys)
            )
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("NPZ points must have shape [N, 3]")

        normals = None
        normal_keys = ("normals", "normal")
        for key in normal_keys:
            if key in data:
                candidate = np.asarray(data[key], dtype=np.float32)
                if candidate.ndim != 2 or candidate.shape != points.shape:
                    raise ValueError("NPZ normals must have shape [N, 3] and align with points")
                normals = _normalize_vectors(candidate)
                break

    return points.astype(np.float32), None if normals is None else normals.astype(np.float32)


def _try_load_with_open3d(path: Path) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    try:
        import open3d as o3d  # type: ignore
    except Exception:
        return None

    pcd = o3d.io.read_point_cloud(str(path))
    points = np.asarray(pcd.points, dtype=np.float32)
    if points.shape[0] == 0:
        raise ValueError(f"Failed to load point cloud from: {path}")

    normals = None
    pcd_normals = np.asarray(pcd.normals, dtype=np.float32)
    if pcd_normals.shape == points.shape and points.shape[0] > 0:
        normals = _normalize_vectors(pcd_normals)

    return points, normals


def load_point_cloud(
    file_path: str,
    estimate_normals: bool = False,
    normal_k: int = 16,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {file_path}")

    suffix = path.suffix.lower()

    if suffix == ".npz":
        loaded = _load_npz_point_cloud(path)
    else:
        loaded = _try_load_with_open3d(path)
        if loaded is None:
            if suffix == ".ply":
                loaded = _load_ascii_ply(path)
            elif suffix == ".xyz":
                loaded = _load_ascii_xyz(path)
            elif suffix == ".pcd":
                loaded = _load_ascii_pcd(path)
            else:
                raise ValueError(f"Unsupported format: {suffix}. Supported: .ply/.xyz/.pcd/.npz")

    points, normals = loaded

    if normals is None and estimate_normals:
        normals = estimate_normals_knn(points, k=normal_k)

    return points.astype(np.float32), None if normals is None else normals.astype(np.float32)


class PointCloudBlockDataset(Dataset):
    def __init__(self, point_cloud_path: str, cfg: PointCloudPreprocessConfig) -> None:
        super().__init__()
        self.cfg = cfg

        points, normals = load_point_cloud(
            file_path=point_cloud_path,
            estimate_normals=cfg.estimate_normals,
            normal_k=cfg.normal_k,
        )

        points, normals = voxel_downsample(
            points=points,
            normals=normals,
            voxel_size=cfg.voxel_size,
        )

        if normals is None and cfg.fill_empty_normals:
            normals = np.zeros_like(points, dtype=np.float32)

        self.points = points.astype(np.float32)
        self.normals = None if normals is None else normals.astype(np.float32)

        self.block_index = BlockIndex(
            points=self.points,
            block_size=cfg.block_size,
            overlap_ratio=cfg.overlap_ratio,
        )

    def __len__(self) -> int:
        return self.block_index.num_blocks

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        block = self.block_index.blocks[idx]
        mask = np.all(
            (self.points >= block.overlap_min[None, :]) & (self.points <= block.overlap_max[None, :]),
            axis=1,
        )
        indices = np.where(mask)[0]

        item: Dict[str, np.ndarray] = {
            "block_id": np.asarray([block.block_id], dtype=np.int64),
            "grid_coord": np.asarray(block.grid_coord, dtype=np.int64),
            "core_aabb": np.stack([block.core_min, block.core_max], axis=0).astype(np.float32),
            "overlap_aabb": np.stack([block.overlap_min, block.overlap_max], axis=0).astype(np.float32),
            "point_indices": indices.astype(np.int64),
            "points": self.points[indices].astype(np.float32),
        }
        if self.normals is not None:
            item["normals"] = self.normals[indices].astype(np.float32)

        return item

    def query_blocks(self, query_point: np.ndarray, top_k: Optional[int] = None):
        if top_k is None:
            top_k = self.cfg.top_k
        return self.block_index.query(query_point, top_k=top_k)

    def query_block_ids(self, query_point: np.ndarray, top_k: Optional[int] = None):
        if top_k is None:
            top_k = self.cfg.top_k
        return self.block_index.query_block_ids(query_point, top_k=top_k)


class PointCloudTrainDataset(Dataset):
    def __init__(
        self,
        point_cloud_path: str,
        preprocess_cfg: PointCloudPreprocessConfig,
        surface_sample_count: int,
        points_per_shape: int,
        virtual_length: int = 0,
        base_seed: int = 42,
    ) -> None:
        super().__init__()
        if surface_sample_count <= 0:
            raise ValueError("surface_sample_count must be > 0")
        if points_per_shape <= 0:
            raise ValueError("points_per_shape must be > 0")

        self.block_dataset = PointCloudBlockDataset(point_cloud_path=point_cloud_path, cfg=preprocess_cfg)
        self.surface_sample_count = int(surface_sample_count)
        self.points_per_shape = int(points_per_shape)
        self.base_seed = int(base_seed)

        self.global_points = self.block_dataset.points.astype(np.float32)
        if self.block_dataset.normals is None:
            self.global_normals = np.zeros_like(self.global_points, dtype=np.float32)
        else:
            self.global_normals = self.block_dataset.normals.astype(np.float32)

        if self.global_points.shape[0] == 0:
            raise ValueError("PointCloudTrainDataset received empty points")

        default_len = max(1, len(self.block_dataset))
        self.virtual_length = int(virtual_length) if int(virtual_length) > 0 else default_len

    @property
    def num_blocks(self) -> int:
        return len(self.block_dataset)

    def __len__(self) -> int:
        return self.virtual_length

    @staticmethod
    def _safe_take(array: np.ndarray, indices: np.ndarray) -> np.ndarray:
        if array.shape[0] == 0:
            return np.zeros((indices.shape[0], array.shape[1]), dtype=np.float32)
        return array[indices].astype(np.float32)

    def _sample_indices(self, n: int, count: int, rng: np.random.Generator) -> np.ndarray:
        if n <= 0:
            return np.zeros((count,), dtype=np.int64)
        replace = n < count
        return rng.choice(n, size=count, replace=replace).astype(np.int64)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        if self.num_blocks <= 0:
            raise RuntimeError("No active blocks found in point cloud")

        block_item = self.block_dataset[int(idx) % self.num_blocks]
        block_points = block_item["points"].astype(np.float32)
        block_normals = block_item.get("normals")
        if block_normals is None:
            block_normals_np = np.zeros_like(block_points, dtype=np.float32)
        else:
            block_normals_np = block_normals.astype(np.float32)

        if block_points.shape[0] == 0:
            block_points = self.global_points
            block_normals_np = self.global_normals

        rng = np.random.default_rng(self.base_seed + int(idx))

        surf_idx = self._sample_indices(block_points.shape[0], self.surface_sample_count, rng)
        surface_points = self._safe_take(block_points, surf_idx)
        surface_normals = self._safe_take(block_normals_np, surf_idx)

        cloud_idx = self._sample_indices(block_points.shape[0], self.points_per_shape, rng)
        point_cloud = self._safe_take(block_points, cloud_idx)

        return {
            "point_cloud": point_cloud.astype(np.float32),
            "surface_points": surface_points.astype(np.float32),
            "surface_normals": surface_normals.astype(np.float32),
        }


def _dataset_category_dirs(dataset_root: Path, categories: Sequence[str]) -> List[Path]:
    if categories:
        dirs = [dataset_root / c for c in categories]
        missing = [str(p) for p in dirs if not p.is_dir()]
        if missing:
            raise FileNotFoundError(f"Dataset categories not found: {missing}")
        return sorted(dirs)

    return sorted([p for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")])


def _read_split_or_all_models(category_dir: Path, split: str) -> List[str]:
    split_file = category_dir / f"{split}.lst"
    if split_file.exists():
        with split_file.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
    return sorted([p.name for p in category_dir.iterdir() if p.is_dir() and not p.name.startswith(".")])


def discover_dataset_point_clouds(
    dataset_root: str,
    dataset_name: str,
    split: str,
    categories: Sequence[str],
) -> List[str]:
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    kind = dataset_name.strip().lower()
    if kind not in {"shapenet", "synthetic_room", "synthetic_rooms"}:
        raise ValueError("dataset_name must be one of: shapenet, synthetic_room, synthetic_rooms")

    scene_paths: List[str] = []
    for category_dir in _dataset_category_dirs(root, categories):
        model_ids = _read_split_or_all_models(category_dir, split)
        for model_id in model_ids:
            model_dir = category_dir / model_id
            if not model_dir.is_dir():
                continue

            if kind == "shapenet":
                candidate = model_dir / "pointcloud.npz"
                if candidate.exists():
                    scene_paths.append(str(candidate))
                continue

            pointcloud_dir = model_dir / "pointcloud"
            if pointcloud_dir.is_dir():
                npz_files = sorted(pointcloud_dir.glob("pointcloud_*.npz"))
                if npz_files:
                    scene_paths.append(str(npz_files[0]))
                    continue

            candidate_npz = model_dir / "pointcloud.npz"
            if candidate_npz.exists():
                scene_paths.append(str(candidate_npz))
                continue

            candidate_ply = model_dir / "pointcloud0.ply"
            if candidate_ply.exists():
                scene_paths.append(str(candidate_ply))

    if not scene_paths:
        raise RuntimeError(
            f"No point clouds found under root={dataset_root} dataset_name={dataset_name} split={split}"
        )
    return scene_paths


class PointCloudCollectionTrainDataset(Dataset):
    def __init__(
        self,
        dataset_root: str,
        dataset_name: str,
        split: str,
        categories: Sequence[str],
        preprocess_cfg: PointCloudPreprocessConfig,
        surface_sample_count: int,
        points_per_shape: int,
        start: int = 0,
        take: int = -1,
        cache_size: int = 8,
        base_seed: int = 42,
    ) -> None:
        super().__init__()
        if surface_sample_count <= 0:
            raise ValueError("surface_sample_count must be > 0")
        if points_per_shape <= 0:
            raise ValueError("points_per_shape must be > 0")

        scene_paths = discover_dataset_point_clouds(
            dataset_root=dataset_root,
            dataset_name=dataset_name,
            split=split,
            categories=categories,
        )
        start_idx = max(0, int(start))
        if start_idx > 0:
            scene_paths = scene_paths[start_idx:]
        if int(take) > 0:
            scene_paths = scene_paths[: int(take)]
        if not scene_paths:
            raise RuntimeError("PointCloudCollectionTrainDataset has no scenes after slicing")

        self.preprocess_cfg = preprocess_cfg
        self.surface_sample_count = int(surface_sample_count)
        self.points_per_shape = int(points_per_shape)
        self.scene_paths = scene_paths
        self.base_seed = int(base_seed)
        self.cache_size = max(1, int(cache_size))
        self._scene_cache: "OrderedDict[str, Tuple[np.ndarray, np.ndarray]]" = OrderedDict()

    @property
    def num_scenes(self) -> int:
        return len(self.scene_paths)

    def __len__(self) -> int:
        return len(self.scene_paths)

    @staticmethod
    def _sample_indices(n: int, count: int, rng: np.random.Generator) -> np.ndarray:
        if n <= 0:
            return np.zeros((count,), dtype=np.int64)
        replace = n < count
        return rng.choice(n, size=count, replace=replace).astype(np.int64)

    def _load_scene(self, scene_path: str) -> Tuple[np.ndarray, np.ndarray]:
        cached = self._scene_cache.get(scene_path)
        if cached is not None:
            self._scene_cache.move_to_end(scene_path)
            return cached

        points, normals = load_point_cloud(
            file_path=scene_path,
            estimate_normals=self.preprocess_cfg.estimate_normals,
            normal_k=self.preprocess_cfg.normal_k,
        )
        points, normals = voxel_downsample(
            points=points,
            normals=normals,
            voxel_size=self.preprocess_cfg.voxel_size,
        )
        if points.shape[0] == 0:
            raise ValueError(f"Empty point cloud loaded from scene: {scene_path}")

        if normals is None:
            normals = np.zeros_like(points, dtype=np.float32)

        scene_data = (points.astype(np.float32), normals.astype(np.float32))
        self._scene_cache[scene_path] = scene_data
        self._scene_cache.move_to_end(scene_path)
        while len(self._scene_cache) > self.cache_size:
            self._scene_cache.popitem(last=False)

        return scene_data

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        scene_path = self.scene_paths[int(idx) % len(self.scene_paths)]
        points, normals = self._load_scene(scene_path)

        rng = np.random.default_rng(self.base_seed + int(idx))
        surf_idx = self._sample_indices(points.shape[0], self.surface_sample_count, rng)
        cloud_idx = self._sample_indices(points.shape[0], self.points_per_shape, rng)

        return {
            "point_cloud": points[cloud_idx].astype(np.float32),
            "surface_points": points[surf_idx].astype(np.float32),
            "surface_normals": normals[surf_idx].astype(np.float32),
        }
