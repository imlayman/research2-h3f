from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

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

    loaded = _try_load_with_open3d(path)
    if loaded is None:
        if suffix == ".ply":
            loaded = _load_ascii_ply(path)
        elif suffix == ".xyz":
            loaded = _load_ascii_xyz(path)
        elif suffix == ".pcd":
            loaded = _load_ascii_pcd(path)
        else:
            raise ValueError(f"Unsupported format: {suffix}. Supported: .ply/.xyz/.pcd")

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
