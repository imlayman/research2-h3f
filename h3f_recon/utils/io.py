from __future__ import annotations

import numpy as np


def _write_ascii_ply(path: str, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")

        if colors is None:
            for p in points:
                f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        else:
            for p, c in zip(points, colors):
                f.write(
                    f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} "
                    f"{int(c[0])} {int(c[1])} {int(c[2])}\n"
                )


def save_point_cloud(path: str, points: np.ndarray, scalar: np.ndarray | None = None) -> None:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")

    colors = None
    if scalar is not None and len(points) > 0:
        scalar = np.asarray(scalar, dtype=np.float32).reshape(-1)
        s_min = float(scalar.min())
        s_max = float(scalar.max())
        norm = (scalar - s_min) / (s_max - s_min + 1e-6)
        colors = np.stack([norm, 1.0 - norm, np.zeros_like(norm)], axis=-1)
        colors = np.clip(colors * 255.0, 0, 255).astype(np.uint8)

    try:
        import open3d as o3d  # type: ignore

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        if colors is not None:
            pcd.colors = o3d.utility.Vector3dVector((colors / 255.0).astype(np.float64))
        o3d.io.write_point_cloud(path, pcd)
    except Exception:
        _write_ascii_ply(path, points, colors)
