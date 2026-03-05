from pathlib import Path

import numpy as np
import pytest

from h3f_recon.data.dataset import load_point_cloud


def test_load_point_cloud_rejects_occupancy_npz(tmp_path: Path) -> None:
    occ_path = tmp_path / "points.npz"
    np.savez(
        occ_path,
        points=np.random.randn(128, 3).astype(np.float32),
        occupancies=np.random.randint(0, 2, size=(128,), dtype=np.uint8),
    )

    with pytest.raises(ValueError, match="occupancy"):
        load_point_cloud(str(occ_path), estimate_normals=False)


def test_load_point_cloud_accepts_pointcloud_npz(tmp_path: Path) -> None:
    pc_path = tmp_path / "pointcloud.npz"
    points = np.random.randn(64, 3).astype(np.float32)
    normals = np.random.randn(64, 3).astype(np.float32)
    np.savez(pc_path, points=points, normals=normals)

    loaded_points, loaded_normals = load_point_cloud(str(pc_path), estimate_normals=False)
    assert loaded_points.shape == (64, 3)
    assert loaded_normals is not None
    assert loaded_normals.shape == (64, 3)
