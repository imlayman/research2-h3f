import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.feature_store import SparseFeatureStore


def _build_store(use_half: bool = False) -> SparseFeatureStore:
    store = SparseFeatureStore(
        cell_size_coarse=1.0,
        cell_size_fine=0.5,
        feat_dim_coarse=2,
        feat_dim_fine=3,
        use_half=use_half,
    )
    store.set_active_cells("coarse", [(0, 0, 0), (1, 0, 0)])
    store.set_active_cells("fine", [(0, 0, 0), (2, 0, 0)])

    store.set_cell_feature("coarse", (0, 0, 0), torch.tensor([1.0, 2.0]))
    store.set_cell_feature("coarse", (1, 0, 0), torch.tensor([3.0, 4.0]))
    store.set_cell_feature("fine", (0, 0, 0), torch.tensor([5.0, 6.0, 7.0]))
    store.set_cell_feature("fine", (2, 0, 0), torch.tensor([8.0, 9.0, 10.0]))
    return store


def test_query_features_shape_and_zero_fill() -> None:
    store = _build_store(use_half=False)
    query = torch.tensor(
        [
            [0.2, 0.1, 0.1],   # active coarse(0,0,0), fine(0,0,0)
            [1.2, 0.2, 0.1],   # active coarse(1,0,0), fine(2,0,0)
            [3.2, 0.0, 0.0],   # inactive on both levels
        ],
        dtype=torch.float32,
    )
    feat = store.query_features(query)

    assert feat.shape == (3, 5)
    expected = torch.tensor(
        [
            [1.0, 2.0, 5.0, 6.0, 7.0],
            [3.0, 4.0, 8.0, 9.0, 10.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(feat, expected, atol=1e-6)


def test_world_to_cell_mapping() -> None:
    store = _build_store(use_half=False)
    query = torch.tensor([[1.2, 0.2, 0.1]], dtype=torch.float32)

    coarse = store.spatial_index.world_to_cell("coarse", query)
    fine = store.spatial_index.world_to_cell("fine", query)

    assert coarse.tolist() == [[1, 0, 0]]
    assert fine.tolist() == [[2, 0, 0]]


def test_half_precision_store_keeps_stable_float_output() -> None:
    store = _build_store(use_half=True)
    query = torch.tensor([[0.2, 0.1, 0.1]], dtype=torch.float32)
    feat = store.query_features(query)

    assert store.coarse_embedding.weight.dtype == torch.float16
    assert store.fine_embedding.weight.dtype == torch.float16
    assert feat.dtype == torch.float32
    assert torch.allclose(
        feat,
        torch.tensor([[1.0, 2.0, 5.0, 6.0, 7.0]], dtype=torch.float32),
        atol=1e-3,
    )


if __name__ == "__main__":
    test_query_features_shape_and_zero_fill()
    test_world_to_cell_mapping()
    test_half_precision_store_keeps_stable_float_output()
    print("All feature store tests passed.")

