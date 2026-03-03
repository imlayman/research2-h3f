import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h3f_recon.data.block_index import BlockIndex


def _generate_random_points(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-2.0, 2.0, size=(n, 3)).astype(np.float32)


def test_query_returns_reasonable_candidates_and_contains_covering_block() -> None:
    points = _generate_random_points(seed=42, n=5000)
    block_index = BlockIndex(points=points, block_size=0.45, overlap_ratio=0.25)

    rng = np.random.default_rng(7)
    sample_ids = rng.choice(points.shape[0], size=200, replace=False)

    for sid in sample_ids:
        query = points[sid]
        core_id = block_index.find_core_block_id(query)
        assert core_id is not None

        candidates = block_index.query(query, top_k=8)
        candidate_ids = [block.block_id for block in candidates]

        assert len(candidates) > 0
        assert core_id in candidate_ids

        truth_covering_ids = set(block_index.all_covering_block_ids(query))
        assert len(truth_covering_ids) > 0
        assert set(candidate_ids).issubset(truth_covering_ids)

        for block in candidates:
            assert block.contains(query, use_overlap=True)


def test_query_topk_and_full_query_consistency() -> None:
    points = _generate_random_points(seed=123, n=4000)
    block_index = BlockIndex(points=points, block_size=0.5, overlap_ratio=0.5)

    query = points[128]
    all_candidates = block_index.query(query, top_k=None)
    top2 = block_index.query(query, top_k=2)

    assert len(top2) <= 2
    assert [b.block_id for b in top2] == [b.block_id for b in all_candidates[:2]]


def test_query_far_point_returns_empty_candidates() -> None:
    points = _generate_random_points(seed=999, n=2000)
    block_index = BlockIndex(points=points, block_size=0.5, overlap_ratio=0.25)

    far_query = np.array([100.0, -100.0, 80.0], dtype=np.float32)
    assert block_index.query(far_query, top_k=4) == []


if __name__ == "__main__":
    test_query_returns_reasonable_candidates_and_contains_covering_block()
    test_query_topk_and_full_query_consistency()
    test_query_far_point_returns_empty_candidates()
    print("All block index tests passed.")
