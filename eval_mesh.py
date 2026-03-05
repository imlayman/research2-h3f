import argparse
import csv
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree

from h3f_recon.data.dataset import load_point_cloud, voxel_downsample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated meshes on a dataset split")
    parser.add_argument("--generate-dir", type=str, required=True, help="Directory produced by generate.py")
    parser.add_argument("--output-csv", type=str, default="", help="Output csv path (default: <generate-dir>/mesh_eval.csv)")
    parser.add_argument("--num-samples", type=int, default=100000, help="Sample points for each side")
    parser.add_argument("--fscore-threshold", type=float, default=0.01, help="Distance threshold for F-score")
    parser.add_argument("--gt-voxel-size", type=float, default=0.0, help="Optional voxel downsample for GT point cloud")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def _normalize(vectors: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.clip(norms, eps, None)


def _sample_points(
    points: np.ndarray,
    normals: Optional[np.ndarray],
    n_samples: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    n = int(points.shape[0])
    if n == 0:
        return np.zeros((0, 3), dtype=np.float32), None
    if n_samples <= 0:
        return points.astype(np.float32), None if normals is None else normals.astype(np.float32)

    replace = n < n_samples
    idx = rng.choice(n, size=n_samples, replace=replace).astype(np.int64)
    sampled_points = points[idx].astype(np.float32)
    sampled_normals = None if normals is None else normals[idx].astype(np.float32)
    return sampled_points, sampled_normals


def _mesh_to_points(mesh_path: str, n_samples: int) -> Tuple[np.ndarray, Optional[np.ndarray], str]:
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return np.zeros((0, 3), dtype=np.float32), None, "empty_mesh"

    try:
        sampled = mesh.sample_points_uniformly(number_of_points=int(n_samples), use_triangle_normal=True)
    except Exception as exc:  # pragma: no cover - defensive
        return np.zeros((0, 3), dtype=np.float32), None, f"sample_failed:{exc}"

    points = np.asarray(sampled.points, dtype=np.float32)
    if points.shape[0] == 0:
        return points, None, "empty_mesh"

    normals = None
    sampled_normals = np.asarray(sampled.normals, dtype=np.float32)
    if sampled_normals.shape == points.shape:
        normals = _normalize(sampled_normals)

    return points, normals, "ok"


def _query_nn(tree: cKDTree, query_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    try:
        dist, idx = tree.query(query_points, k=1, workers=-1)
    except TypeError:
        dist, idx = tree.query(query_points, k=1)
    return np.asarray(dist, dtype=np.float64), np.asarray(idx, dtype=np.int64)


def _compute_metrics(
    gt_points: np.ndarray,
    gt_normals: Optional[np.ndarray],
    pd_points: np.ndarray,
    pd_normals: Optional[np.ndarray],
    threshold: float,
) -> Dict[str, float]:
    tree_pd = cKDTree(pd_points)
    dist_comp, idx_comp = _query_nn(tree_pd, gt_points)

    tree_gt = cKDTree(gt_points)
    dist_acc, idx_acc = _query_nn(tree_gt, pd_points)

    cd_l1 = 0.5 * (float(dist_comp.mean()) + float(dist_acc.mean()))

    recall = float(np.mean(dist_comp < threshold))
    precision = float(np.mean(dist_acc < threshold))
    f_score = 0.0
    if precision + recall > 0.0:
        f_score = float(2.0 * precision * recall / (precision + recall))

    nc = float("nan")
    if gt_normals is not None and pd_normals is not None:
        align_comp = np.abs(np.sum(gt_normals * pd_normals[idx_comp], axis=1))
        align_acc = np.abs(np.sum(pd_normals * gt_normals[idx_acc], axis=1))
        nc = 0.5 * (float(np.mean(align_comp)) + float(np.mean(align_acc)))

    return {
        "cd_l1": cd_l1,
        "nc": nc,
        "f_score": f_score,
        "precision": precision,
        "recall": recall,
    }


def _mean_ignore_nan(values: List[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if not valid:
        return float("nan")
    return float(sum(valid) / len(valid))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    logger = logging.getLogger("mesh_eval")

    generate_dir = Path(args.generate_dir)
    summary_csv = generate_dir / "summary.csv"
    if not summary_csv.exists():
        raise FileNotFoundError(f"summary.csv not found in generate dir: {generate_dir}")

    output_csv = Path(args.output_csv) if args.output_csv else (generate_dir / "mesh_eval.csv")
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with summary_csv.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"No rows in {summary_csv}")

    logger.info("Loaded %d generated scenes from %s", len(rows), summary_csv)
    logger.info(
        "Eval setup | num_samples=%d threshold=%.6f gt_voxel=%.6f",
        args.num_samples,
        args.fscore_threshold,
        args.gt_voxel_size,
    )

    rng = np.random.default_rng(args.seed)
    result_rows: List[Dict[str, object]] = []

    cd_vals: List[float] = []
    nc_vals: List[float] = []
    fs_vals: List[float] = []
    ok_count = 0

    for i, row in enumerate(rows):
        scene_id = row.get("scene_id", f"scene_{i:06d}")
        scene_path = row.get("scene_path", "")
        mesh_path = row.get("mesh_path", "")
        status = row.get("status", "")

        base_out: Dict[str, object] = {
            "scene_id": scene_id,
            "scene_path": scene_path,
            "mesh_path": mesh_path,
            "status": status,
            "eval_status": "",
            "num_gt_points": 0,
            "num_pred_points": 0,
            "cd_l1": float("nan"),
            "nc": float("nan"),
            "f_score": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "threshold": float(args.fscore_threshold),
            "error": "",
        }

        try:
            if status != "ok":
                base_out["eval_status"] = "skip_generate_failed"
                result_rows.append(base_out)
                continue

            gt_points, gt_normals = load_point_cloud(file_path=scene_path, estimate_normals=False)
            if args.gt_voxel_size > 0.0:
                gt_points, gt_normals = voxel_downsample(
                    points=gt_points,
                    normals=gt_normals,
                    voxel_size=float(args.gt_voxel_size),
                )
            if gt_points.shape[0] == 0:
                base_out["eval_status"] = "empty_gt"
                result_rows.append(base_out)
                continue

            pred_points, pred_normals, pred_status = _mesh_to_points(mesh_path=mesh_path, n_samples=args.num_samples)
            if pred_status != "ok" or pred_points.shape[0] == 0:
                base_out["eval_status"] = pred_status
                base_out["num_gt_points"] = int(gt_points.shape[0])
                base_out["num_pred_points"] = int(pred_points.shape[0])
                base_out["f_score"] = 0.0
                base_out["precision"] = 0.0
                base_out["recall"] = 0.0
                result_rows.append(base_out)
                continue

            gt_eval_points, gt_eval_normals = _sample_points(
                points=gt_points,
                normals=gt_normals,
                n_samples=args.num_samples,
                rng=rng,
            )

            metrics = _compute_metrics(
                gt_points=gt_eval_points,
                gt_normals=gt_eval_normals,
                pd_points=pred_points,
                pd_normals=pred_normals,
                threshold=float(args.fscore_threshold),
            )

            base_out.update(
                {
                    "eval_status": "ok",
                    "num_gt_points": int(gt_eval_points.shape[0]),
                    "num_pred_points": int(pred_points.shape[0]),
                    "cd_l1": metrics["cd_l1"],
                    "nc": metrics["nc"],
                    "f_score": metrics["f_score"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                }
            )
            result_rows.append(base_out)

            cd_vals.append(float(metrics["cd_l1"]))
            nc_vals.append(float(metrics["nc"]))
            fs_vals.append(float(metrics["f_score"]))
            ok_count += 1

            logger.info(
                "Scene %d/%d ok | id=%s cd_l1=%.6f nc=%.6f f_score=%.6f",
                i + 1,
                len(rows),
                scene_id,
                metrics["cd_l1"],
                metrics["nc"],
                metrics["f_score"],
            )
        except Exception as exc:  # pragma: no cover - defensive
            base_out["eval_status"] = "failed"
            base_out["error"] = str(exc)
            result_rows.append(base_out)
            logger.exception("Scene %d/%d failed | id=%s", i + 1, len(rows), scene_id)

    mean_row = {
        "scene_id": "__mean__",
        "scene_path": "",
        "mesh_path": "",
        "status": "",
        "eval_status": "ok",
        "num_gt_points": "",
        "num_pred_points": "",
        "cd_l1": _mean_ignore_nan(cd_vals),
        "nc": _mean_ignore_nan(nc_vals),
        "f_score": _mean_ignore_nan(fs_vals),
        "precision": "",
        "recall": "",
        "threshold": float(args.fscore_threshold),
        "error": "",
    }
    result_rows.append(mean_row)

    fieldnames = [
        "scene_id",
        "scene_path",
        "mesh_path",
        "status",
        "eval_status",
        "num_gt_points",
        "num_pred_points",
        "cd_l1",
        "nc",
        "f_score",
        "precision",
        "recall",
        "threshold",
        "error",
    ]
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result_rows)

    logger.info(
        "Evaluation finished | scenes=%d ok=%d output=%s mean_cd_l1=%.6f mean_nc=%.6f mean_f_score=%.6f",
        len(rows),
        ok_count,
        output_csv,
        float(mean_row["cd_l1"]) if not math.isnan(float(mean_row["cd_l1"])) else float("nan"),
        float(mean_row["nc"]) if not math.isnan(float(mean_row["nc"])) else float("nan"),
        float(mean_row["f_score"]) if not math.isnan(float(mean_row["f_score"])) else float("nan"),
    )


if __name__ == "__main__":
    main()
