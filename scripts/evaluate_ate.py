from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.geometry import umeyama_alignment
from imcpe.io_pose import PoseRow, read_pose_txt

PRED_FILENAME = "pose_predictions.txt"


def _valid_mask(xyz: np.ndarray) -> np.ndarray:
    return np.isfinite(xyz).all(axis=1)


def ate_rmse(gt_xyz: np.ndarray, pred_xyz: np.ndarray) -> tuple[float, float]:
    """Sim(3)-aligned ATE RMSE plus % of GT frames with a finite prediction."""
    if gt_xyz.shape != pred_xyz.shape:
        raise ValueError(f"Shape mismatch: gt {gt_xyz.shape} vs pred {pred_xyz.shape}")

    mask = _valid_mask(pred_xyz)
    n_valid = int(mask.sum())
    registered_pct = 100.0 * n_valid / gt_xyz.shape[0]

    if n_valid < 3:
        # Umeyama needs at least 3 non-collinear points to be meaningful.
        return float("nan"), registered_pct

    gt = gt_xyz[mask]
    pred = pred_xyz[mask]
    s, R, t = umeyama_alignment(pred, gt)
    pred_aligned = (s * (R @ pred.T)).T + t
    err = np.linalg.norm(pred_aligned - gt, axis=1)
    rmse = float(np.sqrt(np.mean(err**2)))
    return rmse, registered_pct


def _align_pred_to_gt_by_frame(
    gt_rows: list[PoseRow], pred_rows: list[PoseRow]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align prediction rows to GT rows by frame_idx. Missing predictions become
    NaN rows in the output (counted as unregistered). Extra predictions
    (frames not in GT) are silently dropped.
    """
    pred_by_idx = {r.frame_idx: r for r in pred_rows}
    gt_xyz = np.stack([r.t for r in gt_rows], axis=0)
    pred_xyz = np.full_like(gt_xyz, np.nan)
    for i, gt_row in enumerate(gt_rows):
        pr = pred_by_idx.get(gt_row.frame_idx)
        if pr is not None:
            pred_xyz[i] = pr.t
    return gt_xyz, pred_xyz


def eval_sequence(gt_pose_path: Path, pred_pose_path: Path) -> dict:
    if not pred_pose_path.exists():
        raise FileNotFoundError(f"Missing prediction file: {pred_pose_path}")
    gt_rows = read_pose_txt(gt_pose_path)
    pred_rows = read_pose_txt(pred_pose_path)
    gt_xyz, pred_xyz = _align_pred_to_gt_by_frame(gt_rows, pred_rows)
    rmse, reg = ate_rmse(gt_xyz, pred_xyz)
    return {
        "ate_rmse_mm": rmse,
        "registered_pct": reg,
        "n_gt_frames": int(gt_xyz.shape[0]),
        "n_pred_frames": int(len(pred_rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-root", type=Path, required=True,
                        help="Directory containing per-sequence subdirs with pose.txt (GT).")
    parser.add_argument("--pred-root", type=Path, required=True,
                        help="Directory containing per-sequence subdirs with pose_predictions.txt.")
    parser.add_argument("--json-out", type=Path, default=None,
                        help="Optional path to write per-sequence + aggregate JSON.")
    args = parser.parse_args()

    start = time.perf_counter()
    sequence_dirs = sorted(p for p in args.gt_root.iterdir() if p.is_dir())
    if not sequence_dirs:
        print(f"ERROR: No sequence subdirs under {args.gt_root}", file=sys.stderr)
        sys.exit(2)

    per_seq: dict[str, dict] = {}
    for seq in sequence_dirs:
        try:
            metrics = eval_sequence(seq / "pose.txt", args.pred_root / seq.name / PRED_FILENAME)
        except FileNotFoundError as e:
            print(f"{seq.name}: MISSING ({e})", file=sys.stderr)
            metrics = {
                "ate_rmse_mm": float("nan"),
                "registered_pct": 0.0,
                "n_gt_frames": 0,
                "n_pred_frames": 0,
                "error": "missing_prediction_file",
            }
        per_seq[seq.name] = metrics
        print(
            f"{seq.name}: "
            f"ATE_RMSE={metrics['ate_rmse_mm']:.6f} mm "
            f"registered={metrics['registered_pct']:.2f}% "
            f"({metrics['n_pred_frames']}/{metrics['n_gt_frames']} frames)"
        )

    valid_rmses = [m["ate_rmse_mm"] for m in per_seq.values() if np.isfinite(m["ate_rmse_mm"])]
    aggregate = {
        "mean_ate_rmse_mm": float(np.mean(valid_rmses)) if valid_rmses else float("nan"),
        "mean_registered_pct": float(np.mean([m["registered_pct"] for m in per_seq.values()])),
        "n_sequences": len(per_seq),
        "n_scored_sequences": len(valid_rmses),
    }
    print(
        f"AVG: ATE_RMSE={aggregate['mean_ate_rmse_mm']:.6f} mm "
        f"registered={aggregate['mean_registered_pct']:.2f}% "
        f"({aggregate['n_scored_sequences']}/{aggregate['n_sequences']} scored)"
    )
    print(f"Evaluation runtime: {time.perf_counter() - start:.3f}s")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps({"per_sequence": per_seq, "aggregate": aggregate}, indent=2))
        print(f"Wrote metrics to: {args.json_out}")


if __name__ == "__main__":
    main()