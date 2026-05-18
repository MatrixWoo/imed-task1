from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.alignment import horn_align_sim3
from imcpe.io_pose import read_pose_txt


def _valid_mask(xyz: np.ndarray) -> np.ndarray:
    return np.isfinite(xyz).all(axis=1)


def _rows_to_xyz(rows) -> np.ndarray:
    """Translations from pose.txt (already frame-to-initial-frame)."""
    xyz = np.full((len(rows), 3), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        if np.isfinite(r.t).all():
            xyz[i] = r.t
    return xyz


def _align_pred_to_gt(gt_xyz: np.ndarray, pred_xyz: np.ndarray) -> np.ndarray:
    mask = _valid_mask(pred_xyz) & _valid_mask(gt_xyz)
    if mask.sum() < 3:
        return pred_xyz.copy()
    model = pred_xyz[mask].T
    data = gt_xyz[mask].T
    rot, trans_gt, _, scale = horn_align_sim3(model, data)
    pred_aligned = pred_xyz.copy()
    pred_aligned[mask] = (scale * rot @ model + trans_gt).T
    return pred_aligned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-pose", type=Path, required=True)
    parser.add_argument("--pred-pose", type=Path, required=True)
    parser.add_argument("--title", type=str, default="Trajectory Overlay")
    parser.add_argument("--save-path", type=Path, default=None)
    args = parser.parse_args()

    gt_rows = read_pose_txt(args.gt_pose)
    pred_rows = read_pose_txt(args.pred_pose)
    if len(gt_rows) != len(pred_rows):
        raise RuntimeError("GT and prediction pose lengths do not match.")

    gt_xyz = _rows_to_xyz(gt_rows)
    pred_xyz = _rows_to_xyz(pred_rows)
    pred_aligned = _align_pred_to_gt(gt_xyz, pred_xyz)
    valid = _valid_mask(pred_aligned)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label="GT", linewidth=2)
    ax.plot(
        pred_aligned[valid, 0],
        pred_aligned[valid, 1],
        pred_aligned[valid, 2],
        label="Pred (Horn aligned)",
        linewidth=2,
    )

    ax.set_title(args.title)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_zlabel("z (mm)")
    ax.legend()
    ax.grid(True)

    if args.save_path is not None:
        args.save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.save_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to: {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
