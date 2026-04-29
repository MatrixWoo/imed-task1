from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.geometry import umeyama_alignment
from imcpe.io_pose import read_pose_txt


def _valid_mask(xyz: np.ndarray) -> np.ndarray:
    return np.isfinite(xyz).all(axis=1)


def _align_pred_to_gt(gt_xyz: np.ndarray, pred_xyz: np.ndarray) -> np.ndarray:
    mask = _valid_mask(pred_xyz)
    pred_valid = pred_xyz[mask]
    gt_valid = gt_xyz[mask]
    s, R, t = umeyama_alignment(pred_valid, gt_valid)
    pred_aligned = pred_xyz.copy()
    pred_aligned[mask] = (s * (R @ pred_valid.T)).T + t
    return pred_aligned


def _rows_to_relative_xyz(rows) -> np.ndarray:
    Ts = []
    for r in rows:
        if not np.isfinite(r.t).all() or not np.isfinite(r.q_xyzw).all():
            Ts.append(None)
            continue
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = Rotation.from_quat(r.q_xyzw).as_matrix()
        T[:3, 3] = r.t
        Ts.append(T)

    first_valid = next((T for T in Ts if T is not None), None)
    if first_valid is None:
        return np.full((len(rows), 3), np.nan, dtype=np.float64)

    T0_inv = np.linalg.inv(first_valid)
    xyz = np.full((len(rows), 3), np.nan, dtype=np.float64)
    for i, T in enumerate(Ts):
        if T is None:
            continue
        T_rel = T0_inv @ T
        xyz[i] = T_rel[:3, 3]
    return xyz


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

    gt_xyz = _rows_to_relative_xyz(gt_rows)
    pred_xyz = _rows_to_relative_xyz(pred_rows)
    pred_aligned = _align_pred_to_gt(gt_xyz, pred_xyz)
    valid = _valid_mask(pred_aligned)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(gt_xyz[:, 0], gt_xyz[:, 1], gt_xyz[:, 2], label="GT", linewidth=2)
    ax.plot(
        pred_aligned[valid, 0],
        pred_aligned[valid, 1],
        pred_aligned[valid, 2],
        label="Pred (aligned)",
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
