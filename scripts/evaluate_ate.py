from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.geometry import umeyama_alignment
from imcpe.io_pose import read_pose_txt


def valid_mask(pred_xyz: np.ndarray) -> np.ndarray:
    return np.isfinite(pred_xyz).all(axis=1)


def ate_rmse(gt_xyz: np.ndarray, pred_xyz: np.ndarray) -> tuple[float, float]:
    mask = valid_mask(pred_xyz)
    gt = gt_xyz[mask]
    pred = pred_xyz[mask]
    s, R, t = umeyama_alignment(pred, gt)
    pred_aligned = (s * (R @ pred.T)).T + t
    err = np.linalg.norm(pred_aligned - gt, axis=1)
    rmse = float(np.sqrt(np.mean(err**2)))
    registered_pct = 100.0 * float(mask.mean())
    return rmse, registered_pct


def eval_sequence(gt_pose_path: Path, pred_pose_path: Path) -> tuple[float, float]:
    gt_rows = read_pose_txt(gt_pose_path)
    pred_rows = read_pose_txt(pred_pose_path)

    if len(gt_rows) != len(pred_rows):
        raise RuntimeError(f"Mismatched lengths: {gt_pose_path} vs {pred_pose_path}")

    gt_xyz = np.stack([r.t for r in gt_rows], axis=0)
    pred_xyz = np.stack([r.t for r in pred_rows], axis=0)
    return ate_rmse(gt_xyz, pred_xyz)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--pred-root", type=Path, required=True)
    args = parser.parse_args()

    start = time.perf_counter()
    split_dir = args.data_root / args.split
    pred_split_dir = args.pred_root / args.split
    sequence_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])

    rmses = []
    regs = []
    for seq in sequence_dirs:
        rmse, reg = eval_sequence(seq / "pose.txt", pred_split_dir / seq.name / "pose.txt")
        rmses.append(rmse)
        regs.append(reg)
        print(f"{seq.name}: ATE_RMSE={rmse:.6f} mm registered={reg:.2f}%")

    print(f"AVG: ATE_RMSE={np.mean(rmses):.6f} mm registered={np.mean(regs):.2f}%")
    print(f"Evaluation runtime: {time.perf_counter() - start:.3f}s")


if __name__ == "__main__":
    main()
