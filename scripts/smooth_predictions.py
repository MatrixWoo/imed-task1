"""Post-process baseline predictions with temporal smoothing.

Usage:
    python scripts/smooth_predictions.py \
        --pred-root <pred-root> --split train \
        --out-root <pred-root>-smooth \
        --window 4 --sigma 2.0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.io_pose import read_pose_txt, write_pose_txt
from imcpe.temporal_smooth import smooth_pose_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument("--sigma", type=float, default=2.0)
    args = parser.parse_args()

    split_dir = args.pred_root / args.split
    if not split_dir.is_dir():
        print(f"ERROR: {split_dir} not found", file=sys.stderr)
        sys.exit(2)

    for pose_file in sorted(split_dir.glob("*/pose.txt")):
        rows = read_pose_txt(pose_file)
        smoothed = smooth_pose_sequence(rows, window=args.window, sigma=args.sigma)
        out_path = args.out_root / args.split / pose_file.parent.name / "pose.txt"
        write_pose_txt(out_path, smoothed)
        print(f"smoothed: {pose_file.parent.name}")

    print(f"Done. Output written to {args.out_root / args.split}")


if __name__ == "__main__":
    main()
