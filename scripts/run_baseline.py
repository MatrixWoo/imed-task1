from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.inference import predict_sequence_cross_camera
from imcpe.io_pose import write_pose_txt
from imcpe.matcher import ALikeLightGlueMatcher


def run_sequence(sequence_dir: Path, output_root: Path, matcher: ALikeLightGlueMatcher) -> dict[str, float]:
    start = time.perf_counter()
    pred_rows, stats = predict_sequence_cross_camera(sequence_dir, matcher)
    pred_path = output_root / sequence_dir.name / "pose.txt"
    write_pose_txt(pred_path, pred_rows)

    runtime_s = time.perf_counter() - start
    return {"runtime_s": runtime_s, "registered_pct": stats["registered_pct"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=str, required=True, choices=["train", "test"])
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    split_dir = args.data_root / args.split
    sequence_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])
    matcher = ALikeLightGlueMatcher(device=args.device)

    runtimes = []
    regs = []
    for sequence_dir in tqdm(sequence_dirs):
        stats = run_sequence(sequence_dir, args.output_root / args.split, matcher)
        runtimes.append(stats["runtime_s"])
        regs.append(stats["registered_pct"])
        print(
            f"{sequence_dir.name}: "
            f"registered={stats['registered_pct']:.2f}% "
            f"runtime={stats['runtime_s']:.3f}s"
        )

    print(
        f"AVG: registered={np.mean(regs):.2f}% "
        f"runtime={np.mean(runtimes):.3f}s"
    )


if __name__ == "__main__":
    main()
