"""
Baseline ALIKED + LightGlue + Essential matrix entry point for containerized
submission. Reads sequences from /input/, writes pose_predictions.txt files
to /output/.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.inference import predict_sequence_cross_camera
from imcpe.io_pose import write_pose_txt
from imcpe.matcher import ALikeLightGlueMatcher

PRED_FILENAME = "pose_predictions.txt"


def predict_sequence(
    sequence_dir: Path,
    output_dir: Path,
    matcher: ALikeLightGlueMatcher,
) -> dict:
    pred_rows, stats = predict_sequence_cross_camera(sequence_dir, matcher)
    out_path = output_dir / sequence_dir.name / PRED_FILENAME
    write_pose_txt(out_path, pred_rows)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(os.environ.get("INPUT_DIR", "/input")),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", "/output")),
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sequence_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if not sequence_dirs:
        print(f"ERROR: No sequence directories found under {input_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(sequence_dirs)} sequence(s) under {input_dir}", flush=True)

    matcher = ALikeLightGlueMatcher(device=args.device)

    total_start = time.perf_counter()
    for sequence_dir in tqdm(sequence_dirs, desc="Sequences"):
        seq_start = time.perf_counter()
        try:
            stats = predict_sequence(sequence_dir, output_dir, matcher)
        except Exception as e:
            print(
                f"  {sequence_dir.name}: FAILED ({type(e).__name__}: {e})",
                file=sys.stderr,
                flush=True,
            )
            raise
        print(
            f"  {sequence_dir.name}: "
            f"frames={stats['n_frames']} "
            f"registered={stats['registered_pct']:.2f}% "
            f"runtime={time.perf_counter() - seq_start:.2f}s",
            flush=True,
        )

    print(f"Total runtime: {time.perf_counter() - total_start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
