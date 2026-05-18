"""
Participant entry point for the iMED Pose Estimation challenge.

CONTRACT
--------
Input  (read-only mount at /input):
    /input/<sequence_name>/
        K.txt                                  intrinsics for all 4 cameras
        endoscope1/L/frame_*.png               left  view, scope 1
        endoscope1/R/frame_*.png               right view, scope 1
        endoscope2/L/frame_*.png               left  view, scope 2
        endoscope2/R/frame_*.png               right view, scope 2

Output (writable mount at /output):
    /output/<sequence_name>/pose_predictions.txt
        One row per frame, whitespace-separated:
            <frame_idx> <tx> <ty> <tz> <qx> <qy> <qz> <qw>
        Frame 0 must be (t=0, q=identity): endoscope2/L relative to
        endoscope1/L at the first frame index in endoscope2/L.
        Use NaN for any frame you cannot register.
        Translation is scale-ambiguous (scoring uses Sim(3) alignment).
        Quaternion convention: (qx, qy, qz, qw) - scipy default.

Runtime constraints:
    - 1x RTX 4090, CUDA 11.8 host driver
    - 10 minutes total wall-clock for all sequences
    - --network=none (no internet at runtime; bake weights into the image)
    - 20 GB RAM
    - Read-only /input, writable /output, nothing else persists
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from tqdm import tqdm

from imcpe.inference import predict_sequence_cross_camera
from imcpe.io_pose import write_pose_txt
from imcpe.matcher import ALikeLightGlueMatcher

PRED_FILENAME = "pose_predictions.txt"


def predict_sequence(sequence_dir: Path, output_dir: Path, matcher: ALikeLightGlueMatcher) -> dict:
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

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sequence_dirs = sorted(p for p in args.input_dir.iterdir() if p.is_dir())
    if not sequence_dirs:
        print(f"ERROR: No sequence directories under {args.input_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"Found {len(sequence_dirs)} sequence(s).", flush=True)

    matcher = ALikeLightGlueMatcher(device=args.device)

    t0 = time.perf_counter()
    for seq in tqdm(sequence_dirs, desc="Sequences"):
        seq_t0 = time.perf_counter()
        try:
            stats = predict_sequence(seq, args.output_dir, matcher)
        except Exception as e:
            print(
                f"  {seq.name}: FAILED ({type(e).__name__}: {e})",
                file=sys.stderr,
                flush=True,
            )
            raise
        print(
            f"  {seq.name}: "
            f"frames={stats['n_frames']} "
            f"registered={stats['registered_pct']:.2f}% "
            f"runtime={time.perf_counter() - seq_t0:.2f}s",
            flush=True,
        )
    print(f"Total: {time.perf_counter() - t0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
