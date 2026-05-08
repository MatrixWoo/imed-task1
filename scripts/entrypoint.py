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

import numpy as np
from scipy.spatial.transform import Rotation
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from imcpe.data_io import load_sequence
from imcpe.geometry import to_homogeneous
from imcpe.io_pose import PoseRow, write_pose_txt
from imcpe.matcher import ALikeLightGlueMatcher, estimate_relative_pose

PRED_FILENAME = "pose_predictions.txt"
MIN_MATCHES = 8
MIN_INLIERS = 8


def _nan_row(frame_idx: int) -> PoseRow:
    return PoseRow(
        frame_idx=frame_idx,
        t=np.full(3, np.nan, dtype=np.float64),
        q_xyzw=np.full(4, np.nan, dtype=np.float64),
    )


def predict_sequence(
    sequence_dir: Path,
    output_dir: Path,
    matcher: ALikeLightGlueMatcher,
) -> dict:
    seq = load_sequence(sequence_dir)

    pred_rows: list[PoseRow] = [
        PoseRow(
            frame_idx=seq.frame_ids[0],
            t=np.zeros(3, dtype=np.float64),
            q_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )
    ]
    registered_flags = [True]
    abs_poses: dict[int, np.ndarray] = {0: np.eye(4, dtype=np.float64)}
    last_registered_idx = 0

    for i in range(1, len(seq.frame_ids)):
        pts0, pts1 = matcher.match(
            seq.e2_l_images[last_registered_idx], seq.e2_l_images[i]
        )
        if pts0.shape[0] < MIN_MATCHES:
            pred_rows.append(_nan_row(seq.frame_ids[i]))
            registered_flags.append(False)
            continue

        pose = estimate_relative_pose(pts0, pts1, seq.k2_l)
        if pose.num_inliers < MIN_INLIERS:
            pred_rows.append(_nan_row(seq.frame_ids[i]))
            registered_flags.append(False)
            continue

        T_last_to_i = to_homogeneous(pose.r, pose.t)
        T_0_to_i = abs_poses[last_registered_idx] @ T_last_to_i
        abs_poses[i] = T_0_to_i
        pred_rows.append(
            PoseRow(
                frame_idx=seq.frame_ids[i],
                t=T_0_to_i[:3, 3],
                q_xyzw=Rotation.from_matrix(T_0_to_i[:3, :3]).as_quat(),
            )
        )
        registered_flags.append(True)
        last_registered_idx = i

    out_path = output_dir / sequence_dir.name / PRED_FILENAME
    write_pose_txt(out_path, pred_rows)

    return {
        "n_frames": len(seq.frame_ids),
        "registered_pct": 100.0 * float(np.mean(registered_flags)),
    }


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