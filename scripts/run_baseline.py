from __future__ import annotations

import argparse
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

_NAN_POSE = PoseRow(
    frame_idx=-1,
    t=np.array([np.nan, np.nan, np.nan], dtype=np.float64),
    q_xyzw=np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64),
)


def _nan_row(frame_idx: int) -> PoseRow:
    return PoseRow(
        frame_idx=frame_idx,
        t=_NAN_POSE.t.copy(),
        q_xyzw=_NAN_POSE.q_xyzw.copy(),
    )


def run_sequence(sequence_dir: Path, output_root: Path, matcher: ALikeLightGlueMatcher) -> dict[str, float]:
    start = time.perf_counter()
    seq = load_sequence(sequence_dir)

    pred_rows: list[PoseRow] = []
    registered_flags: list[bool] = []

    for i, frame_id in enumerate(seq.frame_ids):
        if i == 0:
            pred_rows.append(
                PoseRow(
                    frame_idx=frame_id,
                    t=np.zeros(3, dtype=np.float64),
                    q_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
                )
            )
            registered_flags.append(True)
            continue

        pts0, pts1 = matcher.match(seq.e1_l_images[i], seq.e2_l_images[i])
        if pts0.shape[0] < 8:
            pred_rows.append(_nan_row(frame_id))
            registered_flags.append(False)
            continue

        try:
            pose = estimate_relative_pose(pts0, pts1, seq.k1_l, seq.k2_l)
        except RuntimeError:
            pred_rows.append(_nan_row(frame_id))
            registered_flags.append(False)
            continue

        if pose.num_inliers < 8:
            pred_rows.append(_nan_row(frame_id))
            registered_flags.append(False)
            continue

        T = to_homogeneous(pose.r, pose.t)
        q_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
        t_xyz = T[:3, 3]
        pred_rows.append(PoseRow(frame_idx=frame_id, t=t_xyz, q_xyzw=q_xyzw))
        registered_flags.append(True)

    pred_path = output_root / sequence_dir.name / "pose.txt"
    write_pose_txt(pred_path, pred_rows)

    runtime_s = time.perf_counter() - start
    registered_pct = 100.0 * np.mean(np.array(registered_flags, dtype=np.float64))
    return {"runtime_s": runtime_s, "registered_pct": float(registered_pct)}


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
