from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .data_io import load_sequence
from .geometry import to_homogeneous
from .io_pose import PoseRow
from .matcher import ALikeLightGlueMatcher, estimate_relative_pose

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


def predict_sequence_cross_camera(
    sequence_dir: Path,
    matcher: ALikeLightGlueMatcher,
    min_matches: int = 8,
    min_inliers: int = 8,
) -> tuple[list[PoseRow], dict]:
    """Cross-camera baseline: endoscope2/L relative to endoscope1/L per frame.

    Frame 0 is identity. Subsequent frames estimate same-time relative pose
    between endoscope1/L and endoscope2/L.
    """
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
        if pts0.shape[0] < min_matches:
            pred_rows.append(_nan_row(frame_id))
            registered_flags.append(False)
            continue

        try:
            pose = estimate_relative_pose(pts0, pts1, seq.k1_l, seq.k2_l)
        except RuntimeError:
            pred_rows.append(_nan_row(frame_id))
            registered_flags.append(False)
            continue

        if pose.num_inliers < min_inliers:
            pred_rows.append(_nan_row(frame_id))
            registered_flags.append(False)
            continue

        T = to_homogeneous(pose.r, pose.t)
        q_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
        t_xyz = T[:3, 3]
        pred_rows.append(PoseRow(frame_idx=frame_id, t=t_xyz, q_xyzw=q_xyzw))
        registered_flags.append(True)

    stats = {
        "n_frames": len(seq.frame_ids),
        "registered_pct": 100.0 * float(np.mean(registered_flags)),
    }
    return pred_rows, stats
