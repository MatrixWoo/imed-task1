from __future__ import annotations

import numpy as np

from .geometry import from_homogeneous, quaternion_xyzw_from_r, to_homogeneous
from .io_pose import PoseRow


def chain_relative_poses(
    frame_ids: list[int],
    rel_poses: list[tuple[np.ndarray, np.ndarray] | None],
) -> tuple[list[PoseRow], list[bool]]:
    rows: list[PoseRow] = []
    registered: list[bool] = []
    T_w_c = np.eye(4, dtype=np.float64)

    rows.append(
        PoseRow(
            frame_idx=frame_ids[0],
            t=np.zeros(3, dtype=np.float64),
            q_xyzw=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        )
    )
    registered.append(True)

    for i in range(1, len(frame_ids)):
        rel = rel_poses[i - 1]
        if rel is None:
            rows.append(
                PoseRow(
                    frame_idx=frame_ids[i],
                    t=np.array([np.nan, np.nan, np.nan], dtype=np.float64),
                    q_xyzw=np.array([np.nan, np.nan, np.nan, np.nan], dtype=np.float64),
                )
            )
            registered.append(False)
            continue

        r_rel, t_rel = rel
        T_prev_cur = to_homogeneous(r_rel, t_rel)
        T_w_c = T_w_c @ T_prev_cur
        r_abs, t_abs = from_homogeneous(T_w_c)
        q_abs = quaternion_xyzw_from_r(r_abs)
        rows.append(PoseRow(frame_idx=frame_ids[i], t=t_abs, q_xyzw=q_abs))
        registered.append(True)

    return rows, registered
