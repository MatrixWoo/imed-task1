from __future__ import annotations

import numpy as np


def compute_rpe_by_index(
    R_ref_wc: np.ndarray,
    C_ref_wc: np.ndarray,
    ids: np.ndarray,
    R_est_wc: np.ndarray,
    C_est_wc: np.ndarray,
    delta: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """RPE using ``delta`` steps along the sorted pose list (for non-consecutive frame_idx)."""
    ids = np.asarray(ids, dtype=int)
    n = len(ids)
    if n == 0:
        raise ValueError("Empty pose ID list.")

    pair_ids: list[tuple[int, int]] = []
    trans_errors: list[float] = []
    rot_errors_deg: list[float] = []

    def build_twc(R_wc: np.ndarray, C: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_wc
        T[:3, 3] = C
        return T

    for k in range(n - delta):
        j = k + delta
        T_ref_i = build_twc(R_ref_wc[k], C_ref_wc[k])
        T_ref_j = build_twc(R_ref_wc[j], C_ref_wc[j])
        T_est_i = build_twc(R_est_wc[k], C_est_wc[k])
        T_est_j = build_twc(R_est_wc[j], C_est_wc[j])

        T_ref_rel = np.linalg.inv(T_ref_i) @ T_ref_j
        T_est_rel = np.linalg.inv(T_est_i) @ T_est_j
        T_err = np.linalg.inv(T_ref_rel) @ T_est_rel

        trans_errors.append(float(np.linalg.norm(T_err[:3, 3])))
        cos_angle = (np.trace(T_err[:3, :3]) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        rot_errors_deg.append(float(np.degrees(np.arccos(cos_angle))))
        pair_ids.append((int(ids[k]), int(ids[j])))

    if len(pair_ids) == 0:
        return (
            np.array([], dtype=int).reshape(0, 2),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    return (
        np.array(pair_ids, dtype=int),
        np.array(trans_errors, dtype=np.float64),
        np.array(rot_errors_deg, dtype=np.float64),
    )


def compute_rpe(
    R_ref_wc: np.ndarray,
    C_ref_wc: np.ndarray,
    ids_ref: np.ndarray,
    R_est_wc: np.ndarray,
    C_est_wc: np.ndarray,
    ids_est: np.ndarray,
    delta: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Relative pose error using ``frame_idx + delta`` on common IDs (Endomapper-style)."""
    ref_idx = {int(i): k for k, i in enumerate(ids_ref)}
    est_idx = {int(i): k for k, i in enumerate(ids_est)}

    common_ids = sorted(set(ref_idx.keys()) & set(est_idx.keys()))
    if len(common_ids) == 0:
        raise ValueError("No common IDs between reference and estimated poses.")

    pair_ids_ref: list[tuple[int, int]] = []
    trans_errors: list[float] = []
    rot_errors_deg: list[float] = []

    def build_twc(R_wc: np.ndarray, C: np.ndarray) -> np.ndarray:
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_wc
        T[:3, 3] = C
        return T

    for i_id in common_ids:
        j_id = i_id + delta
        if j_id not in common_ids:
            continue

        T_ref_i = build_twc(R_ref_wc[ref_idx[i_id]], C_ref_wc[ref_idx[i_id]])
        T_ref_j = build_twc(R_ref_wc[ref_idx[j_id]], C_ref_wc[ref_idx[j_id]])
        T_est_i = build_twc(R_est_wc[est_idx[i_id]], C_est_wc[est_idx[i_id]])
        T_est_j = build_twc(R_est_wc[est_idx[j_id]], C_est_wc[est_idx[j_id]])

        T_ref_rel = np.linalg.inv(T_ref_i) @ T_ref_j
        T_est_rel = np.linalg.inv(T_est_i) @ T_est_j
        T_err = np.linalg.inv(T_ref_rel) @ T_est_rel

        e_trans = float(np.linalg.norm(T_err[:3, 3]))
        cos_angle = (np.trace(T_err[:3, :3]) - 1.0) / 2.0
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        e_rot_deg = float(np.degrees(np.arccos(cos_angle)))

        pair_ids_ref.append((i_id, j_id))
        trans_errors.append(e_trans)
        rot_errors_deg.append(e_rot_deg)

    if len(pair_ids_ref) == 0:
        return (
            np.array([], dtype=int).reshape(0, 2),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    return (
        np.array(pair_ids_ref, dtype=int),
        np.array(trans_errors, dtype=np.float64),
        np.array(rot_errors_deg, dtype=np.float64),
    )


def compute_rpe_metrics(
    R_ref_wc: np.ndarray,
    C_ref_wc: np.ndarray,
    ids_ref: np.ndarray,
    R_est_wc: np.ndarray,
    C_est_wc: np.ndarray,
    ids_est: np.ndarray,
    deltas: tuple[int, ...] = (1, 10, 20, 40),
    by_index: bool = True,
) -> dict[int, dict[str, float]]:
    out: dict[int, dict[str, float]] = {}
    for delta in deltas:
        if by_index:
            _, trans_rpe, rot_rpe_deg = compute_rpe_by_index(
                R_ref_wc, C_ref_wc, ids_ref,
                R_est_wc, C_est_wc,
                delta=delta,
            )
        else:
            _, trans_rpe, rot_rpe_deg = compute_rpe(
                R_ref_wc, C_ref_wc, ids_ref,
                R_est_wc, C_est_wc, ids_est,
                delta=delta,
            )
        if trans_rpe.size == 0:
            out[delta] = {
                "trans_mean": float("nan"),
                "trans_std": float("nan"),
                "rot_mean": float("nan"),
                "rot_std": float("nan"),
            }
        else:
            out[delta] = {
                "trans_mean": float(trans_rpe.mean()),
                "trans_std": float(trans_rpe.std()),
                "rot_mean": float(rot_rpe_deg.mean()),
                "rot_std": float(rot_rpe_deg.std()),
            }
    return out
