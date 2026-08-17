# temporal_smooth.py
"""Gaussian-window temporal smoothing for per-frame pose sequences.

The per-frame cross-camera baseline estimates every pose independently,
so the sequence carries white per-frame noise (mainly rotational). This
module smooths the frame-to-initial pose sequence in SE(3):

  - rotation: weighted quaternion average (sign-aligned) + normalize
  - translation: weighted mean
  - the frame-0-identity convention is restored afterwards by
    re-normalizing the whole sequence: T'(t) = S(0)^-1 S(t)

NaN (failed) frames are skipped inside windows and stay NaN in the output.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from .io_pose import PoseRow


def _is_valid(row: PoseRow) -> bool:
    return bool(np.isfinite(row.t).all()) and bool(np.isfinite(row.q_xyzw).all())


def _align_quaternion_signs(qs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Flip signs so every quaternion points to the same hemisphere."""
    out = qs.copy()
    ref = None
    for i in range(len(out)):
        if not valid[i]:
            continue
        if ref is None:
            ref = out[i]
            continue
        if float(np.dot(out[i], ref)) < 0.0:
            out[i] = -out[i]
        ref = out[i]
    return out


def smooth_pose_sequence(
    rows: list[PoseRow],
    window: int = 4,
    sigma: float = 2.0,
) -> list[PoseRow]:
    """Smooth a frame-to-initial pose sequence with a Gaussian window.

    Args:
        rows: per-frame poses (frame 0 is identity).
        window: half-width of the smoothing window (frames).
        sigma: Gaussian kernel width (frames).

    Returns:
        New PoseRow list with the same frame indices; first frame is
        restored to identity; NaN frames remain NaN.
    """
    n = len(rows)
    if n <= 2:
        return rows

    valid = np.array([_is_valid(r) for r in rows], dtype=bool)
    ts = np.full((n, 3), np.nan, dtype=np.float64)
    qs = np.full((n, 4), np.nan, dtype=np.float64)
    for i, r in enumerate(rows):
        if valid[i]:
            ts[i] = r.t
            qs[i] = r.q_xyzw

    # sign-align so q and -q are treated as the same rotation
    qs = _align_quaternion_signs(qs, valid)

    # precompute Gaussian weights for all lags in [-window, window]
    lags = np.arange(-window, window + 1, dtype=np.float64)
    weights = np.exp(-(lags**2) / (2.0 * sigma**2))

    ts_s = np.full((n, 3), np.nan, dtype=np.float64)
    qs_s = np.full((n, 4), np.nan, dtype=np.float64)

    for t in range(n):
        if not valid[t]:
            continue
        lo = max(0, t - window)
        hi = min(n - 1, t + window)
        idx = np.arange(lo, hi + 1)
        mask = valid[idx]
        w = weights[idx - t][mask]
        if w.sum() <= 0:
            continue

        ts_s[t] = (w[:, None] * ts[idx][mask]).sum(axis=0) / w.sum()
        q_mean = (w[:, None] * qs[idx][mask]).sum(axis=0)
        q_norm = np.linalg.norm(q_mean)
        if q_norm < 1e-12:
            qs_s[t] = np.array([0.0, 0.0, 0.0, 1.0])
        else:
            qs_s[t] = q_mean / q_norm

    # restore the frame-0-identity convention: T'(t) = S(0)^-1 S(t)
    first_valid = int(np.argmax(valid))
    R0 = Rotation.from_quat(qs_s[first_valid]).as_matrix()
    t0 = ts_s[first_valid]
    R0T = R0.T

    out: list[PoseRow] = []
    for i, r in enumerate(rows):
        if not valid[i]:
            out.append(r)
            continue
        R_i = Rotation.from_quat(qs_s[i]).as_matrix()
        R_new = R0T @ R_i
        t_new = R0T @ (ts_s[i] - t0)
        out.append(
            PoseRow(
                frame_idx=r.frame_idx,
                t=t_new,
                q_xyzw=Rotation.from_matrix(R_new).as_quat(),
            )
        )
    return out
