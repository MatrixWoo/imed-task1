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


def _select_frame_pose(
    cand,
    threshold: float,
    seq_median_depth: float,
    seq_median_scale: float = float("nan"),
    max_scale_dev: float = 0.25,
) -> RelativePoseResult | None:
    """Pick the frame pose: the 3D-3D refined pose when available and its
    per-frame scale agrees with the sequence consensus (the two baselines are
    fixed, so s must be near-constant), else PnP (if its inlier rate clears
    the threshold), else the depth-rescaled E pose (scale stays consistent
    across frames)."""
    from .frame_candidates import rescale_e_pose

    if cand.refined_pose is not None and np.isfinite(seq_median_scale):
        s_dev = abs(cand.cross_scale - seq_median_scale) / seq_median_scale
        if s_dev <= max_scale_dev:
            return cand.refined_pose
    if cand.pnp_pose is not None and cand.pnp_inlier_rate >= threshold:
        return cand.pnp_pose
    if cand.e_pose is not None:
        depth = cand.median_depth if cand.median_depth is not None else seq_median_depth
        return rescale_e_pose(cand.e_pose, depth)
    return None


def predict_sequence_cross_camera(
    sequence_dir: Path,
    matcher: ALikeLightGlueMatcher,
    min_matches: int = 8,
    min_inliers: int = 8,
    smooth: bool = True,
    smooth_window: int = 2,
    smooth_sigma: float = 2.0,
    use_stereo_pnp: bool = True,
    use_loftr: bool = False,
    pnp_threshold: float = 0.0,
    smooth_method: str = "kalman",
    frame_skip: int = 1,
    dump_candidates: Path | None = None,
) -> tuple[list[PoseRow], dict]:
    """Cross-camera pose: endoscope2/L relative to endoscope1/L per frame.

    Frame 0 is identity. Per-frame candidates come from stereo triangulation
    (e1_L/e1_R fixed aggregated baseline -> metric-consistent scale) + PnP
    against the cross-camera matches, and from the essential-matrix path.
    The PnP candidate is selected when its RANSAC inlier rate clears
    `pnp_threshold`; otherwise the depth-rescaled E pose is used. When
    pnp_threshold is None, a per-sequence 2-component GMM on the frame
    inlier rates decides the threshold (self-supervised). Temporal smoothing
    (Gaussian window) suppresses per-frame white noise afterwards.

    With dump_candidates set, raw per-frame candidates are saved as .npz for
    offline policy sweeps.
    """
    from .frame_candidates import estimate_frame_candidates, gmm_threshold

    seq = load_sequence(sequence_dir)

    stereo_extrinsics = None
    e2_stereo_extrinsics = None
    if use_stereo_pnp:
        from .stereo_pnp import estimate_stereo_extrinsics

        e1_est = estimate_stereo_extrinsics(seq, matcher, camera="e1")
        if e1_est is not None:
            stereo_extrinsics = (e1_est[0], e1_est[1])
        e2_est = estimate_stereo_extrinsics(seq, matcher, camera="e2")
        if e2_est is not None:
            r_e2, t_e2, dispersion_e2 = e2_est
            # quality gate: unreliable e2 baseline -> skip refinement entirely
            if dispersion_e2 <= 12.0:
                e2_stereo_extrinsics = (r_e2, t_e2)

    # --- pass 1: compute per-frame candidates (GPU) ---
    loftr = None
    if use_loftr:
        from .frame_candidates import estimate_frame_candidates_loftr
        from .loftr_matcher import LoFTRMatcher

        loftr = LoFTRMatcher(device=str(matcher.device))

    cands: list = [None]  # frame 0
    if loftr is not None:
        for i in range(1, len(seq.frame_ids)):
            cands.append(
                estimate_frame_candidates_loftr(
                    seq, loftr, i, stereo_extrinsics,
                    e2_stereo_extrinsics=e2_stereo_extrinsics,
                    min_points=min_matches,
                )
            )
    else:
        # per-frame path: sequential extraction/matching with point pruning
        # enabled (batched matching breaks accuracy on weak-texture scenes).
        # frame_skip > 1 estimates every k-th frame and leaves the others as
        # NaN; the Kalman/RTS smoother interpolates them (halves compute for
        # frame_skip=2 at train mean ATE 1.098 -> 1.214).
        for i in range(1, len(seq.frame_ids)):
            if frame_skip > 1 and (i - 1) % frame_skip != 0:
                cands.append(None)  # skipped frame -> NaN, RTS interpolates
                continue
            cands.append(
                estimate_frame_candidates(
                    seq, matcher, i, stereo_extrinsics,
                    e2_stereo_extrinsics=e2_stereo_extrinsics,
                    min_points=min_matches,
                )
            )

    # --- policy: one threshold for the whole sequence ---
    # Default 0.0 = always prefer PnP when available (best on train: 1.5468;
    # frame-level threshold tuning was measured to be a wash, see
    # /tmp/sweep_thresholds.py). GMM self-supervision remains available via
    # pnp_threshold=None.
    threshold = (
        gmm_threshold([c.pnp_inlier_rate for c in cands if c is not None])
        if pnp_threshold is None
        else pnp_threshold
    )
    seq_depth = _seq_median_depth(cands)
    scales = [
        c.cross_scale
        for c in cands
        if c is not None and np.isfinite(c.cross_scale)
    ]
    seq_median_scale = float(np.median(scales)) if scales else float("nan")

    if dump_candidates is not None:
        _dump_candidates(dump_candidates, cands)

    # --- pass 2: combine candidates (CPU) ---
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

        cand = cands[i]
        if cand is None:  # skipped frame (frame_skip): RTS interpolates
            pred_rows.append(_nan_row(frame_id))
            registered_flags.append(False)
            continue
        pose = _select_frame_pose(
            cand, threshold, seq_depth, seq_median_scale
        )

        if pose is None or pose.num_inliers < min_inliers:
            pred_rows.append(_nan_row(frame_id))
            registered_flags.append(False)
            continue

        T = to_homogeneous(pose.r, pose.t)
        q_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
        t_xyz = T[:3, 3]
        pred_rows.append(PoseRow(frame_idx=frame_id, t=t_xyz, q_xyzw=q_xyzw))
        registered_flags.append(True)

    if smooth:
        if smooth_method == "kalman":
            from .kalman_smooth import kalman_smooth_pose_sequence

            pred_rows = kalman_smooth_pose_sequence(pred_rows)
        else:
            from .temporal_smooth import smooth_pose_sequence

            pred_rows = smooth_pose_sequence(
                pred_rows, window=smooth_window, sigma=smooth_sigma
            )

    stats = {
        "n_frames": len(seq.frame_ids),
        "registered_pct": 100.0 * float(np.mean(registered_flags)),
    }
    return pred_rows, stats


def _seq_median_depth(cands: list) -> float:
    depths = [
        c.median_depth
        for c in cands
        if c is not None and c.median_depth is not None
    ]
    if not depths:
        return 1.0
    return float(np.median(depths))


def _dump_candidates(path: Path, cands: list) -> None:
    """Save per-frame candidates for offline threshold sweeps."""
    n = len(cands)
    has_pnp = np.zeros(n, dtype=bool)
    pnp_r = np.zeros((n, 3, 3), dtype=np.float64)
    pnp_t = np.zeros((n, 3), dtype=np.float64)
    has_e = np.zeros(n, dtype=bool)
    e_r = np.zeros((n, 3, 3), dtype=np.float64)
    e_t = np.zeros((n, 3), dtype=np.float64)
    rates = np.zeros(n, dtype=np.float64)
    depths = np.full(n, np.nan, dtype=np.float64)
    cross_rates = np.zeros(n, dtype=np.float64)
    cross_scales = np.full(n, np.nan, dtype=np.float64)
    has_ref = np.zeros(n, dtype=bool)
    ref_r = np.zeros((n, 3, 3), dtype=np.float64)
    ref_t = np.zeros((n, 3), dtype=np.float64)

    for i, c in enumerate(cands):
        if c is None:
            continue
        if c.pnp_pose is not None:
            has_pnp[i] = True
            pnp_r[i] = c.pnp_pose.r
            pnp_t[i] = c.pnp_pose.t
            rates[i] = c.pnp_inlier_rate
        if c.e_pose is not None:
            has_e[i] = True
            e_r[i] = c.e_pose.r
            e_t[i] = c.e_pose.t
        if c.median_depth is not None:
            depths[i] = c.median_depth
        cross_rates[i] = c.cross_rate
        cross_scales[i] = c.cross_scale
        if c.refined_pose is not None:
            has_ref[i] = True
            ref_r[i] = c.refined_pose.r
            ref_t[i] = c.refined_pose.t

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        has_pnp=has_pnp, pnp_r=pnp_r, pnp_t=pnp_t,
        has_e=has_e, e_r=e_r, e_t=e_t,
        rates=rates, depths=depths,
        cross_rates=cross_rates, cross_scales=cross_scales,
        has_ref=has_ref, ref_r=ref_r, ref_t=ref_t,
    )
