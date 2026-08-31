# frame_candidates.py
"""Per-frame candidate poses (PnP + E fallback) with quality signals.

One GPU pass per sequence produces, for every frame:
  - pnp_pose / e_pose: candidate poses (or None)
  - pnp_inlier_rate: PnP RANSAC inlier rate (quality signal)
  - median_depth: median triangulated depth (scale harmonization for E)

The combination policy (fixed threshold vs per-sequence GMM) is applied
offline afterwards, so policy sweeps never touch the GPU again.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_io import SequenceData
from .matcher import ALikeLightGlueMatcher, RelativePoseResult, estimate_relative_pose
from .stereo_pnp import (
    common_keypoint_mask,
    estimate_pose_pnp,
    triangulate_points,
)


@dataclass
class FrameCandidates:
    frame_idx: int
    pnp_pose: RelativePoseResult | None
    e_pose: RelativePoseResult | None  # unit-scale translation
    pnp_inlier_rate: float
    median_depth: float | None
    cross_rate: float = 0.0  # independent e2-stereo consistency (0..1)
    cross_scale: float = float("nan")  # per-frame e2/e1 baseline scale ratio
    refined_pose: RelativePoseResult | None = None  # 3D-3D Umeyama refinement


def estimate_frame_candidates(
    seq: SequenceData,
    matcher: ALikeLightGlueMatcher,
    i: int,
    stereo_extrinsics: tuple[np.ndarray, np.ndarray] | None,
    e2_stereo_extrinsics: tuple[np.ndarray, np.ndarray] | None = None,
    min_points: int = 8,
) -> FrameCandidates:
    """Per-frame wrapper: match the three view pairs, then CPU geometry."""
    feats_l = matcher.extract(seq.e1_l_images[i])
    feats_2 = matcher.extract(seq.e2_l_images[i])
    pts_cross_l, pts_cross_2 = matcher.match_features(feats_l, feats_2)

    pts_e1_l = pts_e1_r = pts_e2_l = pts_e2_r = None
    if stereo_extrinsics is not None:
        feats_r = matcher.extract(seq.e1_r_images[i])
        pts_e1_l, pts_e1_r = matcher.match_features(feats_l, feats_r)
    if e2_stereo_extrinsics is not None:
        feats_2r = matcher.extract(seq.e2_r_images[i])
        pts_e2_l, pts_e2_r = matcher.match_features(feats_2, feats_2r)

    return candidates_from_matches(
        seq, i, stereo_extrinsics, e2_stereo_extrinsics,
        pts_cross_l, pts_cross_2, pts_e1_l, pts_e1_r, pts_e2_l, pts_e2_r,
        min_points=min_points,
    )


def candidates_from_matches(
    seq: SequenceData,
    i: int,
    stereo_extrinsics: tuple[np.ndarray, np.ndarray] | None,
    e2_stereo_extrinsics: tuple[np.ndarray, np.ndarray] | None,
    pts_cross_l: np.ndarray | None,
    pts_cross_2: np.ndarray | None,
    pts_e1_l: np.ndarray | None,
    pts_e1_r: np.ndarray | None,
    pts_e2_l: np.ndarray | None,
    pts_e2_r: np.ndarray | None,
    min_points: int = 8,
) -> FrameCandidates:
    """CPU geometry: build PnP / refined / E candidates from precomputed
    matches. Pure NumPy/OpenCV — safe to run after batched GPU matching."""
    from .stereo_pnp import cross_check_pose_3d3d, refine_pose_3d3d

    r_st, t_st = stereo_extrinsics or (None, None)

    pnp_pose = None
    e_pose = None
    pnp_rate = 0.0
    median_depth: float | None = None
    cross_rate = 0.0
    cross_scale = float("nan")
    refined_pose = None

    has_cross = pts_cross_l is not None and pts_cross_l.shape[0] >= min_points

    # --- PnP candidate ---
    if (
        stereo_extrinsics is not None
        and has_cross
        and pts_e1_l is not None
        and pts_e1_l.shape[0] >= min_points
    ):
        X, valid = triangulate_points(
            pts_e1_l, pts_e1_r, seq.k1_l, seq.k1_r, r_st, t_st
        )
        if valid.sum() >= min_points:
            median_depth = float(np.median(X[valid, 2]))
            matched, idx_cross = common_keypoint_mask(pts_e1_l[valid], pts_cross_l)
            if matched.sum() >= min_points:
                try:
                    pose = estimate_pose_pnp(
                        X[valid][matched],
                        pts_cross_2[idx_cross[matched]],
                        seq.k2_l,
                    )
                    pnp_pose = pose
                    pnp_rate = pose.num_inliers / float(matched.sum())

                    if (
                        e2_stereo_extrinsics is not None
                        and pts_e2_l is not None
                        and pts_e2_l.shape[0] >= min_points
                    ):
                        r_e2, t_e2 = e2_stereo_extrinsics
                        cross_rate, cross_scale, _ = cross_check_pose_3d3d(
                            pose,
                            X[valid][matched],
                            pts_cross_2[idx_cross[matched]],
                            pts_e2_l,
                            pts_e2_r,
                            seq.k2_l,
                            seq.k2_r,
                            r_e2,
                            t_e2,
                        )
                        try:
                            refined_pose = refine_pose_3d3d(
                                pose,
                                X[valid][matched],
                                pts_cross_2[idx_cross[matched]],
                                pts_e2_l,
                                pts_e2_r,
                                seq.k2_l,
                                seq.k2_r,
                                r_e2,
                                t_e2,
                            )
                        except RuntimeError:
                            refined_pose = None
                except RuntimeError:
                    pnp_pose = None

    # --- E candidate ---
    if has_cross:
        try:
            e_pose = estimate_relative_pose(
                pts_cross_l, pts_cross_2, seq.k1_l, seq.k2_l
            )
        except RuntimeError:
            e_pose = None

    return FrameCandidates(
        frame_idx=seq.frame_ids[i],
        pnp_pose=pnp_pose,
        e_pose=e_pose,
        pnp_inlier_rate=pnp_rate,
        median_depth=median_depth,
        cross_rate=cross_rate,
        cross_scale=cross_scale,
        refined_pose=refined_pose,
    )


def estimate_frame_candidates_reverse(
    seq: SequenceData,
    matcher: ALikeLightGlueMatcher,
    i: int,
    e2_stereo_extrinsics: tuple[np.ndarray, np.ndarray],
    min_points: int = 8,
) -> FrameCandidates:
    """Fallback when the e1 stereo extrinsics are unavailable: triangulate
    with the e2 stereo pair instead and solve the REVERSE PnP (3D in the e2_L
    frame -> 2D in e1_L), then invert the pose to e2-in-e1. The scale is in
    e2 baseline units; consistent within the sequence when e1 is broken
    everywhere (which is exactly the case this fallback serves).
    """
    from .stereo_pnp import common_keypoint_mask, estimate_pose_pnp, triangulate_points

    r_e2, t_e2 = e2_stereo_extrinsics

    feats_l = matcher.extract(seq.e1_l_images[i])
    feats_2 = matcher.extract(seq.e2_l_images[i])
    pts_cross_l, pts_cross_2 = matcher.match_features(feats_l, feats_2)

    pnp_pose = None
    e_pose = None
    median_depth: float | None = None

    if pts_cross_l.shape[0] >= min_points:
        feats_2r = matcher.extract(seq.e2_r_images[i])
        pts_st2_l, pts_st2_r = matcher.match_features(feats_2, feats_2r)
        if pts_st2_l.shape[0] >= min_points:
            X2, valid = triangulate_points(
                pts_st2_l, pts_st2_r, seq.k2_l, seq.k2_r, r_e2, t_e2
            )
            if valid.sum() >= min_points:
                median_depth = float(np.median(X2[valid, 2]))
                matched, idx_st = common_keypoint_mask(
                    pts_cross_2, pts_st2_l[valid]
                )
                if matched.sum() >= min_points:
                    try:
                        pose_rev = estimate_pose_pnp(
                            X2[valid][idx_st[matched]],
                            pts_cross_l[matched],
                            seq.k1_l,
                        )
                        # invert: pose_rev is e1-in-e2; we need e2-in-e1
                        R_inv = pose_rev.r.T
                        t_inv = -R_inv @ pose_rev.t
                        pnp_pose = RelativePoseResult(
                            r=R_inv,
                            t=t_inv,
                            num_matches=pose_rev.num_matches,
                            num_inliers=pose_rev.num_inliers,
                        )
                    except RuntimeError:
                        pnp_pose = None

        try:
            e_pose = estimate_relative_pose(
                pts_cross_l, pts_cross_2, seq.k1_l, seq.k2_l
            )
        except RuntimeError:
            e_pose = None

    return FrameCandidates(
        frame_idx=seq.frame_ids[i],
        pnp_pose=pnp_pose,
        e_pose=e_pose,
        pnp_inlier_rate=(
            pnp_pose.num_inliers / float(matched.sum())
            if pnp_pose is not None and matched.sum() > 0
            else 0.0
        ),
        median_depth=median_depth,
    )


def estimate_frame_candidates_loftr(
    seq: SequenceData,
    loftr,
    i: int,
    stereo_extrinsics: tuple[np.ndarray, np.ndarray] | None,
    e2_stereo_extrinsics: tuple[np.ndarray, np.ndarray] | None = None,
    min_points: int = 8,
    nn_tol: float = 2.0,
) -> FrameCandidates:
    """LoFTR variant: dense matches everywhere.

    LoFTR keypoints are pair-specific, so the PnP 2D-3D correspondences are
    built by nearest-neighbour lookup of the cross-camera e1/L points in the
    triangulated e1 stereo points (dense matches make the lookup reliable).
    """
    from .stereo_pnp import (
        common_keypoint_mask,
        cross_check_pose_3d3d,
        estimate_pose_pnp,
        refine_pose_3d3d,
        triangulate_points,
    )

    r_st, t_st = stereo_extrinsics or (None, None)

    # cross-camera dense matches
    k1_c, k2_c = loftr.match(seq.e1_l_images[i], seq.e2_l_images[i])

    pnp_pose = None
    e_pose = None
    pnp_rate = 0.0
    median_depth: float | None = None
    cross_rate = 0.0
    cross_scale = float("nan")
    refined_pose = None

    # --- PnP via dense stereo triangulation + NN depth lookup ---
    if stereo_extrinsics is not None and k1_c.shape[0] >= min_points:
        k1_s, kr_s = loftr.match(seq.e1_l_images[i], seq.e1_r_images[i])
        if k1_s.shape[0] >= min_points:
            X, valid = triangulate_points(
                k1_s, kr_s, seq.k1_l, seq.k1_r, r_st, t_st
            )
            k1_s_v, X_v = k1_s[valid], X[valid]
            if X_v.shape[0] >= min_points:
                median_depth = float(np.median(X_v[:, 2]))
                matched, idx_s = common_keypoint_mask(k1_c, k1_s_v, tol=nn_tol)
                if matched.sum() >= min_points:
                    try:
                        pose = estimate_pose_pnp(
                            X_v[idx_s[matched]], k2_c[matched], seq.k2_l
                        )
                        pnp_pose = pose
                        pnp_rate = pose.num_inliers / float(matched.sum())

                        if e2_stereo_extrinsics is not None:
                            r_e2, t_e2 = e2_stereo_extrinsics
                            k2_s, k2r_s = loftr.match(
                                seq.e2_l_images[i], seq.e2_r_images[i]
                            )
                            if k2_s.shape[0] >= min_points:
                                cross_rate, cross_scale, _ = cross_check_pose_3d3d(
                                    pose,
                                    X_v[idx_s[matched]],
                                    k2_c[matched],
                                    k2_s,
                                    k2r_s,
                                    seq.k2_l,
                                    seq.k2_r,
                                    r_e2,
                                    t_e2,
                                )
                                try:
                                    refined_pose = refine_pose_3d3d(
                                        pose,
                                        X_v[idx_s[matched]],
                                        k2_c[matched],
                                        k2_s,
                                        k2r_s,
                                        seq.k2_l,
                                        seq.k2_r,
                                        r_e2,
                                        t_e2,
                                    )
                                except RuntimeError:
                                    refined_pose = None
                    except RuntimeError:
                        pnp_pose = None

    # --- E candidate from dense cross matches ---
    if k1_c.shape[0] >= min_points:
        try:
            e_pose = estimate_relative_pose(k1_c, k2_c, seq.k1_l, seq.k2_l)
        except RuntimeError:
            e_pose = None

    return FrameCandidates(
        frame_idx=seq.frame_ids[i],
        pnp_pose=pnp_pose,
        e_pose=e_pose,
        pnp_inlier_rate=pnp_rate,
        median_depth=median_depth,
        cross_rate=cross_rate,
        cross_scale=cross_scale,
        refined_pose=refined_pose,
    )


def rescale_e_pose(e_pose: RelativePoseResult, depth: float) -> RelativePoseResult:
    """Rescale the unit-norm E translation to the metric frame."""
    if depth is None or not np.isfinite(depth) or depth <= 0:
        return e_pose
    out = RelativePoseResult(
        r=e_pose.r,
        t=e_pose.t * depth,
        num_matches=e_pose.num_matches,
        num_inliers=e_pose.num_inliers,
    )
    return out


def gmm_threshold(
    rates: list[float],
    n_components: int = 2,
    min_separation: float = 0.08,
    default: float = 0.15,
) -> float:
    """Self-supervised per-sequence threshold.

    Fits a 2-component Gaussian mixture to the frame inlier rates. When the
    components separate clearly (means differ by >= min_separation), the
    midpoint is the threshold; otherwise the default applies.
    """
    rates = np.asarray([r for r in rates if np.isfinite(r)], dtype=np.float64)
    if rates.size < 20:
        return default

    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(
        n_components=n_components, covariance_type="full", random_state=0
    )
    gmm.fit(rates.reshape(-1, 1))
    means = gmm.means_.reshape(-1)
    order = np.argsort(means)
    lo, hi = means[order[0]], means[order[1]]
    if hi - lo < min_separation:
        return default
    threshold = float((lo + hi) * 0.5)
    return max(0.05, min(threshold, 0.5))
