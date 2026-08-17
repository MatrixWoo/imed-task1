# stereo_pnp.py
"""Stereo triangulation + PnP pose estimation for the iMED-PE task.

The two cameras of each endoscope are rigidly mounted, so their relative
transform [R_st | t_st] is constant over a sequence. We aggregate per-frame
E-matrix estimates of the endoscope-1 stereo pair into one fixed baseline,
triangulate matched stereo points (metric-consistent scale across the whole
sequence), then solve PnP against the cross-camera correspondences. This
removes the per-frame scale ambiguity of the pure essential-matrix baseline.

Scale note: the true stereo baseline length is unknown, so the reconstructed
trajectory has one constant global scale; the evaluator's Sim(3) alignment
absorbs it.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .data_io import SequenceData
from .matcher import ALikeLightGlueMatcher, RelativePoseResult, estimate_relative_pose


def estimate_stereo_extrinsics(
    seq: SequenceData,
    matcher: ALikeLightGlueMatcher,
    camera: str = "e1",
    sample_frames: int = 15,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Estimate the fixed L <- R stereo extrinsics [R_st | t_st_unit].

    `camera` selects the endoscope ("e1" or "e2"). Samples frames across
    the sequence, estimates E per frame, and averages rotations (sign-aligned
    quaternions) and translation directions.
    Returns (R, t_unit, t_dispersion_deg) or None if too few frames yield
    valid estimates. The dispersion is the mean angular deviation of the
    per-frame t directions from the aggregate — a quality signal (large
    dispersion = unreliable baseline estimate).
    """
    if camera == "e1":
        l_images, r_images = seq.e1_l_images, seq.e1_r_images
        k_l, k_r = seq.k1_l, seq.k1_r
    elif camera == "e2":
        l_images, r_images = seq.e2_l_images, seq.e2_r_images
        k_l, k_r = seq.k2_l, seq.k2_r
    else:
        raise ValueError(f"Unknown camera: {camera}")

    n = len(seq.frame_ids)
    indices = np.linspace(0, n - 1, min(sample_frames, n)).astype(int)

    Rs: list[np.ndarray] = []
    ts: list[np.ndarray] = []
    for i in indices:
        try:
            pts_l, pts_r = matcher.match(l_images[i], r_images[i])
            if pts_l.shape[0] < 8:
                continue
            pose = estimate_relative_pose(pts_l, pts_r, k_l, k_r)
            if pose.num_inliers < 8:
                continue
            Rs.append(pose.r)
            ts.append(pose.t)
        except RuntimeError:
            continue

    if len(Rs) < 3:
        return None

    Rs = np.stack(Rs)
    ts = np.stack(ts)

    # sign-aligned quaternion mean for R
    qs = Rotation.from_matrix(Rs).as_quat()
    ref = qs[0]
    for k in range(1, len(qs)):
        if float(np.dot(qs[k], ref)) < 0.0:
            qs[k] = -qs[k]
        ref = qs[k]
    R_mean = Rotation.from_quat(qs.mean(axis=0)).as_matrix()

    # sign-aligned unit-vector mean for t direction
    t_ref = ts[0]
    for k in range(1, len(ts)):
        if float(np.dot(ts[k], t_ref)) < 0.0:
            ts[k] = -ts[k]
    t_mean = ts.mean(axis=0)
    t_mean = t_mean / np.linalg.norm(t_mean)

    # quality: angular deviation of per-frame directions from the aggregate
    cos_dev = np.clip(np.abs(ts @ t_mean), -1.0, 1.0)
    dispersion_deg = float(np.degrees(np.arccos(cos_dev)).mean())

    return R_mean, t_mean, dispersion_deg


def triangulate_points(
    pts_l: np.ndarray,
    pts_r: np.ndarray,
    k_l: np.ndarray,
    k_r: np.ndarray,
    r_st: np.ndarray,
    t_st: np.ndarray,
    baseline: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Triangulate stereo matches into 3D points in the e1_L frame.

    Returns (X, valid): X has shape (N, 3); valid marks points with positive
    depth in both cameras.
    """
    p1 = k_l @ np.hstack([np.eye(3), np.zeros((3, 1))])
    t = (baseline * t_st).reshape(3, 1)
    p2 = k_r @ np.hstack([r_st, t])

    X4 = cv2.triangulatePoints(
        p1.astype(np.float64),
        p2.astype(np.float64),
        pts_l.T.astype(np.float64),
        pts_r.T.astype(np.float64),
    )
    X = (X4[:3] / X4[3]).T  # (N, 3), in e1_L frame

    depth1 = X[:, 2]
    X2 = (r_st @ X.T + t).T
    depth2 = X2[:, 2]
    valid = (depth1 > 0.0) & (depth2 > 0.0)
    return X, valid


def estimate_pose_pnp(
    obj_pts: np.ndarray,
    img_pts: np.ndarray,
    k_cam: np.ndarray,
    ransac_thresh: float = 2.0,
) -> RelativePoseResult:
    """Solve PnP: 3D points in the e1_L frame -> 2D points in e2_L.

    Raises RuntimeError on failure.
    """
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        obj_pts.astype(np.float64),
        img_pts.astype(np.float64),
        k_cam.astype(np.float64),
        None,
        iterationsCount=300,
        reprojectionError=ransac_thresh,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None or len(inliers) < 6:
        raise RuntimeError("PnP RANSAC failed.")

    inlier_idx = inliers.reshape(-1)
    # refine on inliers with iterative (Levenberg-Marquardt) minimization
    ok, rvec, tvec = cv2.solvePnP(
        obj_pts[inlier_idx],
        img_pts[inlier_idx],
        k_cam.astype(np.float64),
        None,
        rvec,
        tvec,
        useExtrinsicGuess=True,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not ok:
        raise RuntimeError("PnP refinement failed.")

    R = cv2.Rodrigues(rvec)[0]
    return RelativePoseResult(
        r=R.astype(np.float64),
        t=tvec.reshape(3).astype(np.float64),
        num_matches=int(obj_pts.shape[0]),
        num_inliers=int(len(inlier_idx)),
    )


def cross_check_pose_3d3d(
    pose: RelativePoseResult,
    x_e1: np.ndarray,
    e2l_matched: np.ndarray,
    e2_st_l: np.ndarray,
    e2_st_r: np.ndarray,
    k2_l: np.ndarray,
    k2_r: np.ndarray,
    r_e2st: np.ndarray,
    t_e2st: np.ndarray,
    rel_thresh: float = 0.15,
    min_points: int = 6,
) -> tuple[float, float, int]:
    """Independent consistency check of a PnP pose against e2 stereo depth.

    Two independent reconstructions of the same physical points, both in the
    e2_L camera frame, are compared up to a global scale s (the two stereo
    baselines differ by a fixed, unknown ratio):

      X_pred = pose.R @ X_e1 + pose.t          (e1 triangulation + PnP)
      X_st   = triangulate(e2_L, e2_R)         (e2 stereo, its own units)

    Returns (cross_rate, scale_s, num_checked). cross_rate is the fraction
    of points whose relative residual |X_st - s*X_pred| / |X_st| is below
    rel_thresh. The pose was never fitted to e2 stereo data, so this is
    genuine independent evidence.
    """
    # e2_L points common to the cross-camera set and the e2 stereo set
    matched, idx_st = common_keypoint_mask(e2l_matched, e2_st_l)
    if matched.sum() < min_points:
        return 0.0, float("nan"), int(matched.sum())

    # triangulate the e2 stereo points (e2_L frame, e2 baseline units)
    X_st, valid = triangulate_points(
        e2_st_l[idx_st[matched]],
        e2_st_r[idx_st[matched]],
        k2_l,
        k2_r,
        r_e2st,
        t_e2st,
    )
    m = matched.copy()
    m[m] = valid
    if m.sum() < min_points:
        return 0.0, float("nan"), int(m.sum())

    # PnP prediction of the same points in the e2_L frame
    X_pred = (pose.r @ x_e1[m].T + pose.t.reshape(3, 1)).T
    X_st = X_st[valid]

    # robust per-frame scale between the two reconstructions
    n_pts = X_st.shape[0]
    ratio = np.full(n_pts, np.nan)
    nonzero = np.abs(X_pred[:, 2]) > 1e-6
    ratio[nonzero] = X_st[nonzero, 2] / X_pred[nonzero, 2]
    if not np.isfinite(ratio).any():
        return 0.0, float("nan"), n_pts
    s = float(np.median(ratio[np.isfinite(ratio)]))

    # relative residual in depth-normalized 3D
    denom = np.linalg.norm(X_st, axis=1)
    res = np.linalg.norm(X_st - s * X_pred, axis=1) / np.maximum(denom, 1e-6)
    rate = float((res < rel_thresh).mean())
    return rate, s, n_pts


def refine_pose_3d3d(
    pose: RelativePoseResult,
    x_e1: np.ndarray,
    e2l_matched: np.ndarray,
    e2_st_l: np.ndarray,
    e2_st_r: np.ndarray,
    k2_l: np.ndarray,
    k2_r: np.ndarray,
    r_e2st: np.ndarray,
    t_e2st: np.ndarray,
    min_points: int = 6,
) -> RelativePoseResult:
    """Refine a PnP pose with 3D-3D Umeyama alignment against e2 stereo depth.

    The PnP pose predicts each e1-triangulated point in the e2_L frame
    (X_pred); the e2 stereo pair triangulates the same points independently
    (X_st, its own units). Umeyama solves X_st = s*R*X_pred + t; the refined
    pose in e1 units is [R @ R_pnp | R @ t_pnp + t/s].
    """
    from .geometry import umeyama_alignment

    matched, idx_st = common_keypoint_mask(e2l_matched, e2_st_l)
    if matched.sum() < min_points:
        raise RuntimeError("Too few common points for 3D-3D refinement.")

    X_st, valid = triangulate_points(
        e2_st_l[idx_st[matched]],
        e2_st_r[idx_st[matched]],
        k2_l,
        k2_r,
        r_e2st,
        t_e2st,
    )
    m = matched.copy()
    m[m] = valid
    if m.sum() < min_points:
        raise RuntimeError("Too few valid points for 3D-3D refinement.")

    X_pred = (pose.r @ x_e1[m].T + pose.t.reshape(3, 1)).T  # e2_L frame, e1 units
    X_st = X_st[valid]
    s, R, t = umeyama_alignment(X_pred, X_st)

    if s <= 1e-9 or not np.isfinite(s):
        raise RuntimeError("Degenerate 3D-3D refinement scale.")

    r_new = R @ pose.r
    t_new = R @ pose.t + t / s
    return RelativePoseResult(
        r=r_new,
        t=t_new,
        num_matches=pose.num_matches,
        num_inliers=int(m.sum()),
    )


def common_keypoint_mask(
    pts_query: np.ndarray,
    pts_target: np.ndarray,
    tol: float = 0.5,
) -> np.ndarray:
    """For each row of pts_query find the index of the (nearly) identical row
    in pts_target. ALIKED extraction is deterministic, so the same image
    yields bit-identical keypoints across matcher calls.

    Returns (matched, idx): matched is a boolean mask over pts_query,
    idx holds the corresponding row indices in pts_target (garbage where
    matched is False).
    """
    tree = cKDTree(pts_target)
    dist, idx = tree.query(pts_query, distance_upper_bound=tol)
    matched = np.isfinite(dist)
    return matched, idx
