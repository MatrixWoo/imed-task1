from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def to_homogeneous(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = r
    T[:3, 3] = t.reshape(3)
    return T


def from_homogeneous(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return T[:3, :3], T[:3, 3]


def quaternion_xyzw_from_r(r: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(r).as_quat()


def umeyama_alignment(src_xyz: np.ndarray, dst_xyz: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_mean = src_xyz.mean(axis=0)
    dst_mean = dst_xyz.mean(axis=0)
    src_c = src_xyz - src_mean
    dst_c = dst_xyz - dst_mean

    cov = (dst_c.T @ src_c) / src_xyz.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt

    var_src = np.sum(src_c**2) / src_xyz.shape[0]
    scale = np.trace(np.diag(D) @ S) / var_src
    t = dst_mean - scale * R @ src_mean
    return float(scale), R, t
