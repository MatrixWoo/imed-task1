from __future__ import annotations

import numpy as np


def horn_align_sim3(
    model: np.ndarray,
    data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Align model trajectory to data using Horn's method with scale (Sim3).

    Args:
        model: Predicted positions, shape (3, N).
        data: Ground-truth positions, shape (3, N).

    Returns:
        rot: (3, 3) rotation mapping model -> data frame.
        trans_gt: (3, 1) translation with scale applied.
        trans_error_gt: (N,) per-frame Euclidean error after scale alignment (mm).
        scale: Similarity scale factor.
    """
    model = np.asarray(model, dtype=np.float64)
    data = np.asarray(data, dtype=np.float64)

    if model.shape != data.shape:
        raise ValueError("model and data must have the same shape")
    if model.shape[0] != 3:
        raise ValueError("model and data must be 3xN")

    model_mean = model.mean(axis=1, keepdims=True)
    data_mean = data.mean(axis=1, keepdims=True)

    model_zc = model - model_mean
    data_zc = data - data_mean

    W = np.zeros((3, 3), dtype=np.float64)
    for i in range(model.shape[1]):
        W += np.outer(model_zc[:, i], data_zc[:, i])

    U, _, Vt = np.linalg.svd(W.T)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0

    rot = U @ S @ Vt

    rot_model = rot @ model_zc
    dots = np.sum(data_zc * rot_model)
    norms = np.sum(model_zc**2)
    scale = float(dots / norms) if norms > 0 else 1.0

    trans_gt = data_mean - scale * rot @ model_mean
    model_aligned = scale * rot @ model + trans_gt
    trans_error_gt = np.sqrt(np.sum((model_aligned - data) ** 2, axis=0))

    return rot, trans_gt, trans_error_gt, scale
