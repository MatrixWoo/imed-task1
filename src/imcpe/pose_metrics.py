from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .alignment import horn_align_sim3
from .io_pose import PoseRow, read_pose_txt
from .rpe import compute_rpe_metrics

RPE_DELTAS = (1, 10, 20, 40)


def _row_valid(row: PoseRow) -> bool:
    return bool(np.isfinite(row.t).all() and np.isfinite(row.q_xyzw).all())


def pose_rows_to_poses(rows: list[PoseRow]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert pose rows to rotations, centers, ids, and per-row validity mask."""
    n = len(rows)
    R = np.zeros((n, 3, 3), dtype=np.float64)
    C = np.zeros((n, 3), dtype=np.float64)
    ids = np.zeros(n, dtype=int)
    valid = np.zeros(n, dtype=bool)

    for i, row in enumerate(rows):
        ids[i] = row.frame_idx
        if not _row_valid(row):
            continue
        valid[i] = True
        R[i] = Rotation.from_quat(row.q_xyzw).as_matrix()
        C[i] = row.t

    return R, C, ids, valid


def build_correspondence(
    gt_rows: list[PoseRow],
    pred_rows: list[PoseRow],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Build matched GT/pred trajectories on common frame_idx with finite predictions."""
    gt_dict = {r.frame_idx: r for r in gt_rows}
    pred_dict = {r.frame_idx: r for r in pred_rows}
    common_ids = sorted(set(gt_dict.keys()) & set(pred_dict.keys()))

    gt_pts: list[np.ndarray] = []
    pred_pts: list[np.ndarray] = []
    gt_R: list[np.ndarray] = []
    gt_C: list[np.ndarray] = []
    pred_R: list[np.ndarray] = []
    pred_C: list[np.ndarray] = []
    matched_ids: list[int] = []

    for fid in common_ids:
        g = gt_dict[fid]
        p = pred_dict[fid]
        if not _row_valid(g) or not _row_valid(p):
            continue
        gt_pts.append(g.t)
        pred_pts.append(p.t)
        gt_R.append(Rotation.from_quat(g.q_xyzw).as_matrix())
        gt_C.append(g.t)
        pred_R.append(Rotation.from_quat(p.q_xyzw).as_matrix())
        pred_C.append(p.t)
        matched_ids.append(fid)

    if len(matched_ids) < 3:
        raise ValueError(f"Not enough matched poses (only {len(matched_ids)}).")

    return (
        np.array(pred_pts, dtype=np.float64).T,
        np.array(gt_pts, dtype=np.float64).T,
        np.array(pred_R, dtype=np.float64),
        np.array(pred_C, dtype=np.float64),
        np.array(gt_R, dtype=np.float64),
        np.array(gt_C, dtype=np.float64),
        np.array(matched_ids, dtype=int),
    )


def evaluate_sequence(gt_rows: list[PoseRow], pred_rows: list[PoseRow]) -> dict[str, float | int]:
    num_frames = len(gt_rows)
    registered_pct = 100.0 * sum(_row_valid(r) for r in pred_rows) / max(num_frames, 1)

    try:
        model, data, pred_R, pred_C, gt_R, gt_C, matched_ids = build_correspondence(gt_rows, pred_rows)
    except ValueError:
        return _empty_result(num_frames, registered_pct)

    rot, trans_gt, trans_error_gt, scale = horn_align_sim3(model, data)

    pred_R_aligned = np.array([rot @ R for R in pred_R])
    pred_C_aligned = (scale * rot @ pred_C.T + trans_gt).T

    rpe = compute_rpe_metrics(
        gt_R, gt_C, matched_ids,
        pred_R_aligned, pred_C_aligned, matched_ids,
        deltas=RPE_DELTAS,
    )

    result: dict[str, float | int] = {
        "mean_ate": float(trans_error_gt.mean()),
        "std_ate": float(trans_error_gt.std()),
        "median_ate": float(np.median(trans_error_gt)),
        "num_matched_poses": int(model.shape[1]),
        "num_frames": num_frames,
        "registered_pct": float(registered_pct),
        "scale": float(scale),
    }

    for delta in RPE_DELTAS:
        result[f"trans_rpe_{delta}frame"] = rpe[delta]["trans_mean"]
        result[f"rot_rpe_deg_{delta}frame"] = rpe[delta]["rot_mean"]

    result["mean_trans_rpe_40frame"] = rpe[40]["trans_mean"]
    result["std_trans_rpe_40frame"] = rpe[40]["trans_std"]
    result["mean_rot_rpe_deg_40frame"] = rpe[40]["rot_mean"]
    result["std_rot_rpe_deg_40frame"] = rpe[40]["rot_std"]

    return result


def _empty_result(num_frames: int, registered_pct: float) -> dict[str, float | int]:
    nan = float("nan")
    result: dict[str, float | int] = {
        "mean_ate": nan,
        "std_ate": nan,
        "median_ate": nan,
        "num_matched_poses": 0,
        "num_frames": num_frames,
        "registered_pct": registered_pct,
        "scale": nan,
        "mean_trans_rpe_40frame": nan,
        "std_trans_rpe_40frame": nan,
        "mean_rot_rpe_deg_40frame": nan,
        "std_rot_rpe_deg_40frame": nan,
    }
    for delta in RPE_DELTAS:
        result[f"trans_rpe_{delta}frame"] = nan
        result[f"rot_rpe_deg_{delta}frame"] = nan
    return result


def format_sequence_line(seq_name: str, r: dict[str, float | int]) -> str:
    return (
        f"{seq_name}: mean_ate={r['mean_ate']:.4f} std_ate={r['std_ate']:.4f} "
        f"median_ate={r['median_ate']:.4f} mm | "
        f"matched={r['num_matched_poses']}/{r['num_frames']} "
        f"({r['registered_pct']:.1f}% registered)"
    )


def format_rpe_lines(r: dict[str, float | int]) -> list[str]:
    lines = []
    for delta in RPE_DELTAS:
        lines.append(
            f"  RPE d={delta:2d}: trans={r[f'trans_rpe_{delta}frame']:.4f} mm  "
            f"rot={r[f'rot_rpe_deg_{delta}frame']:.4f} deg"
        )
    return lines


def evaluate_split(
    data_root: Path,
    pred_root: Path,
    split: str,
) -> tuple[dict[str, dict[str, float | int]], dict[str, float]]:
    split_dir = data_root / split
    pred_split_dir = pred_root / split
    sequence_dirs = sorted([p for p in split_dir.iterdir() if p.is_dir()])

    per_seq: dict[str, dict[str, float | int]] = {}
    for seq_dir in sequence_dirs:
        gt_rows = read_pose_txt(seq_dir / "pose.txt")
        pred_path = pred_split_dir / seq_dir.name / "pose.txt"
        if not pred_path.exists():
            per_seq[seq_dir.name] = _empty_result(len(gt_rows), 0.0)
            continue
        pred_rows = read_pose_txt(pred_path)
        per_seq[seq_dir.name] = evaluate_sequence(gt_rows, pred_rows)

    valid = [r for r in per_seq.values() if r["num_matched_poses"] > 0]
    if not valid:
        global_mean = {k: float("nan") for k in ["mean_ate", "std_ate", "median_ate", "num_sequences"]}
        return per_seq, global_mean

    global_mean: dict[str, float] = {
        "mean_ate": float(np.mean([r["mean_ate"] for r in valid])),
        "std_ate": float(np.mean([r["std_ate"] for r in valid])),
        "median_ate": float(np.mean([r["median_ate"] for r in valid])),
        "num_sequences": float(len(valid)),
    }
    for delta in RPE_DELTAS:
        global_mean[f"trans_rpe_{delta}frame"] = float(
            np.mean([r[f"trans_rpe_{delta}frame"] for r in valid])
        )
        global_mean[f"rot_rpe_deg_{delta}frame"] = float(
            np.mean([r[f"rot_rpe_deg_{delta}frame"] for r in valid])
        )
    global_mean["mean_trans_rpe_40frame"] = float(np.mean([r["mean_trans_rpe_40frame"] for r in valid]))
    global_mean["mean_rot_rpe_deg_40frame"] = float(np.mean([r["mean_rot_rpe_deg_40frame"] for r in valid]))

    return per_seq, global_mean
