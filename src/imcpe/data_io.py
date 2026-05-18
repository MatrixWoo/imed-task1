from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .io_pose import PoseRow, read_pose_txt


@dataclass
class SequenceData:
    """One iMED-PE sequence.

    ``gt_rows`` holds endoscope2/L relative to endoscope1/L as frame-to-initial
    poses (T_rel(t) = T_0^{-1} T(t)); the first frame is identity.
    """
    sequence_name: str
    frame_ids: list[int]
    gt_rows: list[PoseRow]
    k1_l: np.ndarray
    k2_l: np.ndarray
    e1_l_images: list[Path]
    e2_l_images: list[Path]


def _read_k_txt(k_path: Path) -> dict[str, np.ndarray]:
    lines = [line.strip() for line in k_path.read_text().splitlines() if line.strip()]
    mats: dict[str, np.ndarray] = {}
    current = None
    rows: list[list[float]] = []
    for line in lines:
        if line.startswith("#"):
            if current is not None:
                mats[current] = np.array(rows, dtype=np.float64)
            current = line.replace("#", "").strip().split()[0]
            rows = []
            continue
        rows.append([float(x) for x in line.split()])
        if len(rows) == 3 and current is not None:
            mats[current] = np.array(rows, dtype=np.float64)
            current = None
            rows = []
    return mats


def _image_path_map(img_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in sorted(img_dir.glob("frame_*.png")):
        frame_id = int(p.stem.split("_")[1])
        out[frame_id] = p
    return out


def load_sequence(sequence_dir: Path) -> SequenceData:
    sequence_name = sequence_dir.name
    gt_rows = read_pose_txt(sequence_dir / "pose.txt")
    frame_ids = [r.frame_idx for r in gt_rows]

    intrinsics = _read_k_txt(sequence_dir / "K.txt")
    k1_l = intrinsics["K1_L"]
    k2_l = intrinsics["K2_L"]

    e1_l_map = _image_path_map(sequence_dir / "endoscope1" / "L")
    e2_l_map = _image_path_map(sequence_dir / "endoscope2" / "L")
    e1_l_images = [e1_l_map[i] for i in frame_ids]
    e2_l_images = [e2_l_map[i] for i in frame_ids]

    return SequenceData(
        sequence_name=sequence_name,
        frame_ids=frame_ids,
        gt_rows=gt_rows,
        k1_l=k1_l,
        k2_l=k2_l,
        e1_l_images=e1_l_images,
        e2_l_images=e2_l_images,
    )


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not load image: {path}")
    return img
