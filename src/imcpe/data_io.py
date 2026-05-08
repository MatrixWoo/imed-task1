from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .io_pose import PoseRow, read_pose_txt


@dataclass
class SequenceData:
    sequence_name: str
    frame_ids: list[int]
    k1_l: np.ndarray
    k1_r: np.ndarray
    k2_l: np.ndarray
    k2_r: np.ndarray
    e1_l_images: list[Path]
    e1_r_images: list[Path]
    e2_l_images: list[Path]
    e2_r_images: list[Path]
    gt_rows: list[PoseRow] | None  # None when pose.txt is absent (test time)


_REQUIRED_K_KEYS = ("K1_L", "K1_R", "K2_L", "K2_R")


def _read_k_txt(k_path: Path) -> dict[str, np.ndarray]:
    """Parse K.txt with `# <NAME>` headers followed by 3x3 matrices."""
    lines = [ln.strip() for ln in k_path.read_text().splitlines() if ln.strip()]
    mats: dict[str, np.ndarray] = {}
    current: str | None = None
    rows: list[list[float]] = []
    for line in lines:
        if line.startswith("#"):
            payload = line.lstrip("#").strip()
            current = payload.split()[0] if payload else None
            rows = []
            continue
        if current is None:
            continue  # tolerate stray lines before any header
        rows.append([float(x) for x in line.split()])
        if len(rows) == 3:
            mats[current] = np.array(rows, dtype=np.float64)
            current = None
            rows = []

    missing = [k for k in _REQUIRED_K_KEYS if k not in mats]
    if missing:
        raise RuntimeError(f"{k_path} is missing entries: {missing}")
    return mats


def _image_path_map(img_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in sorted(img_dir.glob("frame_*.png")):
        try:
            frame_id = int(p.stem.split("_")[1])
        except (IndexError, ValueError) as e:
            raise RuntimeError(f"Cannot parse frame index from {p.name}") from e
        out[frame_id] = p
    return out


def load_sequence(sequence_dir: Path) -> SequenceData:
    """
    Load images, intrinsics, and (if present) GT poses for a sequence.

    Frame IDs come from `endoscope2/L/frame_*.png`. The other three cameras must
    contain matching frame IDs or this raises. `pose.txt` is loaded if present
    and exposed as `gt_rows`; missing pose.txt is allowed (test-time use).
    """
    intrinsics = _read_k_txt(sequence_dir / "K.txt")

    e1_l_map = _image_path_map(sequence_dir / "endoscope1" / "L")
    e1_r_map = _image_path_map(sequence_dir / "endoscope1" / "R")
    e2_l_map = _image_path_map(sequence_dir / "endoscope2" / "L")
    e2_r_map = _image_path_map(sequence_dir / "endoscope2" / "R")

    if not e2_l_map:
        raise RuntimeError(f"No frames found in {sequence_dir}/endoscope2/L")

    frame_ids = sorted(e2_l_map.keys())

    for name, m in (("endoscope1/L", e1_l_map),
                    ("endoscope1/R", e1_r_map),
                    ("endoscope2/R", e2_r_map)):
        if sorted(m.keys()) != frame_ids:
            missing = sorted(set(frame_ids) - set(m.keys()))
            extra = sorted(set(m.keys()) - set(frame_ids))
            raise RuntimeError(
                f"Frame ID mismatch in {sequence_dir.name}: endoscope2/L vs {name}. "
                f"Missing in {name}: {missing[:5]}{'...' if len(missing) > 5 else ''}. "
                f"Extra in {name}: {extra[:5]}{'...' if len(extra) > 5 else ''}."
            )

    gt_path = sequence_dir / "pose.txt"
    gt_rows = read_pose_txt(gt_path) if gt_path.exists() else None

    return SequenceData(
        sequence_name=sequence_dir.name,
        frame_ids=frame_ids,
        k1_l=intrinsics["K1_L"],
        k1_r=intrinsics["K1_R"],
        k2_l=intrinsics["K2_L"],
        k2_r=intrinsics["K2_R"],
        e1_l_images=[e1_l_map[i] for i in frame_ids],
        e1_r_images=[e1_r_map[i] for i in frame_ids],
        e2_l_images=[e2_l_map[i] for i in frame_ids],
        e2_r_images=[e2_r_map[i] for i in frame_ids],
        gt_rows=gt_rows,
    )


def read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not load image: {path}")
    return img