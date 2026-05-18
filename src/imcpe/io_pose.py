# io_pose.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PoseRow:
    frame_idx: int
    t: np.ndarray
    q_xyzw: np.ndarray


def read_pose_txt(path: Path) -> list[PoseRow]:
    rows: list[PoseRow] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        frame_idx = int(parts[0])
        t = np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=np.float64)
        q = np.array([float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])], dtype=np.float64)
        rows.append(PoseRow(frame_idx=frame_idx, t=t, q_xyzw=q))
    return rows


def write_pose_txt(path: Path, rows: list[PoseRow]) -> None:
    lines = []
    for row in rows:
        tx, ty, tz = row.t.tolist()
        qx, qy, qz, qw = row.q_xyzw.tolist()
        lines.append(
            f"{row.frame_idx} "
            f"{tx:.10f} {ty:.10f} {tz:.10f} "
            f"{qx:.10f} {qy:.10f} {qz:.10f} {qw:.10f}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
