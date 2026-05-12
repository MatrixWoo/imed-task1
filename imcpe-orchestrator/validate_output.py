"""
Validate a submission's /output directory before scoring.

Catches the common malformed-submission failures cleanly, with specific error
messages we can forward to participants:
  - missing pose_predictions.txt for a sequence
  - wrong number of rows
  - unparseable rows
  - frame_idx mismatch with GT
  - non-unit quaternions
  - wrong-shape data

Returns (ok: bool, errors: list[str]). On failure the orchestrator marks the
submission INVALID and forwards `errors` to the participant.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

PRED_FILENAME = "pose_predictions.txt"
QUAT_NORM_TOLERANCE = 1e-3


def _expected_frame_ids(seq_dir: Path) -> list[int]:
    """Source frame IDs from endoscope2/L like the contract dictates."""
    e2l = seq_dir / "endoscope2" / "L"
    if not e2l.is_dir():
        return []
    ids = []
    for p in sorted(e2l.glob("frame_*.png")):
        try:
            ids.append(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return sorted(ids)


def _parse_pred_file(path: Path) -> tuple[list[tuple[int, np.ndarray, np.ndarray]], list[str]]:
    """Returns (rows, errors). Each row is (frame_idx, t_xyz, q_xyzw)."""
    errors: list[str] = []
    rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    text = path.read_text()
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 8:
            errors.append(f"{path.name}:{lineno}: expected 8 fields, got {len(parts)}")
            continue
        try:
            frame_idx = int(parts[0])
        except ValueError:
            errors.append(f"{path.name}:{lineno}: cannot parse frame_idx '{parts[0]}'")
            continue
        try:
            vals = [float(x) for x in parts[1:]]
        except ValueError as e:
            errors.append(f"{path.name}:{lineno}: cannot parse floats ({e})")
            continue
        t = np.array(vals[:3], dtype=np.float64)
        q = np.array(vals[3:], dtype=np.float64)
        rows.append((frame_idx, t, q))
    return rows, errors


def validate_submission(output_root: Path, test_data_root: Path) -> tuple[bool, list[str]]:
    """
    Validate /output structure against the test data layout.

    Rules enforced:
      1. For every sequence dir under test_data_root, output_root must contain
         <seq>/pose_predictions.txt.
      2. Each prediction file's rows have 8 fields (frame_idx + 3 + 4).
      3. Predicted frame_idx values are a subset of expected frame IDs.
      4. Finite quaternions must be approximately unit-norm.

    Permissive on:
      - extra rows (logged as a warning, not an error)
      - row order (frames matched by index downstream)
      - NaN values (legitimate "unregistered" signal)
    """
    errors: list[str] = []

    seq_dirs = sorted(p for p in test_data_root.iterdir() if p.is_dir())
    if not seq_dirs:
        return False, [f"No sequences found under {test_data_root}"]

    for seq_dir in seq_dirs:
        seq = seq_dir.name
        pred_path = output_root / seq / PRED_FILENAME
        if not pred_path.exists():
            errors.append(f"{seq}: missing {PRED_FILENAME}")
            continue

        expected_ids = set(_expected_frame_ids(seq_dir))
        if not expected_ids:
            errors.append(f"{seq}: no frames found in test_data (cannot validate)")
            continue

        rows, parse_errors = _parse_pred_file(pred_path)
        errors.extend(f"{seq}: {e}" for e in parse_errors)
        if not rows:
            errors.append(f"{seq}: prediction file contains no parseable rows")
            continue

        seen_ids: set[int] = set()
        for frame_idx, _, q in rows:
            if frame_idx in seen_ids:
                errors.append(f"{seq}: duplicate frame_idx {frame_idx}")
            seen_ids.add(frame_idx)
            if frame_idx not in expected_ids:
                errors.append(
                    f"{seq}: frame_idx {frame_idx} not present in test data"
                )
            if np.isfinite(q).all():
                qn = float(np.linalg.norm(q))
                if abs(qn - 1.0) > QUAT_NORM_TOLERANCE:
                    errors.append(
                        f"{seq}: frame {frame_idx} quaternion not unit-norm "
                        f"(|q|={qn:.4f})"
                    )

        missing = sorted(expected_ids - seen_ids)
        if missing:
            sample = ", ".join(str(i) for i in missing[:5])
            tail = "..." if len(missing) > 5 else ""
            errors.append(
                f"{seq}: missing predictions for {len(missing)} frame(s) "
                f"[{sample}{tail}]"
            )

    return len(errors) == 0, errors