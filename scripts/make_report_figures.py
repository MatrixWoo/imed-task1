"""Generate report figures for iMED-PE baseline results.

Figure 1: ATE summary horizontal bar chart (61 train sequences, sorted).
Figure 2: 3D trajectory overlays (GT vs Horn-aligned pred) for
          best / median / worst sequences.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("/home/wuzuoxu/official-imedpe")
sys.path.insert(0, str(ROOT / "src"))

from imcpe.alignment import horn_align_sim3
from imcpe.io_pose import read_pose_txt

# ---- palette (light mode, validated reference instance) ----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_GT = "#2a78d6"    # categorical slot 1 (blue)
SERIES_PRED = "#eb6834"  # categorical slot 2 (orange)

DATA_ROOT = Path("/home/wuzuoxu/Data/imed_pe")
PRED_ROOT = Path("/home/wuzuoxu/Data/imed_pe_preds")
FIG_DIR = Path("/home/wuzuoxu/Data/imed_pe_figures")
ATE_FILE = Path("/tmp/ate_pairs.txt")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def load_ate_pairs() -> list[tuple[str, float]]:
    pairs = []
    for line in ATE_FILE.read_text().strip().splitlines():
        name, val = line.rsplit(" ", 1)
        pairs.append((name, float(val)))
    return sorted(pairs, key=lambda p: p[1])


def fig1_ate_bars(pairs: list[tuple[str, float]]) -> None:
    names = [p[0] for p in pairs]
    vals = [p[1] for p in pairs]
    mean = float(np.mean(vals))

    fig, ax = plt.subplots(figsize=(9, 14))
    y = np.arange(len(names))
    ax.barh(y, vals, height=0.72, color=SERIES_GT, zorder=3)

    # mean reference line
    ax.axvline(mean, color=INK2, linewidth=1.2, linestyle="--", zorder=4)
    ax.text(max(vals) * 1.02, len(names) - 0.6, f"mean = {mean:.2f} mm",
            color=INK2, fontsize=9, va="top", ha="right")

    # selective direct labels: best / worst only
    ax.text(vals[0] + 0.06, y[0], f"{vals[0]:.2f}", color=INK,
            fontsize=9, va="center")
    ax.text(vals[-1] + 0.06, y[-1], f"{vals[-1]:.2f}", color=INK,
            fontsize=9, va="center")

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=7.5)
    ax.set_xlabel("ATE (mm)", fontsize=11)
    ax.set_title(
        "iMED-PE baseline: per-sequence ATE (train split, 61 sequences)",
        fontsize=12, color=INK, pad=12)
    ax.set_xlim(0, max(vals) * 1.12)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    out = FIG_DIR / "ate_summary.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def horn_align_xyz(gt_xyz: np.ndarray, pred_xyz: np.ndarray) -> np.ndarray:
    mask = np.isfinite(pred_xyz).all(axis=1) & np.isfinite(gt_xyz).all(axis=1)
    out = pred_xyz.copy()
    if mask.sum() < 3:
        return out
    model = pred_xyz[mask].T
    data = gt_xyz[mask].T
    rot, trans, _, scale = horn_align_sim3(model, data)
    out[mask] = (scale * rot @ model + trans).T
    return out


def load_xyz(seq_name: str, split: str) -> tuple[np.ndarray, np.ndarray]:
    gt_rows = read_pose_txt(DATA_ROOT / split / seq_name / "pose.txt")
    pred_rows = read_pose_txt(PRED_ROOT / split / seq_name / "pose.txt")
    gt = np.full((len(gt_rows), 3), np.nan)
    pred = np.full((len(pred_rows), 3), np.nan)
    for i, r in enumerate(gt_rows):
        if np.isfinite(r.t).all():
            gt[i] = r.t
    for i, r in enumerate(pred_rows):
        if np.isfinite(r.t).all():
            pred[i] = r.t
    return gt, pred


def fig2_trajectories(pairs: list[tuple[str, float]]) -> None:
    picks = [  # (label, sequence, ate)
        ("Best", pairs[0][0], pairs[0][1]),
        ("Median", pairs[len(pairs) // 2][0], pairs[len(pairs) // 2][1]),
        ("Worst", pairs[-1][0], pairs[-1][1]),
    ]

    fig = plt.figure(figsize=(15, 5.2))
    for i, (label, seq, ate) in enumerate(picks):
        gt, pred = load_xyz(seq, "train")
        pred_aligned = horn_align_xyz(gt, pred)
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot(gt[:, 0], gt[:, 1], gt[:, 2], color=SERIES_GT,
                linewidth=1.6, label="GT")
        ax.plot(pred_aligned[:, 0], pred_aligned[:, 1], pred_aligned[:, 2],
                color=SERIES_PRED, linewidth=1.6, label="Pred (aligned)")
        ax.set_title(f"{label}  —  ATE = {ate:.2f} mm", fontsize=11, pad=8)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor(SURFACE)
            axis.pane.set_edgecolor(GRID)
            axis.line.set_color(BASELINE)
            axis.set_tick_params(labelsize=8, colors=MUTED)
        ax.grid(True, color=GRID, linewidth=0.6)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               frameon=False, fontsize=10)
    fig.suptitle("iMED-PE baseline: trajectory overlay (Horn Sim(3) aligned)",
                 fontsize=12, y=1.0)
    fig.tight_layout()
    out = FIG_DIR / "trajectory_overlay.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    pairs = load_ate_pairs()
    fig1_ate_bars(pairs)
    fig2_trajectories(pairs)
