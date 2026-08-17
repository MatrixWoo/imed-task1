"""Comparison figure: baseline vs temporal smoothing (train + test splits)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# palette (validated light-mode reference)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_BASE = "#eb6834"    # orange: baseline
SERIES_SMOOTH = "#2a78d6"  # blue: + temporal smoothing

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


def main() -> None:
    data = json.loads(Path("/tmp/ate_comparison.json").read_text())
    fig_dir = Path(__file__).resolve().parent
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

    for ax, split in zip(axes, ["train", "test"]):
        base = data[f"{split}_base"]
        smooth = data[f"{split}_smooth"]
        names = list(base.keys())
        names_short = [n.replace("session_", "s").replace("_", " ")[:22] for n in names]
        x = np.arange(len(names))
        w = 0.38

        b = ax.bar(x - w / 2, [base[n] for n in names], width=w,
                   color=SERIES_BASE, label="Baseline (ALIKED+LG+E)", zorder=3)
        s = ax.bar(x + w / 2, [smooth[n] for n in names], width=w,
                   color=SERIES_SMOOTH, label="+ Temporal smoothing", zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(names_short, rotation=75, fontsize=6.5, ha="right")
        ax.set_ylabel("ATE (mm)", fontsize=11)
        ax.set_title(f"{split} split ({len(names)} sequences)", fontsize=12)

        mean_b = float(np.mean(list(base.values())))
        mean_s = float(np.mean(list(smooth.values())))
        gain = (mean_b - mean_s) / mean_b * 100
        ax.text(0.02, 0.96, f"mean: {mean_b:.2f} → {mean_s:.2f} mm ({gain:.1f}%)",
                transform=ax.transAxes, fontsize=10, color=INK,
                va="top", bbox=dict(facecolor=SURFACE, edgecolor=GRID, pad=4))

    handles = [axes[0].patches[0], axes[0].patches[1]]
    labels = ["Baseline", "+ Temporal smoothing"]
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=10)
    fig.suptitle("iMED-PE Task 1: per-sequence ATE, baseline vs temporal smoothing",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    out = fig_dir / "ate_baseline_vs_smooth.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
