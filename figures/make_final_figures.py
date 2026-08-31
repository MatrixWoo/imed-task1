"""Final-method figures: 4-stage ablation + per-sequence baseline vs final."""
import json
import subprocess
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FIG_DIR = Path(__file__).resolve().parent
PY = "/home/wuzuoxu/miniconda3/envs/fomo_env/bin/python"
REPO = "/home/wuzuoxu/official-imedpe"
DATA = "/home/wuzuoxu/Data/imed_pe"

# palette (validated light-mode reference)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES_BASE = "#eb6834"
SERIES_FINAL = "#2a78d6"
SERIES_STAGE = "#3987e5"

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


def per_seq_ate(pred_root: str, split: str) -> dict:
    out = subprocess.run(
        [PY, f"{REPO}/scripts/evaluate_ate.py", "--data-root", DATA,
         "--split", split, "--pred-root", pred_root],
        capture_output=True, text=True, cwd=REPO,
    ).stdout
    return {m.group(1): float(m.group(2)) for line in out.splitlines()
            if (m := re.match(r"(session\S+): mean_ate=([0-9.]+)", line))}


def fig_ablation() -> None:
    """Four-stage ablation, train mean ATE."""
    stages = ["Baseline", "+ Gaussian\nsmoothing", "+ Stereo\nPnP",
              "+ 3D-3D\nrefinement", "+ Kalman\n+RTS"]
    train = [2.163, 1.760, 1.547, 1.271, 1.098]
    test = [2.508, 2.043, 1.868, 1.514, 1.301]

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(stages))
    w = 0.36
    b1 = ax.bar(x - w / 2, train, width=w, color=SERIES_BASE, label="Train (61 seqs)", zorder=3)
    b2 = ax.bar(x + w / 2, test, width=w, color=SERIES_FINAL, label="Test (19 seqs)", zorder=3)

    for rect, v in zip(b1, train):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.04, f"{v:.2f}",
                ha="center", fontsize=8.5, color=INK)
    for rect, v in zip(b2, test):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.04, f"{v:.2f}",
                ha="center", fontsize=8.5, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(stages, fontsize=9)
    ax.set_ylabel("Mean ATE (mm)", fontsize=11)
    ax.set_title("iMED-PE Task 1: four-stage ablation (mean ATE)", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.set_ylim(0, 2.9)
    fig.tight_layout()
    out = FIG_DIR / "ablation_four_stage.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def fig_per_sequence() -> None:
    """Per-sequence baseline vs final, train + test."""
    base_train = per_seq_ate("/home/wuzuoxu/Data/imed_pe_preds", "train")
    final_train = per_seq_ate("/home/wuzuoxu/Data/imed_pe_preds_gmm", "train")
    base_test = per_seq_ate("/home/wuzuoxu/Data/imed_pe_preds_nosmooth", "test")
    final_test = per_seq_ate("/home/wuzuoxu/Data/imed_pe_preds_final_test", "test")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, split, base, final in zip(
        axes, ["train", "test"],
        [base_train, base_test], [final_train, final_test],
    ):
        names = list(base.keys())
        names_short = [n.replace("session_", "").replace("_", " ")[:20] for n in names]
        x = np.arange(len(names))
        w = 0.38
        ax.bar(x - w / 2, [base[n] for n in names], width=w,
               color=SERIES_BASE, label="Baseline", zorder=3)
        ax.bar(x + w / 2, [final[n] for n in names], width=w,
               color=SERIES_FINAL, label="Ours (final)", zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(names_short, rotation=75, fontsize=6.5, ha="right")
        ax.set_ylabel("ATE (mm)", fontsize=11)
        mb = float(np.mean(list(base.values())))
        mf = float(np.mean(list(final.values())))
        gain = (mb - mf) / mb * 100
        ax.text(0.02, 0.96, f"mean: {mb:.2f} → {mf:.2f} mm ({gain:.1f}%)",
                transform=ax.transAxes, fontsize=10, color=INK, va="top",
                bbox=dict(facecolor=SURFACE, edgecolor=GRID, pad=4))
        ax.set_title(f"{split} split ({len(names)} sequences)", fontsize=12)

    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center",
               ncol=2, frameon=False, fontsize=10)
    fig.suptitle("iMED-PE Task 1: per-sequence ATE, baseline vs final method",
                 fontsize=13, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = FIG_DIR / "per_sequence_final.png"
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    fig_ablation()
    fig_per_sequence()
