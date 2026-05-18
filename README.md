# iMED-PE baseline

<p align="center"><em>Example sequences with pose overlays</em></p>
<p align="center">
  <video src="assets/session_002_pig_intestine_zoom_in_trajectory_overlay.gif" width="32%" autoplay loop muted playsinline></video>
  <video src="assets/session_004_scene_2_circular_trajectory_overlay.gif" width="32%" autoplay loop muted playsinline></video>
  <video src="assets/session_007_scene_3_zoom_in_trajectory_overlay.gif" width="32%" autoplay loop muted playsinline></video>
</p>

Minimal baseline for iMED-PE trajectory estimation using ALIKED + LightGlue + essential matrix.

## Dataset splits

Sequences are organized under `train/` and `test/` for convenience (local development, ablations, and reporting numbers on data with public ground truth). You are welcome to **train on all released sequences** (`train` + `test` combined) if that helps your method—the challenge maintains a separate **held-out** set (`hidden_test/`) that is not part of either split and is used for final evaluation.

## Pose convention

Ground-truth `pose.txt` stores the trajectory of **endoscope2/L relative to endoscope1/L** as frame-to-initial transforms:

\[
T_{\mathrm{rel}}(t) = T_0^{-1}\, T(t)
\]

The first frame is identity. This is **not** single-camera temporal VO on `endoscope2/L` alone: in in-vivo use, endoscope1 can move (sometimes not insignificantly due to physiological movements from the subject), so one-camera relative motion couples scope motion into a drifting “world.” The task is same-time cross-camera pose between `endoscope1/L` and `endoscope2/L`.

## What it does

- Loads sequences from `train/` or `test/`.
- Matches `endoscope1/L` and `endoscope2/L` at the **same** frame index.
- Estimates cross-camera relative pose per frame (identity at frame 0).
- Writes predictions in `pose.txt` format: `frame_idx tx ty tz qx qy qz qw`.

## Install

```bash
cd <repo>
python -m pip install -r requirements.txt
```

## Run baseline

```bash
python scripts/run_baseline.py \
  --data-root <data-root> \
  --split train \
  --output-root <pred-root> \
  --device cuda
```

Outputs:

- `<pred-root>/train/<sequence_name>/pose.txt`
- `<pred-root>/test/<sequence_name>/pose.txt`

## Evaluate

We use the same metrics scripts as [CLiMB](https://www.synapse.org/Synapse:syn74370700/wiki/639986) for EndoVIS consistency. Huge thanks to the CLiMB team! 

```bash
python scripts/evaluate_ate.py \
  --data-root <data-root> \
  --split train \
  --pred-root <pred-root>
```

Uses Horn Sim(3) alignment (Endomapper-style) on translations, then reports:

- **ATE**: `mean_ate`, `std_ate`, `median_ate` (mm)
- **RPE** at frame deltas 1, 10, 20, 40: translational (mm) and rotational (deg)
- `num_matched_poses`, `registered_pct`

Optional JSON export: `--json-out results.json`

### Example baseline results (train split)

Cross-camera ALIKED + LightGlue + essential matrix baseline on **65** `train/` sequences (Horn Sim(3) alignment, same metrics as above):

| Metric | Value |
|--------|------:|
| Mean ATE | 2.18 mm |
| Mean of per-sequence median ATE | 2.06 mm |
| Mean std ATE (per sequence) | 1.13 mm |
| Registered frames | 100% |
| RPE trans / rot, δ=1 | 1.17 mm / 6.05° |
| RPE trans / rot, δ=10 | 3.13 mm / 7.45° |
| RPE trans / rot, δ=20 | 3.07 mm / 7.70° |
| RPE trans / rot, δ=40 | 3.40 mm / 7.76° |

These numbers are a reference point only; re-run `run_baseline.py` and `evaluate_ate.py` on your machine to reproduce.

## Expected dataset layout

```text
<data-root>/
  train/<sequence_name>/...
  test/<sequence_name>/...
  hidden_test/<sequence_name>/...   # held-out (not in train/test)
```

Each sequence directory contains:

```text
pose.txt
K.txt
endoscope1/L/frame_XXXXXX.png
endoscope1/R/frame_XXXXXX.png
endoscope2/L/frame_XXXXXX.png
endoscope2/R/frame_XXXXXX.png
```
