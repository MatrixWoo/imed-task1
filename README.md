# iMED_PE_baseline

Minimal baseline for iMED-PE trajectory estimation using ALIKED + LightGlue + Essential matrix.

## What it does

- Loads sequences from `train/` or `test/`.
- Uses `endoscope2/L` consecutive frames to estimate relative motion.
- Chains relative poses into a trajectory normalized at first frame.
- Writes predictions in `pose.txt` format:
  - `frame_idx tx ty tz qx qy qz qw`
- Evaluates ATE with Umeyama alignment in a separate script.

## Install

```bash
cd /raid/scratch_not_backed_up/sbonilla/iMED/iMED_pe/iMED_PE_baseline
python -m pip install -r requirements.txt
```

## Run baseline

```bash
python scripts/run_baseline.py \
  --data-root {} \
  --split train \
  --output-root {}
  predictions \
  --device cuda
```

Outputs are written to:

- `predictions/train/<sequence_name>/pose.txt`
- or `predictions/test/<sequence_name>/pose.txt`

## Evaluate ATE

```bash
python scripts/evaluate_ate.py \
  --data-root {} \
  --split train \
  --pred-root {}
```

Printed metrics:

- `ATE_RMSE` (mm, after Umeyama alignment)
- `% registered frames`
- runtime summary

## Expected dataset layout

Each sequence under `train/` or `test/` is expected as:

```text
session_xxx_scene_y_<trajectory>/
  pose.txt
  K.txt
  endoscope1/L/frame_XXXXXX.png
  endoscope1/R/frame_XXXXXX.png
  endoscope2/L/frame_XXXXXX.png
  endoscope2/R/frame_XXXXXX.png
```
