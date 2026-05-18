# iMED Pose Estimation — Submission Template

This template builds a Docker image that the iMED challenge evaluator can run
against held-out test sequences. Fill in `predict.py`, build, push to Synapse,
submit.

## Quick start

1. **Clone this repo** and copy/move it to your project directory.
2. **Edit `requirements.txt`** — add the Python deps your method needs.
3. **Edit `predict.py`** — replace the `predict_sequence` stub with your method.
4. **Build and self-test** against the train split:
   ```bash
   docker build -t my-pose-submission:dev .
   ./scripts/local_test.sh my-pose-submission:dev \
       /path/to/iMED_pe/train \
       /tmp/my_test_output
   ```
5. **Submit to Synapse** (see "Submission" below).

## I/O contract

Your container is run with these mounts and flags:

```
docker run --rm --gpus all --network=none --memory=20g \
  -v <hidden_test_data>:/input:ro \
  -v <out_dir>:/output \
  <your_image>
```

### Input layout (`/input`, read-only)

```
/input/<sequence_name>/
    K.txt                      # 4 intrinsic matrices: K1_L, K1_R, K2_L, K2_R
    endoscope1/L/frame_*.png   # left  view, scope 1
    endoscope1/R/frame_*.png   # right view, scope 1
    endoscope2/L/frame_*.png   # left  view, scope 2
    endoscope2/R/frame_*.png   # right view, scope 2
```

Frame filenames are zero-padded indices like `frame_000008.png`. Indices are
not necessarily contiguous or starting from zero. **`pose.txt` is intentionally
absent at evaluation time.**

`K.txt` format:
```
# K1_L (endoscope1 left)
fx 0  cx
0  fy cy
0  0  1
# K1_R (endoscope1 right)
...
# K2_L (endoscope2 left)
...
# K2_R (endoscope2 right)
...
```

### Output layout (`/output`, writable)

For each sequence, write:
```
/output/<sequence_name>/pose_predictions.txt
```

One row per frame, whitespace-separated:
```
<frame_idx> <tx> <ty> <tz> <qx> <qy> <qz> <qw>
```

- **Anchor:** the first frame in `endoscope2/L` is the world origin. Its row
  must be `(t=0,0,0  q=0,0,0,1)`.
- **Pose convention:** world-from-camera. The 6-DoF pose places the camera
  in the world frame defined by frame 0.
- **Quaternion convention:** `(qx, qy, qz, qw)` — scipy `Rotation.as_quat()`
  default. Unit length.
- **Scale:** translation is **scale-ambiguous**. Scoring uses Sim(3) Umeyama
  alignment, so the absolute scale of your translations doesn't matter — only
  the trajectory shape.
- **Failed registrations:** write `nan` for every numeric field of that frame.

## Runtime constraints

| Constraint | Value |
|---|---|
| GPU | 1x NVIDIA RTX 4090 (24 GB VRAM) |
| CUDA | 11.8 host driver |
| RAM | 20 GB |
| Wall clock | 10 minutes total for all sequences |
| Network | **disabled** (`--network=none`) |
| Filesystem | read-only `/input`, writable `/output`, nothing else persists |

### Network is disabled

Your container will not have internet access at runtime. **Bake any model
weights into the image at build time** with a `RUN python -c "..."` step in
the Dockerfile, or `COPY` the weight files in. The most common first-submission
failure is a method that tries to download from HuggingFace or `torch.hub` at
runtime.

## Scoring

For each sequence, the evaluator computes:

- **ATE RMSE** (mm) after Sim(3) Umeyama alignment of your trajectory to GT
- **% registered** (rows with finite values / total GT frames)

Aggregate score is the mean over all test sequences.

## Submission

1. Tag your image for Synapse:
   ```bash
   docker tag my-pose-submission:dev \
       docker.synapse.org/syn74277461/<your_team>:v1
   ```
2. Log in (use a Synapse personal access token as the password):
   ```bash
   docker login docker.synapse.org
   ```
3. Push:
   ```bash
   docker push docker.synapse.org/syn74277461/<your_team>:v1
   ```
4. Submit the Docker entity to the challenge evaluation queue
   (Synapse web UI → "Submit to Challenge").

## Common failure modes

- **`Error: model weights not found`** — you forgot to bake weights at build time.
  Add a `RUN python -c "..."` step that triggers the download in your Dockerfile.
- **Output validation failed** — wrong filename (`pose.txt` instead of
  `pose_predictions.txt`), wrong frame count, wrong quaternion order.
- **Timeout** — your method took >10 min. The matcher is the usual bottleneck;
  reduce keypoint count or skip frames.
- **CUDA OOM** — drop batch size to 1 or use a smaller backbone.

## Need help?

Post in the challenge forum on Synapse, or open an issue in the template repo.