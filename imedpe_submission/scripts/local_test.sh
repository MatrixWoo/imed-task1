#!/usr/bin/env bash
# Run the built submission image the way the evaluation server will run it.
# Usage: ./scripts/local_test.sh <image_tag> <input_dir> <output_dir>
#
# Example:
#   docker build -t my-pose-submission:dev .
#   ./scripts/local_test.sh my-pose-submission:dev \
#       ~/imcpe_train_subset \
#       /tmp/my_test_output

set -euo pipefail

IMAGE="${1:?Usage: $0 <image_tag> <input_dir> <output_dir>}"
INPUT_DIR="${2:?missing input_dir}"
OUTPUT_DIR="${3:?missing output_dir}"

if [ ! -d "$INPUT_DIR" ]; then
  echo "ERROR: input_dir does not exist: $INPUT_DIR" >&2
  exit 2
fi
mkdir -p "$OUTPUT_DIR"

# Match the orchestrator's flags as closely as possible: sealed network,
# capped memory, read-only input, 10-min wall-clock budget.
timeout 660 docker run --rm \
  --gpus all \
  --network=none \
  --memory=20g \
  --pids-limit=512 \
  -v "$(realpath "$INPUT_DIR")":/input:ro \
  -v "$(realpath "$OUTPUT_DIR")":/output \
  "$IMAGE"

echo
echo "Outputs written to: $OUTPUT_DIR"
ls -R "$OUTPUT_DIR"