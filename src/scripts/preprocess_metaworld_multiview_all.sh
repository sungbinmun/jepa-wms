#!/usr/bin/env bash
set -euo pipefail

# Full Metaworld preprocessing runner.
# - Re-renders third/gripper views
# - Writes instance/category/robot/non-robot masks into parquet shards
# - Does NOT generate any visualization mp4 files

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${JEPAWM_DSET:-}" ]]; then
  echo "ERROR: JEPAWM_DSET is not set."
  echo "Set it first, e.g. export JEPAWM_DSET=/path/to/datasets"
  exit 1
fi

INPUT_DIR="${INPUT_DIR:-${JEPAWM_DSET}/Metaworld/data}"
OUTPUT_DIR="${OUTPUT_DIR:-${JEPAWM_DSET}/Metaworld_multiview_full/data}"
CACHE_DIR="${CACHE_DIR:-/tmp/metaworld_hf_cache}"

# Rendering/data options
THIRD_CAMERA="${THIRD_CAMERA:-corner2}"
GRIPPER_CAMERA="${GRIPPER_CAMERA:-gripperPOV}"
THIRD_CAM_POS="${THIRD_CAM_POS:-0.75,0.075,0.7}"
WIDTH="${WIDTH:-224}"
HEIGHT="${HEIGHT:-224}"
FPS="${FPS:-20}"
FRAME_OFFSET="${FRAME_OFFSET:-0}"
SHARD_SIZE="${SHARD_SIZE:-64}"

# Optional slicing for distributed/multi-job preprocessing
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-}"

# Optional behavior flags (set to 1 to enable)
OVERWRITE="${OVERWRITE:-0}"
SKIP_FAILED="${SKIP_FAILED:-1}"

ARGS=(
  src/scripts/preprocess_metaworld_multiview.py
  --input-dir "${INPUT_DIR}"
  --output-dir "${OUTPUT_DIR}"
  --cache-dir "${CACHE_DIR}"
  --third-camera "${THIRD_CAMERA}"
  --gripper-camera "${GRIPPER_CAMERA}"
  --third-cam-pos "${THIRD_CAM_POS}"
  --width "${WIDTH}"
  --height "${HEIGHT}"
  --fps "${FPS}"
  --frame-offset "${FRAME_OFFSET}"
  --shard-size "${SHARD_SIZE}"
  --start-index "${START_INDEX}"
)

if [[ -n "${END_INDEX}" ]]; then
  ARGS+=(--end-index "${END_INDEX}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
  ARGS+=(--overwrite)
fi
if [[ "${SKIP_FAILED}" == "1" ]]; then
  ARGS+=(--skip-failed)
fi

echo "Running full Metaworld preprocessing"
echo "  input:  ${INPUT_DIR}"
echo "  output: ${OUTPUT_DIR}"
echo "  cache:  ${CACHE_DIR}"
echo "  range:  [${START_INDEX}, ${END_INDEX:-END})"
echo "  shard_size: ${SHARD_SIZE}"

uv run python "${ARGS[@]}"

