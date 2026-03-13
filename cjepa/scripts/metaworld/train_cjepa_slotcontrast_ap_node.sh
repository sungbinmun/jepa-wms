#!/usr/bin/env bash
set -euo pipefail

# Can be run from anywhere:
#   bash /home/sungbin/jepa-wms/cjepa/scripts/metaworld/train_cjepa_slotcontrast_ap_node.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CJEPA_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${CJEPA_ROOT}/.." && pwd)"

cd "${CJEPA_ROOT}"
export PYTHONPATH="${CJEPA_ROOT}"

SLOTCONTRAST_CKPT="${SLOTCONTRAST_CKPT:-${REPO_ROOT}/_local/checkpoints/slotcontrast/metaworld_dinov3_512_mv/checkpoints/step=93000.ckpt}"
SLOTCONTRAST_CFG="${SLOTCONTRAST_CFG:-${REPO_ROOT}/_local/checkpoints/slotcontrast/metaworld_dinov3_512_mv/settings.yaml}"
VIDEO_KEY="${VIDEO_KEY:-video}"

OUT_PREFIX="${OUT_PREFIX:-./outputs/metaworld_slotcontrast_exterior_step10000}"
SLOTPATH="${SLOTPATH:-${OUT_PREFIX}_slots.pkl}"
ACTION_PATH="${ACTION_PATH:-${OUT_PREFIX}_actions.pkl}"
PROPRIO_PATH="${PROPRIO_PATH:-${OUT_PREFIX}_proprio.pkl}"
STATE_PATH="${STATE_PATH:-${OUT_PREFIX}_states.pkl}"

if [[ ! -f "${SLOTPATH}" || ! -f "${ACTION_PATH}" || ! -f "${PROPRIO_PATH}" || ! -f "${STATE_PATH}" ]]; then
  echo "[INFO] Missing slot/action/proprio/state pkl. Extracting from SlotContrast checkpoint..."
  if ! python3 -c "import pyarrow.parquet" >/dev/null 2>&1; then
    echo "[ERROR] Missing Python dependency: pyarrow"
    echo "[ERROR] Install it in the active environment, then rerun."
    exit 1
  fi
  python3 src/custom_codes/extract_metaworld_slotcontrast_slots.py \
    --checkpoint "${SLOTCONTRAST_CKPT}" \
    --slotcontrast-config "${SLOTCONTRAST_CFG}" \
    --video-key "${VIDEO_KEY}" \
    --output-prefix "${OUT_PREFIX}" \
    --device cuda
fi

WANDB_OVERRIDES=()
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  WANDB_OVERRIDES+=(wandb.entity="${WANDB_ENTITY}")
fi
if [[ -n "${WANDB_PROJECT:-}" ]]; then
  WANDB_OVERRIDES+=(wandb.project="${WANDB_PROJECT}")
fi
if [[ -n "${WANDB_NAME:-}" ]]; then
  WANDB_OVERRIDES+=(wandb.name="${WANDB_NAME}")
fi

python3 src/train/train_causalwm_AP_node_pusht_slot.py \
  --config-name config_train_causal_metaworld_ap_node_slot \
  output_model_name="metaworld_cjepa_apnode_v1" \
  embedding_dir="${SLOTPATH}" \
  action_dir="${ACTION_PATH}" \
  proprio_dir="${PROPRIO_PATH}" \
  state_dir="${STATE_PATH}" \
  dinowm.action_dim=4 \
  dinowm.proprio_dim=4 \
  videosaur.NUM_SLOTS=7 \
  videosaur.SLOT_DIM=64 \
  trainer.devices=1 \
  trainer.strategy=auto \
  "${WANDB_OVERRIDES[@]}"
