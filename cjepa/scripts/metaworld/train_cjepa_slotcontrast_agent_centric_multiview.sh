#!/usr/bin/env bash
set -euo pipefail

# Run from cjepa root:
#   cd /home/sungbinmun/jepa-wms/cjepa
#   bash scripts/metaworld/train_cjepa_slotcontrast_agent_centric_multiview.sh

export PYTHONPATH="$(pwd)"

SLOTCONTRAST_CKPT="${SLOTCONTRAST_CKPT:-/home/sungbinmun/jepa-wms/slotcontrast/logs/metaworld_mv_agseg/slotcontrast/metaworld_dinov3_512_mv_1/checkpoints/step=10000.ckpt}"
SLOTCONTRAST_CFG="${SLOTCONTRAST_CFG:-/home/sungbinmun/jepa-wms/slotcontrast/logs/metaworld_mv_agseg/slotcontrast/metaworld_dinov3_512_mv_1/settings.yaml}"

OUT_PREFIX="${OUT_PREFIX:-./outputs/metaworld_slotcontrast_mv_step10000}"
GRIPPER_PREFIX="${GRIPPER_PREFIX:-${OUT_PREFIX}_gripper}"
THIRD_PREFIX="${THIRD_PREFIX:-${OUT_PREFIX}_third}"

SLOT_GRIPPER_PATH="${SLOT_GRIPPER_PATH:-${GRIPPER_PREFIX}_slots.pkl}"
SLOT_THIRD_PATH="${SLOT_THIRD_PATH:-${THIRD_PREFIX}_slots.pkl}"
ACTION_PATH="${ACTION_PATH:-${GRIPPER_PREFIX}_actions.pkl}"
PROPRIO_PATH="${PROPRIO_PATH:-${GRIPPER_PREFIX}_proprio.pkl}"
STATE_PATH="${STATE_PATH:-${GRIPPER_PREFIX}_states.pkl}"

if [[ ! -f "${SLOT_GRIPPER_PATH}" || ! -f "${ACTION_PATH}" || ! -f "${PROPRIO_PATH}" || ! -f "${STATE_PATH}" ]]; then
  echo "[INFO] Missing gripper-view slots or aligned action/proprio/state pkl. Extracting..."
  python3 src/custom_codes/extract_metaworld_slotcontrast_slots.py \
    --checkpoint "${SLOTCONTRAST_CKPT}" \
    --slotcontrast-config "${SLOTCONTRAST_CFG}" \
    --video-key "video_gripper" \
    --output-prefix "${GRIPPER_PREFIX}" \
    --device cuda
fi

if [[ ! -f "${SLOT_THIRD_PATH}" ]]; then
  echo "[INFO] Missing third-view slots pkl. Extracting..."
  python3 src/custom_codes/extract_metaworld_slotcontrast_slots.py \
    --checkpoint "${SLOTCONTRAST_CKPT}" \
    --slotcontrast-config "${SLOTCONTRAST_CFG}" \
    --video-key "video_third" \
    --output-prefix "${THIRD_PREFIX}" \
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

python3 src/train/train_causalwm_agent_centric_multiview_from_pusht_slot.py \
  --config-name config_train_causal_agent_centric_metaworld_multiview_slot \
  output_model_name="metaworld_cjepa_agent_centric_multiview_slotcontrast" \
  embedding_dir_gripper="${SLOT_GRIPPER_PATH}" \
  embedding_dir_third="${SLOT_THIRD_PATH}" \
  action_dir="${ACTION_PATH}" \
  proprio_dir="${PROPRIO_PATH}" \
  state_dir="${STATE_PATH}" \
  dinowm.action_dim=4 \
  dinowm.proprio_dim=4 \
  videosaur.NUM_SLOTS=7 \
  videosaur.SLOT_DIM=64 \
  multiview.num_views=2 \
  trainer.devices=1 \
  trainer.strategy=auto \
  "${WANDB_OVERRIDES[@]}"
