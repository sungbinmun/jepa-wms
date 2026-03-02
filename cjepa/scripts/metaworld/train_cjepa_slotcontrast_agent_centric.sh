#!/usr/bin/env bash
set -euo pipefail

# Run from cjepa root:
#   cd /home/sungbinmun/jepa-wms/cjepa
#   bash scripts/metaworld/train_cjepa_slotcontrast_agent_centric.sh

export PYTHONPATH="$(pwd)"

SLOTCONTRAST_CKPT="${SLOTCONTRAST_CKPT:-/home/sungbinmun/jepa-wms/slotcontrast/logs/metaworld_mv_agseg/slotcontrast/metaworld_dinov3_512_mv_1/checkpoints/step=10000.ckpt}"
SLOTCONTRAST_CFG="${SLOTCONTRAST_CFG:-/home/sungbinmun/jepa-wms/slotcontrast/logs/metaworld_mv_agseg/slotcontrast/metaworld_dinov3_512_mv_1/settings.yaml}"
VIDEO_KEY="${VIDEO_KEY:-video_gripper}"

OUT_PREFIX="${OUT_PREFIX:-./outputs/metaworld_slotcontrast_mv_step10000}"
SLOTPATH="${SLOTPATH:-${OUT_PREFIX}_slots.pkl}"
ACTION_PATH="${ACTION_PATH:-${OUT_PREFIX}_actions.pkl}"
PROPRIO_PATH="${PROPRIO_PATH:-${OUT_PREFIX}_proprio.pkl}"
STATE_PATH="${STATE_PATH:-${OUT_PREFIX}_states.pkl}"

if [[ ! -f "${SLOTPATH}" || ! -f "${ACTION_PATH}" || ! -f "${PROPRIO_PATH}" || ! -f "${STATE_PATH}" ]]; then
  echo "[INFO] Missing slot/action/proprio/state pkl. Extracting from SlotContrast checkpoint..."
  python3 src/custom_codes/extract_metaworld_slotcontrast_slots.py \
    --checkpoint "${SLOTCONTRAST_CKPT}" \
    --slotcontrast-config "${SLOTCONTRAST_CFG}" \
    --video-key "${VIDEO_KEY}" \
    --output-prefix "${OUT_PREFIX}" \
    --device cuda
fi

python3 src/train/train_causalwm_agent_centric_from_pusht_slot.py \
  --config-name config_train_causal_agent_centric_pusht_slot \
  output_model_name="metaworld_cjepa_agent_centric_slotcontrast" \
  embedding_dir="${SLOTPATH}" \
  action_dir="${ACTION_PATH}" \
  proprio_dir="${PROPRIO_PATH}" \
  state_dir="${STATE_PATH}" \
  dinowm.action_dim=4 \
  dinowm.proprio_dim=4 \
  videosaur.NUM_SLOTS=7 \
  videosaur.SLOT_DIM=64 \
  agent_centric.action_embed_dim=64 \
  trainer.devices=1 \
  trainer.strategy=auto
