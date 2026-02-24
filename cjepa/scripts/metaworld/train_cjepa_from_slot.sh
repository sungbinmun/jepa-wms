#!/usr/bin/env bash
set -euo pipefail

# Run this script from the cjepa repo root:
#   cd /home/sungbinmun/jepa-wms/cjepa
#   bash scripts/metaworld/train_cjepa_from_slot.sh

export PYTHONPATH="$(pwd)"

# =========================
# Required paths
# =========================
# Expected format for each pkl:
# {
#   "train": {"episode_key": np.ndarray, ...},
#   "val":   {"episode_key": np.ndarray, ...}
# }
#
# slots:   [T, S, D]
# actions: [T_raw, A]           (raw per-step actions)
# proprio: [T_raw_or_T, P]
# state:   [T_raw_or_T, X]      (optional)

SLOTPATH="${SLOTPATH:-/path/to/metaworld_slots.pkl}"
ACTION_PATH="${ACTION_PATH:-/path/to/metaworld_actions.pkl}"
PROPRIO_PATH="${PROPRIO_PATH:-/path/to/metaworld_proprio.pkl}"
STATE_PATH="${STATE_PATH:-}"

# Placeholder OC checkpoint used only to instantiate videosaur modules for ckpt compatibility.
# If you do not want to load one, keep this as "null".
OC_CKPT_PATH="${OC_CKPT_PATH:-null}"

# =========================
# Core training hyperparams
# =========================
OUTPUT_MODEL_NAME="${OUTPUT_MODEL_NAME:-metaworld_cjepa}"
MAX_EPOCHS="${MAX_EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-256}"
NUM_WORKERS="${NUM_WORKERS:-8}"

# Temporal setup
HISTORY_SIZE="${HISTORY_SIZE:-4}"
NUM_PREDS="${NUM_PREDS:-1}"
FRAMESKIP="${FRAMESKIP:-5}"

# Slot setup (must match your slotcontrast export)
NUM_SLOTS="${NUM_SLOTS:-7}"
SLOT_DIM="${SLOT_DIM:-64}"
NUM_MASKED_SLOTS="${NUM_MASKED_SLOTS:-2}"

# Control dims for Metaworld
ACTION_DIM="${ACTION_DIM:-4}"
PROPRIO_DIM="${PROPRIO_DIM:-4}"
ACTION_EMBED_DIM="${ACTION_EMBED_DIM:-10}"
PROPRIO_EMBED_DIM="${PROPRIO_EMBED_DIM:-20}"

# Optimizer
PREDICTOR_LR="${PREDICTOR_LR:-5e-4}"
ACTION_ENCODER_LR="${ACTION_ENCODER_LR:-5e-4}"
PROPRIO_ENCODER_LR="${PROPRIO_ENCODER_LR:-5e-4}"

# Matching
USE_HUNGARIAN_MATCHING="${USE_HUNGARIAN_MATCHING:-true}"
HUNGARIAN_COST_TYPE="${HUNGARIAN_COST_TYPE:-mse}"
DRY_RUN="${DRY_RUN:-false}"

# =========================
# Optional auto-alignment
# =========================
# Metaworld often has T_slots = T_actions + 1. If true, trim each episode to common length.
AUTO_ALIGN_PKLS="${AUTO_ALIGN_PKLS:-true}"
ALIGNED_DIR="${ALIGNED_DIR:-./outputs/metaworld_aligned_pkls}"

if [[ ! -f "${SLOTPATH}" ]]; then
  echo "[ERROR] SLOTPATH not found: ${SLOTPATH}" >&2
  exit 1
fi
if [[ ! -f "${ACTION_PATH}" ]]; then
  echo "[ERROR] ACTION_PATH not found: ${ACTION_PATH}" >&2
  exit 1
fi
if [[ ! -f "${PROPRIO_PATH}" ]]; then
  echo "[ERROR] PROPRIO_PATH not found: ${PROPRIO_PATH}" >&2
  exit 1
fi
if [[ -n "${STATE_PATH}" && ! -f "${STATE_PATH}" ]]; then
  echo "[ERROR] STATE_PATH not found: ${STATE_PATH}" >&2
  exit 1
fi

FINAL_SLOT_PATH="${SLOTPATH}"
FINAL_ACTION_PATH="${ACTION_PATH}"
FINAL_PROPRIO_PATH="${PROPRIO_PATH}"
FINAL_STATE_PATH="${STATE_PATH}"

if [[ "${AUTO_ALIGN_PKLS}" == "true" ]]; then
  mkdir -p "${ALIGNED_DIR}"

  export IN_SLOT="${SLOTPATH}"
  export IN_ACTION="${ACTION_PATH}"
  export IN_PROP="${PROPRIO_PATH}"
  export IN_STATE="${STATE_PATH}"
  export OUT_SLOT="${ALIGNED_DIR}/slots_aligned.pkl"
  export OUT_ACTION="${ALIGNED_DIR}/actions_aligned.pkl"
  export OUT_PROP="${ALIGNED_DIR}/proprio_aligned.pkl"
  export OUT_STATE="${ALIGNED_DIR}/state_aligned.pkl"

  python3 - <<'PY'
import os
import pickle as pkl
import numpy as np


def load(path):
    with open(path, "rb") as f:
        return pkl.load(f)


def save(obj, path):
    with open(path, "wb") as f:
        pkl.dump(obj, f)


slot = load(os.environ["IN_SLOT"])
action = load(os.environ["IN_ACTION"])
prop = load(os.environ["IN_PROP"])
state_path = os.environ.get("IN_STATE", "")
state = load(state_path) if state_path else None

for name, data in [("slot", slot), ("action", action), ("proprio", prop)]:
    if not isinstance(data, dict) or "train" not in data or "val" not in data:
        raise ValueError(f"{name} pkl must be dict with train/val keys")
if state is not None and (not isinstance(state, dict) or "train" not in state or "val" not in state):
    raise ValueError("state pkl must be dict with train/val keys")

slot_out = {"train": {}, "val": {}}
action_out = {"train": {}, "val": {}}
prop_out = {"train": {}, "val": {}}
state_out = {"train": {}, "val": {}} if state is not None else None

for split in ["train", "val"]:
    slot_keys = set(slot[split].keys())
    action_keys = set(action[split].keys())
    prop_keys = set(prop[split].keys())
    common = slot_keys & action_keys & prop_keys
    if state is not None:
        common &= set(state[split].keys())

    if not common:
        raise ValueError(f"No common keys in split={split}")

    dropped = len(slot_keys | action_keys | prop_keys) - len(common)
    print(f"[{split}] common episodes: {len(common)}, dropped(non-common): {dropped}")

    for k in sorted(common):
        s = np.asarray(slot[split][k])
        a = np.asarray(action[split][k])
        p = np.asarray(prop[split][k])
        x = np.asarray(state[split][k]) if state is not None else None

        if s.ndim != 3:
            raise ValueError(f"slot[{split}][{k}] must be [T,S,D], got {s.shape}")
        if a.ndim != 2:
            raise ValueError(f"action[{split}][{k}] must be [T,A], got {a.shape}")
        if p.ndim != 2:
            raise ValueError(f"proprio[{split}][{k}] must be [T,P], got {p.shape}")
        if x is not None and x.ndim != 2:
            raise ValueError(f"state[{split}][{k}] must be [T,X], got {x.shape}")

        T = min(s.shape[0], a.shape[0], p.shape[0], x.shape[0] if x is not None else 10**9)
        if T < 2:
            continue

        slot_out[split][k] = s[:T].astype(np.float32)
        action_out[split][k] = a[:T].astype(np.float32)
        prop_out[split][k] = p[:T].astype(np.float32)
        if x is not None:
            state_out[split][k] = x[:T].astype(np.float32)

for split in ["train", "val"]:
    if not slot_out[split]:
        raise ValueError(f"No valid samples after alignment in split={split}")

save(slot_out, os.environ["OUT_SLOT"])
save(action_out, os.environ["OUT_ACTION"])
save(prop_out, os.environ["OUT_PROP"])
if state_out is not None:
    save(state_out, os.environ["OUT_STATE"])

print("Saved aligned pkls:")
print("  slots  ->", os.environ["OUT_SLOT"])
print("  action ->", os.environ["OUT_ACTION"])
print("  proprio->", os.environ["OUT_PROP"])
if state_out is not None:
    print("  state  ->", os.environ["OUT_STATE"])
PY

  FINAL_SLOT_PATH="${OUT_SLOT}"
  FINAL_ACTION_PATH="${OUT_ACTION}"
  FINAL_PROPRIO_PATH="${OUT_PROP}"
  if [[ -n "${STATE_PATH}" ]]; then
    FINAL_STATE_PATH="${OUT_STATE}"
  fi
fi

# =========================
# Shape/key sanity check
# =========================
export V_SLOT="${FINAL_SLOT_PATH}"
export V_ACTION="${FINAL_ACTION_PATH}"
export V_PROP="${FINAL_PROPRIO_PATH}"
export V_STATE="${FINAL_STATE_PATH}"
python3 - <<'PY'
import os
import pickle as pkl
import numpy as np

def load(path):
    with open(path, "rb") as f:
        return pkl.load(f)

slot = load(os.environ["V_SLOT"])
action = load(os.environ["V_ACTION"])
prop = load(os.environ["V_PROP"])
state = load(os.environ["V_STATE"]) if os.environ.get("V_STATE") else None

for split in ["train", "val"]:
    ks = set(slot[split].keys())
    ka = set(action[split].keys())
    kp = set(prop[split].keys())
    common = ks & ka & kp
    if state is not None:
        common &= set(state[split].keys())
    if not common:
        raise RuntimeError(f"No common keys in split={split}")

    one = next(iter(common))
    s = np.asarray(slot[split][one])
    a = np.asarray(action[split][one])
    p = np.asarray(prop[split][one])
    print(f"[{split}] episodes={len(common)}")
    print(f"  sample_key={one}")
    print(f"  slot.shape={s.shape} action.shape={a.shape} proprio.shape={p.shape}")
PY

# =========================
# Launch training
# =========================
CMD=(
  python3 src/train/train_causalwm_from_pusht_slot.py
  cache_dir="~/.stable_worldmodel"
  output_model_name="${OUTPUT_MODEL_NAME}"
  dataset_name="metaworld_slot"
  num_workers="${NUM_WORKERS}"
  batch_size="${BATCH_SIZE}"
  trainer.max_epochs="${MAX_EPOCHS}"
  num_masked_slots="${NUM_MASKED_SLOTS}"
  predictor_lr="${PREDICTOR_LR}"
  proprio_encoder_lr="${PROPRIO_ENCODER_LR}"
  action_encoder_lr="${ACTION_ENCODER_LR}"
  dinowm.history_size="${HISTORY_SIZE}"
  dinowm.num_preds="${NUM_PREDS}"
  dinowm.proprio_dim="${PROPRIO_DIM}"
  dinowm.action_dim="${ACTION_DIM}"
  dinowm.proprio_embed_dim="${PROPRIO_EMBED_DIM}"
  dinowm.action_embed_dim="${ACTION_EMBED_DIM}"
  frameskip="${FRAMESKIP}"
  videosaur.NUM_SLOTS="${NUM_SLOTS}"
  videosaur.SLOT_DIM="${SLOT_DIM}"
  predictor.heads=16
  embedding_dir="${FINAL_SLOT_PATH}"
  action_dir="${FINAL_ACTION_PATH}"
  proprio_dir="${FINAL_PROPRIO_PATH}"
  use_hungarian_matching="${USE_HUNGARIAN_MATCHING}"
  hungarian_cost_type="${HUNGARIAN_COST_TYPE}"
  model.load_weights="${OC_CKPT_PATH}"
)

if [[ -n "${FINAL_STATE_PATH}" ]]; then
  CMD+=(state_dir="${FINAL_STATE_PATH}")
fi

echo "[INFO] Launch command:"
printf ' %q' "${CMD[@]}"
echo

if [[ "${DRY_RUN}" == "true" ]]; then
  echo "[INFO] DRY_RUN=true, skip training launch."
  exit 0
fi

"${CMD[@]}"
