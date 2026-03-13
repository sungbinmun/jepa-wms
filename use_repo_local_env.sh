#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p \
  "${REPO_ROOT}/_local/datasets" \
  "${REPO_ROOT}/_local/logs" \
  "${REPO_ROOT}/_local/checkpoints" \
  "${REPO_ROOT}/_local/oss_ckpt"

export JEPAWM_HOME="${REPO_ROOT}"
export JEPAWM_DSET="${REPO_ROOT}/_local/datasets"
export JEPAWM_LOGS="${REPO_ROOT}/_local/logs"
export JEPAWM_CKPT="${REPO_ROOT}/_local/checkpoints"
export JEPAWM_OSSCKPT="${REPO_ROOT}/_local/oss_ckpt"

python "${REPO_ROOT}/setup_macros.py"

cat <<EOF
Repo-local JEPA-WM paths enabled.
JEPAWM_HOME=${JEPAWM_HOME}
JEPAWM_DSET=${JEPAWM_DSET}
JEPAWM_LOGS=${JEPAWM_LOGS}
JEPAWM_CKPT=${JEPAWM_CKPT}
JEPAWM_OSSCKPT=${JEPAWM_OSSCKPT}
EOF
