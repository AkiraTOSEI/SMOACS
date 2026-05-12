#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${SMOACS_PYTHON:-python}"
DEVICE="${SMOACS_DEVICE:-cuda}"
NUM_STEPS="${SMOACS_NUM_STEPS:-2}"
BATCH_SIZE="${SMOACS_BATCH_SIZE:-4}"
NUM_CANDIDATE="${SMOACS_NUM_CANDIDATE:-4}"
NUM_WORKERS="${SMOACS_NUM_WORKERS:-0}"
SEED="${SMOACS_SEED:-0}"
EXP_NAME="${SMOACS_EXP_NAME:-reproduce_A_perovskite_design}"
INITIAL_STRUCTURES_PATH="${PEROVSKITE_INITIAL_STRUCTURES_PATH:-${REPO_ROOT}/data/raw_data/initial_random_perovskite}"
ALIGNN_WEIGHTS_DIR="${ALIGNN_WEIGHTS_DIR:-${REPO_ROOT}/models/alignn_pretrained}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/smoacs-cache}"
export SMOACS_DATALOADER_WORKERS="${SMOACS_DATALOADER_WORKERS:-${NUM_WORKERS}}"

if [[ ! -d "${INITIAL_STRUCTURES_PATH}" ]]; then
  echo "Missing perovskite initial structures: ${INITIAL_STRUCTURES_PATH}" >&2
  echo "Generate repo-local templates with: ${PYTHON_BIN} create_dataset.py" >&2
  echo "Or set PEROVSKITE_INITIAL_STRUCTURES_PATH=/path/to/initial_random_perovskite" >&2
  exit 2
fi

if [[ ! -f "${ALIGNN_WEIGHTS_DIR}/mp_gappbe_alignnn.zip" || ! -f "${ALIGNN_WEIGHTS_DIR}/mp_e_form_alignnn.zip" ]]; then
  echo "Missing ALIGNN weights in: ${ALIGNN_WEIGHTS_DIR}" >&2
  echo "Run scripts/download_model_weights.py or set ALIGNN_WEIGHTS_DIR." >&2
  exit 2
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" main.py \
  --model_name ALIGNN \
  --task bg_and_eform \
  --target_val 0.5 \
  --structure perovskite \
  --initial_structures_path "${INITIAL_STRUCTURES_PATH}" \
  --num_steps "${NUM_STEPS}" \
  --b_size "${BATCH_SIZE}" \
  --num_candidate "${NUM_CANDIDATE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --seed "${SEED}" \
  --alignn_weights_dir "${ALIGNN_WEIGHTS_DIR}" \
  --exp_name "${EXP_NAME}" \
  "$@"
