#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${SMOACS_PYTHON:-python}"
DEVICE="${SMOACS_DEVICE:-cuda}"
NUM_STEPS="${SMOACS_NUM_STEPS:-2}"
GRAD_STEPS="${SMOACS_GRAD_STEPS:-1}"
BATCH_SIZE="${SMOACS_BATCH_SIZE:-4}"
NUM_CANDIDATE="${SMOACS_NUM_CANDIDATE:-4}"
NUM_WORKERS="${SMOACS_NUM_WORKERS:-0}"
SEED="${SMOACS_SEED:-0}"
EXP_NAME="${SMOACS_EXP_NAME:-reproduce_C_half_metal_search}"
INITIAL_STRUCTURES_PATH="${HALF_METAL_INITIAL_STRUCTURES_PATH:-${REPO_ROOT}/data/raw_data/initial_mp_nsites2_30__mask_dict_super_common.pkl}"
MASK_DICT_PATH="${HALF_METAL_MASK_DICT_PATH:-${REPO_ROOT}/data/intermidiate/mask_dict_super_common.npz}"
CONFIG_PATH="${HALF_METAL_CONFIG:-${REPO_ROOT}/configs/half_metal_level5_forbidden.yaml}"
ALIGNN_WEIGHTS_DIR="${HALF_METAL_ALIGNN_WEIGHTS_DIR:-${ALIGNN_WEIGHTS_DIR:-${REPO_ROOT}/models/alignn_pretrained}}"
MAGMOM_CHECKPOINT="${MAGMOM_CHECKPOINT:-jv_magmom_alignn}"
MATTERSIM_MODEL="${SMOACS_MATTERSIM_MODEL:-${REPO_ROOT}/differentiable_MatterSim/pretrained_models/mattersim-v1.0.0-5M.pth}"
ENERGY_COEF="${SMOACS_ENERGY_COEF:-1.0}"
ENERGY_EVERY="${SMOACS_ENERGY_EVERY:-1}"
W_MAG_INT="${SMOACS_W_MAG_INT:-1}"
ELEMENT_COUNT_L1_WEIGHT="${SMOACS_ELEMENT_COUNT_L1_WEIGHT:-1000}"
SA_T0="${SMOACS_SA_T0:-0.18}"
SA_TMIN="${SMOACS_SA_TMIN:-1.2e-3}"
SA_COORDS_SIGMA="${SMOACS_SA_COORDS_SIGMA:-8.8e-3}"
SA_LATTICE_SIGMA="${SMOACS_SA_LATTICE_SIGMA:-1.0e-3}"
SA_ANGLE_SIGMA="${SMOACS_SA_ANGLE_SIGMA:-4.3e-2}"
LATTICE_LR="${SMOACS_LATTICE_LR:-1.1e-6}"
COORDS_LR="${SMOACS_COORDS_LR:-2.4e-6}"
ATOM_LR="${SMOACS_ATOM_LR:-4.1e-4}"
COPY_MUTATION_OVERRIDE="${SMOACS_COPY_MUTATION_OVERRIDE:-copy_mutation___steps_cm-162-324-486__mutation_noise-0.000000012__top_ratio-0.20__group_C_use_rate-0.020__atom_dist_init-True}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp/smoacs-cache}"
export SMOACS_DATALOADER_WORKERS="${SMOACS_DATALOADER_WORKERS:-${NUM_WORKERS}}"
export FORCE_DATALOADER_SINGLE_PROCESS="${FORCE_DATALOADER_SINGLE_PROCESS:-1}"
export MATTERSIM_LOGURU_ENQUEUE="${MATTERSIM_LOGURU_ENQUEUE:-0}"

if [[ ! -e "${INITIAL_STRUCTURES_PATH}" ]]; then
  echo "Missing half-metal initial structures: ${INITIAL_STRUCTURES_PATH}" >&2
  echo "Generate the base-mask pkl or set HALF_METAL_INITIAL_STRUCTURES_PATH to a supported .pkl or raw directory." >&2
  exit 2
fi

if [[ ! -f "${MASK_DICT_PATH}" ]]; then
  echo "Missing half-metal base mask: ${MASK_DICT_PATH}" >&2
  echo "Set HALF_METAL_MASK_DICT_PATH to a mask_dict_super_common-compatible .npz file." >&2
  exit 2
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Missing half-metal config: ${CONFIG_PATH}" >&2
  echo "Set HALF_METAL_CONFIG to a config with element_constraints.forbidden_elements." >&2
  exit 2
fi

if [[ ! -f "${ALIGNN_WEIGHTS_DIR}/mp_gappbe_alignnn.zip" || ! -f "${ALIGNN_WEIGHTS_DIR}/mp_e_form_alignnn.zip" ]]; then
  echo "Missing ALIGNN band-gap/e-form weights in: ${ALIGNN_WEIGHTS_DIR}" >&2
  echo "Run scripts/download_model_weights.py or set HALF_METAL_ALIGNN_WEIGHTS_DIR." >&2
  exit 2
fi

if [[ ! -f "${ALIGNN_WEIGHTS_DIR}/${MAGMOM_CHECKPOINT}.zip" ]]; then
  echo "Missing magmom checkpoint ${MAGMOM_CHECKPOINT}.zip in: ${ALIGNN_WEIGHTS_DIR}" >&2
  echo "Download it from manifests/model_weights_manifest.tsv or set HALF_METAL_ALIGNN_WEIGHTS_DIR." >&2
  exit 2
fi

if [[ "${ENERGY_COEF}" != "0" && "${ENERGY_COEF}" != "0.0" && ! -f "${MATTERSIM_MODEL}" ]]; then
  echo "Missing MatterSim model checkpoint: ${MATTERSIM_MODEL}" >&2
  echo "Set SMOACS_MATTERSIM_MODEL to a mattersim-v1.0.0-5M.pth checkpoint." >&2
  exit 2
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" main.py \
  --model_name ALIGNN \
  --task bg_and_eform \
  --space_group -1 \
  --bg_target 0.0 \
  --target_val 0.0 \
  --training_margin 0.0 \
  --eval_margin 0.01 \
  --e_form_coef 0.0 \
  --e_form_criteria 10000000 \
  --use_magmom_loss \
  --magmom_checkpoint "${MAGMOM_CHECKPOINT}" \
  --initial_structures_path "${INITIAL_STRUCTURES_PATH}" \
  --mask_dict_path "${MASK_DICT_PATH}" \
  --config "${CONFIG_PATH}" \
  --optimizer grad_sa_hybrid \
  --num_steps "${NUM_STEPS}" \
  --grad_sa_hybrid_grad_steps "${GRAD_STEPS}" \
  --sa_t0 "${SA_T0}" \
  --sa_tmin "${SA_TMIN}" \
  --sa_trials 1 \
  --sa_coords_sigma "${SA_COORDS_SIGMA}" \
  --sa_lattice_sigma "${SA_LATTICE_SIGMA}" \
  --sa_angle_sigma "${SA_ANGLE_SIGMA}" \
  --allow_lr_override \
  --lattice_lr "${LATTICE_LR}" \
  --coords_lr "${COORDS_LR}" \
  --atom_lr "${ATOM_LR}" \
  --copy_mutation_override "${COPY_MUTATION_OVERRIDE}" \
  --b_size "${BATCH_SIZE}" \
  --num_candidate "${NUM_CANDIDATE}" \
  --num_workers "${NUM_WORKERS}" \
  --device "${DEVICE}" \
  --abc_min 2 \
  --abc_max 20 \
  --energy_coef "${ENERGY_COEF}" \
  --energy_every "${ENERGY_EVERY}" \
  --mattersim_model "${MATTERSIM_MODEL}" \
  --use_element_count_l1_loss \
  --w_mag_int "${W_MAG_INT}" \
  --element_count_l1_weight "${ELEMENT_COUNT_L1_WEIGHT}" \
  --seed "${SEED}" \
  --alignn_weights_dir "${ALIGNN_WEIGHTS_DIR}" \
  --exp_name "${EXP_NAME}" \
  "$@"
