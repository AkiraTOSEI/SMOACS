#!/usr/bin/env bash
set -euo pipefail
export SMOACS_DEVICE="${SMOACS_DEVICE:-cuda}"

# 全 10,050 レコードを収録した PKL の先頭 8,192 件を使用します。
section_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$section_dir/../.." && pwd)"
cd -- "$repo_root"
test -f main.py || { echo '公開リポジトリ直下にこの配布物を配置してください。main.py がありません。' >&2; exit 1; }

initial_structures=reproducibility/section_2_4/templates_masks/initial_mp_nsites2_30__mask_dict_super_common_level5.pkl
mattersim_model="${SMOACS_MATTERSIM_MODEL:-${HOME}/.local/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth}"
test -f "$initial_structures" || {
    echo "初期構造とマスクの PKL が必要です: $initial_structures" >&2
    exit 1
}
test -f "$mattersim_model" || {
    echo "MatterSim 5M checkpoint がありません: $mattersim_model" >&2
    echo '公開 README に従い標準ディレクトリに5Mモデルを取得してください。' >&2
    exit 1
}

export MATTERSIM_LOGURU_ENQUEUE=0
export SMOACS_DATALOADER_WORKERS=0
export FORCE_DATALOADER_SINGLE_PROCESS=1

num_candidate=8192
b_size=150
num_steps=650
hybrid_grad_steps=300
mutation_steps=162-324-486
exp_suffix=""
if [[ "${SMOACS_SMOKE_TEST:-0}" == 1 ]]; then
    num_candidate=2
    b_size=2
    num_steps=10
    hybrid_grad_steps=5
    mutation_steps=2-5-7
    exp_suffix="__smoke"
fi

for mag_weight in 1 10 100; do
    for element_weight in 100 1000; do
        exp_name="section_2_4_best_trial00010__elm-w${element_weight}__level5__w-mag-int${mag_weight}${exp_suffix}"
        if [ -e "results/$exp_name" ]; then
            echo "出力先が既に存在します: results/$exp_name" >&2
            exit 1
        fi
        "${PYTHON:-python}" main.py \
            --model_name ALIGNN \
            --task bg_and_eform \
            --space_group -1 \
            --target_val 0 \
            --bg_target 0 \
            --training_margin 0 \
            --eval_margin 0.01 \
            --e_form_coef 0 \
            --e_form_criteria 10000000 \
            --abc_min 2.0 \
            --abc_max 10.0 \
            --num_candidate "$num_candidate" \
            --b_size "$b_size" \
            --num_workers 0 \
            --max_ox_patterns 100 \
            --optimizer grad_sa_hybrid \
            --num_steps "$num_steps" \
            --grad_sa_hybrid_grad_steps "$hybrid_grad_steps" \
            --sa_t0 0.1789656546773165 \
            --sa_tmin 0.0011807247257803618 \
            --sa_trials 1 \
            --sa_coords_sigma 0.0087835578650321 \
            --sa_lattice_sigma 0.00030955323601318513 \
            --sa_angle_sigma 0.04272274816226785 \
            --sa_cache_size 0 \
            --sa_per_sample_accept \
            --allow_lr_override \
            --lattice_lr 1.1053984112218183e-06 \
            --coords_lr 2.44272804733531e-06 \
            --atom_lr 0.0004144823972110048 \
            --copy_mutation_override "copy_mutation___steps_cm-${mutation_steps}__mutation_noise-0.0000000118540572__top_ratio-0.2__group_C_use_rate-0.02__atom_dist_init-True" \
            --energy_coef 0.6954771451368791 \
            --energy_every 1 \
            --mattersim_model "$mattersim_model" \
            --use_magmom_loss \
            --magmom_checkpoint jv_magmom_alignn \
            --magmom_tol 0.05 \
            --magmom_eps 0.01 \
            --w_bg 1.0 \
            --w_mag 1.0 \
            --w_mag_int "$mag_weight" \
            --w_mag_sign 1.0 \
            --use_element_count_l1_loss \
            --element_count_l1_weight "$element_weight" \
            --initial_structures_path "$initial_structures" \
            --alignn_weights_dir models/alignn_pretrained \
            --exp_name "$exp_name" \
            --device "$SMOACS_DEVICE" \
            --seed 42
    done
done
