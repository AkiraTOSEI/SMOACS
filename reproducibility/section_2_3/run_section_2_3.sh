#!/usr/bin/env bash
set -euo pipefail
export SMOACS_DEVICE="${SMOACS_DEVICE:-cuda}"

# 保存された実行引数に基づく5条件。
section_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$section_dir/../.." && pwd)"
cd -- "$repo_root"
test -f main.py || { echo '公開リポジトリ直下にこの配布物を配置してください。main.py がありません。' >&2; exit 1; }

initial_structures=reproducibility/section_2_3/templates_masks/initial_mp_nsites2_12__mask_dict_super_common.pkl
mask_dict=reproducibility/section_2_3/templates_masks/masks/mask_dict_super_common.npz
for input_file in "$initial_structures" "$mask_dict"; do
    test -f "$input_file" || { echo "必要な入力がありません: $input_file" >&2; exit 1; }
done

num_candidate=512
b_size=128
num_steps=600
hybrid_grad_steps=200
mutation_steps=150-300-450
exp_suffix=""
if [[ "${SMOACS_SMOKE_TEST:-0}" == 1 ]]; then
    num_candidate=2
    b_size=2
    num_steps=10
    hybrid_grad_steps=5
    mutation_steps=2-5-7
    exp_suffix="__smoke"
fi

for target in 2.5 3.0 3.5 4.0 4.5; do
    exp_name="section_2_3_bg${target/./p}${exp_suffix}"
    if [ -e "results/$exp_name" ]; then
        echo "出力先が既に存在します: results/$exp_name" >&2
        exit 1
    fi
    "${PYTHON:-python}" main.py \
        --model_name ALIGNN \
        --task bg_and_eform \
        --target_val "$target" \
        --training_margin 0.04 \
        --eval_margin 2.0 \
        --e_form_coef 1.0 \
        --e_form_criteria 1.0 \
        --abc_min 2.0 \
        --abc_max 10.0 \
        --space_group -1 \
        --num_candidate "$num_candidate" \
        --b_size "$b_size" \
        --num_workers 4 \
        --max_ox_patterns 100 \
        --optimizer grad_sa_hybrid \
        --num_steps "$num_steps" \
        --grad_sa_hybrid_grad_steps "$hybrid_grad_steps" \
        --sa_t0 1.3325910286336045 \
        --sa_tmin 6.436646738216772e-05 \
        --sa_trials 1 \
        --sa_coords_sigma 0.004025002212594491 \
        --sa_lattice_sigma 0.000454699806947192 \
        --sa_angle_sigma 0.0024897646570984427 \
        --sa_cache_size 0 \
        --sa_per_sample_accept \
        --allow_lr_override \
        --lattice_lr 2.3041950292841973e-06 \
        --coords_lr 0.009651840930986113 \
        --atom_lr 0.0026276247474973377 \
        --copy_mutation_override "copy_mutation___steps_cm-${mutation_steps}__mutation_noise-0.0005281233296703__top_ratio-0.2__group_C_use_rate-0.02__atom_dist_init-True" \
        --use_element_count_l1_loss \
        --element_count_l1_weight 1.0 \
        --energy_coef 0.0 \
        --initial_structures_path "$initial_structures" \
        --mask_dict_path "$mask_dict" \
        --alignn_weights_dir models/alignn_pretrained \
        --exp_name "$exp_name" \
        --device "$SMOACS_DEVICE" \
        --seed 42
done
