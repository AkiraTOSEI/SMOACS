#!/usr/bin/env bash
set -euo pipefail
export SMOACS_DEVICE="${SMOACS_DEVICE:-cuda}"

# 原稿の条件を学習率の明示指定で実行。
section_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$section_dir/../.." && pwd)"
cd -- "$repo_root"
test -f main.py || { echo '公開リポジトリ直下にこの配布物を配置してください。main.py がありません。' >&2; exit 1; }
test -f src/utils/structure_paths.py || { echo 'template_order.txt 対応コードを src/ に反映してください。' >&2; exit 1; }

# 各ディレクトリには使用する 128 構造だけを置く（256 構造の pool は代用しない）。
shopt -s nullglob
for input_dir in reproducibility/section_2_2/templates_masks/historical_unit_cell reproducibility/section_2_2/templates_masks/historical_3x3x3; do
    test -f "$input_dir/template_order.txt" || { echo "読込順の記録が必要です: $input_dir/template_order.txt" >&2; exit 1; }
    input_files=("$input_dir"/*.vasp)
    if [ "${#input_files[@]}" -ne 128 ]; then
        echo "使用する 128 構造が必要です: $input_dir（現在 ${#input_files[@]} 件）" >&2
        exit 1
    fi
done

# perovskite は --mask_dict_path 非対応のため、コード所定の配置先との一致を確認。
for mask_file in ionic_mask_dict_super_common.npz ionic_radii_dict_super_common.npz; do
    if ! cmp -s "reproducibility/section_2_2/templates_masks/masks/$mask_file" "data/intermidiate/$mask_file"; then
        echo "マスクの所定位置への配置または内容が不一致です: data/intermidiate/$mask_file" >&2
        exit 1
    fi
done

num_candidate=128
b_size=128
num_steps=200
mutation_steps=50-100-150
cf_unit_mutation_steps=100-150
exp_suffix=""
if [[ "${SMOACS_SMOKE_TEST:-0}" == 1 ]]; then
    num_candidate=2
    b_size=2
    num_steps=10
    mutation_steps=2-5-7
    cf_unit_mutation_steps=5-7
    exp_suffix="__smoke"
fi

# ALIGNN: unit_cell
for target in 0.5 4.0; do
    exp_name="section_2_2_ALIGNN_unit_cell_bg${target/./p}${exp_suffix}"
    if [ -e "results/$exp_name" ]; then
        echo "出力先が既に存在します: results/$exp_name" >&2
        exit 1
    fi
    "${PYTHON:-python}" main.py \
        --model_name ALIGNN \
        --task bg_and_eform \
        --structure perovskite \
        --space_group -1 \
        --target_val "$target" \
        --training_margin 0.04 \
        --eval_margin 2.0 \
        --e_form_coef 1.0 \
        --e_form_criteria 1.0 \
        --abc_min 2.0 \
        --abc_max 10.0 \
        --num_candidate "$num_candidate" \
        --b_size "$b_size" \
        --num_workers 4 \
        --max_ox_patterns 100 \
        --optimizer gradient \
        --num_steps "$num_steps" \
        --allow_lr_override \
        --lattice_lr 9.0 \
        --coords_lr 0.002 \
        --atom_lr 5e-05 \
        --copy_mutation_override "copy_mutation___steps_cm-${mutation_steps}__mutation_noise-0.00001__top_ratio-0.87__group_C_use_rate-0.59__atom_dist_init-True" \
        --energy_coef 0.0 \
        --initial_structures_path reproducibility/section_2_2/templates_masks/historical_unit_cell \
        --alignn_weights_dir models/alignn_pretrained \
        --exp_name "$exp_name" \
        --device "$SMOACS_DEVICE" \
        --seed 42
done

# Crystalformer: unit_cell
for target in 0.5 4.0; do
    exp_name="section_2_2_Crystalformer_unit_cell_bg${target/./p}${exp_suffix}"
    if [ -e "results/$exp_name" ]; then
        echo "出力先が既に存在します: results/$exp_name" >&2
        exit 1
    fi
    "${PYTHON:-python}" main.py \
        --model_name Crystalformer \
        --task bg_and_eform \
        --structure perovskite \
        --space_group -1 \
        --target_val "$target" \
        --training_margin 0.04 \
        --eval_margin 2.0 \
        --e_form_coef 1.0 \
        --e_form_criteria 1.0 \
        --abc_min 2.0 \
        --abc_max 10.0 \
        --num_candidate "$num_candidate" \
        --b_size "$b_size" \
        --num_workers 4 \
        --max_ox_patterns 100 \
        --optimizer gradient \
        --num_steps "$num_steps" \
        --allow_lr_override \
        --lattice_lr 0.002 \
        --coords_lr 0.005 \
        --atom_lr 0.007 \
        --copy_mutation_override "copy_mutation___steps_cm-${cf_unit_mutation_steps}__mutation_noise-0.00007__top_ratio-0.1__group_C_use_rate-0.05__atom_dist_init-True" \
        --energy_coef 0.0 \
        --initial_structures_path reproducibility/section_2_2/templates_masks/historical_unit_cell \
        --alignn_weights_dir models/alignn_pretrained \
        --exp_name "$exp_name" \
        --device "$SMOACS_DEVICE" \
        --seed 42
done

# ALIGNN: 3x3x3
for target in 0.5 4.0; do
    exp_name="section_2_2_ALIGNN_3x3x3_bg${target/./p}${exp_suffix}"
    if [ -e "results/$exp_name" ]; then
        echo "出力先が既に存在します: results/$exp_name" >&2
        exit 1
    fi
    "${PYTHON:-python}" main.py \
        --model_name ALIGNN \
        --task bg_and_eform \
        --structure perovskite3x3x3 \
        --space_group -1 \
        --target_val "$target" \
        --training_margin 0.04 \
        --eval_margin 2.0 \
        --e_form_coef 1.0 \
        --e_form_criteria 1.0 \
        --abc_min 2.0 \
        --abc_max 10.0 \
        --num_candidate "$num_candidate" \
        --b_size "$b_size" \
        --num_workers 4 \
        --max_ox_patterns 100 \
        --optimizer gradient \
        --num_steps "$num_steps" \
        --allow_lr_override \
        --lattice_lr 0.01 \
        --coords_lr 0.000667 \
        --atom_lr 0.00004 \
        --copy_mutation_override "copy_mutation___steps_cm-${mutation_steps}__mutation_noise-0.000003__top_ratio-0.3__group_C_use_rate-0.2__atom_dist_init-True" \
        --energy_coef 0.0 \
        --initial_structures_path reproducibility/section_2_2/templates_masks/historical_3x3x3 \
        --alignn_weights_dir models/alignn_pretrained \
        --exp_name "$exp_name" \
        --device "$SMOACS_DEVICE" \
        --seed 42
done

# Crystalformer: 3x3x3
for target in 0.5 4.0; do
    exp_name="section_2_2_Crystalformer_3x3x3_bg${target/./p}${exp_suffix}"
    if [ -e "results/$exp_name" ]; then
        echo "出力先が既に存在します: results/$exp_name" >&2
        exit 1
    fi
    "${PYTHON:-python}" main.py \
        --model_name Crystalformer \
        --task bg_and_eform \
        --structure perovskite3x3x3 \
        --space_group -1 \
        --target_val "$target" \
        --training_margin 0.04 \
        --eval_margin 2.0 \
        --e_form_coef 1.0 \
        --e_form_criteria 1.0 \
        --abc_min 2.0 \
        --abc_max 10.0 \
        --num_candidate "$num_candidate" \
        --b_size "$b_size" \
        --num_workers 4 \
        --max_ox_patterns 100 \
        --optimizer gradient \
        --num_steps "$num_steps" \
        --allow_lr_override \
        --lattice_lr 0.6 \
        --coords_lr 0.00067 \
        --atom_lr 0.002 \
        --copy_mutation_override "copy_mutation___steps_cm-${mutation_steps}__mutation_noise-0.00001__top_ratio-0.3__group_C_use_rate-0.2__atom_dist_init-True" \
        --energy_coef 0.0 \
        --initial_structures_path reproducibility/section_2_2/templates_masks/historical_3x3x3 \
        --alignn_weights_dir models/alignn_pretrained \
        --exp_name "$exp_name" \
        --device "$SMOACS_DEVICE" \
        --seed 42
done
