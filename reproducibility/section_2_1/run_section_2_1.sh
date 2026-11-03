#!/usr/bin/env bash
set -euo pipefail
export SMOACS_DEVICE="${SMOACS_DEVICE:-cuda}"

section_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${SMOACS_ROOT:-$(cd -- "$section_dir/../.." && pwd)}"
cd -- "$repo_root"
test -f main.py || { echo 'Place this package at the public repository root; main.py is missing.' >&2; exit 1; }

num_candidate=128
b_size=128
num_steps=200
mutation_steps=(100 150)
seeds=(42 43 44 45)
exp_suffix=""
if [[ "${SMOACS_SMOKE_TEST:-0}" == 1 ]]; then
    num_candidate=2
    b_size=2
    num_steps=10
    mutation_steps=(5 7)
    seeds=(42)
    exp_suffix="__smoke"
fi

for initialization in random-random random-AtomicMask Struct-random Struct-AtomicMask; do
    for method in grad grad_sa mutation mutation_sa; do
        for seed in "${seeds[@]}"; do
            "${PYTHON:-python}" "$section_dir/section_2_1_main.py" \
                --initialization "$initialization" \
                --method "$method" \
                --num_candidate "$num_candidate" \
                --b_size "$b_size" \
                --num_steps "$num_steps" \
                --mutation_steps "${mutation_steps[@]}" \
                --seed "$seed" \
                --exp_name "section_2_1_${initialization}_${method}__seed${seed}${exp_suffix}" \
                --output_root "$repo_root/results"
        done
    done
done

if [[ "${SMOACS_SMOKE_TEST:-0}" != 1 ]]; then
    "${PYTHON:-python}" "$section_dir/summarize_runs.py" \
        --output-root "$repo_root/results" --seeds "${seeds[@]}"
fi
