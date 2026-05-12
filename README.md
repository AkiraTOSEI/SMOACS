# Rethinking Gradient-Based Methods: Multi-Property Materials Design Beyond Differentiable Targets

**Simultaneous Multi-property Optimization using Adaptive Crystal Synthesizer (SMOACS)** is a framework that utilizes state-of-the-art property prediction models and their gradients to directly optimize input crystal structures for multiple targeted properties simultaneously. It enables the integration of adaptive constraints into the optimization process without necessitating model retraining. This allows for optimizing targeted properties while maintaining specific crystal structures, such as perovskites, even with models trained on diverse crystal types.

![figure1](images/figure1.png)

![optimization1](images/crystal42_time_series.gif)



## Paper
For detailed information, please refer to our paper:

[Rethinking Gradient-Based Methods: Multi-Property Materials Design Beyond Differentiable Targets](https://arxiv.org/abs/2410.08562)

### Citation

If you use this work in your research, please cite:

```text
@article{afujii2024,
  title={Adaptive Constraint Integration for Simultaneously Optimizing Crystal Structures with Multiple Targeted Properties},
  author={Akihiro FUJII, Yoshitaka Ushiku, Koji Shimizu, Anh Khoa Augustin Lu, Satoshi Watanabe },
  journal={arXiv preprint arXiv:2410.08562},
  year={2024}
}
```

The code in this repository uses code derived from Crystalformer and a minimal
vendored runtime subset from ALIGNN. Crystalformer's pretrained models are
provided in [CrystalFormer's official repository](https://github.com/deepmodeling/CrystalFormer),
and ALIGNN's pretrained models are provided in
[ALIGNN's official repository](https://github.com/usnistgov/alignn). Thus, if
you use our code in your projects, please cite both our work and the
Crystalformer paper or ALIGNN:

### Crystalformer
```text
@inproceedings{taniai2024crystalformer,
  title     = {Crystalformer: Infinitely Connected Attention for Periodic Structure Encoding},
  author    = {Tatsunori Taniai and
               Ryo Igarashi and
               Yuta Suzuki and
               Naoya Chiba and
               Kotaro Saito and
               Yoshitaka Ushiku and
               Kanta Ono
               },
  booktitle = {The Twelfth International Conference on Learning Representations},
  year      = {2024},
  url       = {https://openreview.net/forum?id=fxQiecl9HB}
}
```

### ALIGNN
```text
@article{choudhary2021atomistic,
  title={Atomistic line graph neural network for improved materials property predictions},
  author={Choudhary, Kamal and DeCost, Brian},
  journal={npj Computational Materials},
  volume={7},
  number={1},
  pages={185},
  year={2021},
  publisher={Nature Publishing Group UK London}
}
```

### MatterSim
```text
@article{yang2024mattersim,
      title={MatterSim: A Deep Learning Atomistic Model Across Elements, Temperatures and Pressures},
      author={Han Yang and Chenxi Hu and Yichi Zhou and Xixian Liu and Yu Shi and Jielan Li and Guanzhi Li and Zekun Chen and Shuizhou Chen and Claudio Zeni and Matthew Horton and Robert Pinsler and Andrew Fowler and Daniel Zügner and Tian Xie and Jake Smith and Lixin Sun and Qian Wang and Lingyu Kong and Chang Liu and Hongxia Hao and Ziheng Lu},
      year={2024},
      eprint={2405.04967},
      archivePrefix={arXiv},
      primaryClass={cond-mat.mtrl-sci},
      url={https://arxiv.org/abs/2405.04967},
      journal={arXiv preprint arXiv:2405.04967}
}
```


### Third-party code and licenses

This repository includes a minimal ALIGNN runtime subset under
`third_party/alignn_minimal/`, vendored from ALIGNN.
This is done so SMOACS can be tested in modern environments, including
NumPy 2.x environments, without being blocked by the external `alignn` package
dependency metadata. The external `alignn` Python package is not required at
runtime for SMOACS. See `third_party/alignn_minimal/LICENSE.rst` and
`THIRD_PARTY_NOTICES.md` for license, provenance, vendored-file scope, and local
change details. SMOACS original code is MIT licensed; third-party files are
governed by their respective licenses or terms. Pretrained weights are not
vendored.


## Requirements

SMOACS currently documents two environment profiles.

- **Paper environment**: use this for paper reproduction.
- **Modern environment**: use this for newer Python and NumPy 2.x execution.

The external `alignn` Python package is not required in either profile. SMOACS
uses the vendored minimal ALIGNN runtime subset under
`third_party/alignn_minimal/`.

`pyproject.toml` is the modern install metadata. The legacy dependency snapshot
is kept in `pyproject.legacy.toml` for reference only; `pip` and Poetry do not
consume it automatically. For the paper environment, install the pinned
dependencies below explicitly and then install this repository with `--no-deps`.

CUDA-specific PyTorch, DGL, and PyG packages are installed outside Poetry.
Use wheels that match your CUDA runtime. The commands below target CUDA 12.4.

### Paper environment

Verified profile:

- Python 3.10.14
- NumPy 1.26.4
- PyTorch 2.4.0+cu124
- DGL 2.4.0+cu124
- PyTorch Geometric 2.7.0
- pymatgen 2025.6.14
- ASE 3.27.0
- setuptools 80.10.2
- MatterSim 0.1.dev5, only for MatterSim energy and hull workflows

Create a Python 3.10 virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip "setuptools<81" wheel
```

Install PyTorch, DGL, and PyG for CUDA 12.4:

```bash
python -m pip install \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install \
  "dgl==2.4.0+cu124" \
  -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

python -m pip install \
  torch-geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

Install SMOACS runtime dependencies:

```bash
python -m pip install \
  "numpy==1.26.4" \
  "pymatgen==2025.6.14" \
  scipy \
  pandas \
  matplotlib \
  tqdm \
  requests \
  "ase==3.27.0" \
  jarvis-tools \
  periodictable \
  "smact==2.5.5" \
  "ipython==8.10.0" \
  pytest \
  optimade \
  optuna \
  pydantic \
  pydantic-settings \
  scikit-learn \
  lmdb \
  pyyaml \
  mp-api
```

Install this repository without letting the modern `pyproject.toml` alter the
paper environment dependencies:

```bash
python -m pip install -e . --no-deps
```

Because this install deliberately keeps the paper dependency stack while
`pyproject.toml` describes the modern stack, `pip check` reports expected
metadata conflicts in this profile. Use import checks and the smoke commands
below as the validation criteria for the paper environment.

### Modern environment

Verified profile:

- Python 3.12.13
- NumPy 2.4.3
- PyTorch 2.4.0+cu124
- DGL 2.4.0+cu124
- PyTorch Geometric 2.7.0
- pymatgen 2026.5.4
- scipy 1.17.1
- pandas 3.0.2
- ASE 3.27.0
- setuptools 80.10.2
- external `alignn` package: not installed
- differentiable MatterSim: installed from
  <https://github.com/AkiraTOSEI/differentiable-mattersim>

Create a conda environment:

```bash
conda create -n smoacs-modern -y python=3.12.13 pip
conda activate smoacs-modern
python -m pip install -U pip "setuptools<81" wheel
```

Install PyTorch, DGL, and PyG for CUDA 12.4:

```bash
python -m pip install \
  torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu124

python -m pip install \
  "dgl==2.4.0+cu124" \
  -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html

python -m pip install \
  torch-geometric pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
  -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

Install SMOACS and the pure-Python runtime dependencies. In the modern profile,
the normal install also installs the differentiable MatterSim fork declared in
`pyproject.toml`. It does not download MatterSim checkpoints.

```bash
python -m pip install -e .[dev]
```

Check the modern environment:

```bash
PYTHONPATH=. python - <<'PY'
import importlib.metadata as md
import sys

import dgl
import torch
from third_party.alignn_minimal.models.alignn import ALIGNNConfig

print("python:", sys.version.split()[0])
for pkg in ["numpy", "torch", "dgl", "torch-geometric", "pymatgen", "mattersim", "alignn"]:
    try:
        print(f"{pkg}:", md.version(pkg))
    except md.PackageNotFoundError:
        print(f"{pkg}: NOT INSTALLED")
print("torch CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("vendored ALIGNN config:", ALIGNNConfig.__name__)
if torch.cuda.is_available():
    g = dgl.graph(([0, 1], [1, 2])).to("cuda")
    print("DGL graph device:", g.device)
PY
```

For MatterSim energy loss, install the differentiable MatterSim implementation.
The standard PyPI `mattersim` package does not provide
`mattersim.forcefield.differentiable_potential`, which SMOACS uses for
backpropagation through the energy term. The differentiable implementation is a
fork of the MatterSim repository with differentiable energy evaluation added:
<https://github.com/AkiraTOSEI/differentiable-mattersim>.

In the modern environment, `python -m pip install -e .[dev]` installs this fork.
In the paper environment, install the runtime pieces needed by MatterSim energy
evaluation explicitly, then install the fork without dependencies so the pinned
NumPy is not upgraded. Keep ASE at 3.27.0; ASE 3.28.0 removed the
`full_3x3_to_voigt_6_stress` import path used by this fork.

```bash
git clone https://github.com/AkiraTOSEI/differentiable-mattersim differentiable_mattersim
python -m pip install \
  deprecated \
  "e3nn>=0.5.0" \
  loguru \
  opt_einsum_fx \
  seekpath \
  phonopy \
  "torch-ema>=0.3" \
  "torch_runstats>=0.2.0" \
  "torchmetrics>=0.10.0"
python -m pip install -e ./differentiable_mattersim --no-deps
```

Download the 5M checkpoint and export the path before running half-metal
searches or MatterSim autograd checks:

```bash
python -c "from mattersim.utils.download_utils import download_checkpoint; download_checkpoint('mattersim-v1.0.0-5M.pth')"

export SMOACS_MATTERSIM_MODEL="$HOME/.local/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth"

python main.py ... \
  --energy_coef 1.0 \
  --energy_every 1 \
  --mattersim_model "$SMOACS_MATTERSIM_MODEL"
```

The SMOACS weight downloader does not download MatterSim checkpoints because
they are external MatterSim artifacts. If the checkpoint is already available in
the cloned checkout, point the environment variable to that file instead:

```bash
export SMOACS_MATTERSIM_MODEL="$PWD/differentiable_mattersim/pretrained_models/mattersim-v1.0.0-5M.pth"
```

The modern environment was verified with NumPy 2.4.3 and no external `alignn`
package.

### Pretrained model weights
Model weights are not committed to this repository. Download the ALIGNN weights
used by the smoke examples below with:

```bash
python scripts/download_model_weights.py \
  --only mp_gappbe_alignnn mp_e_form_alignnn jv_magmom_alignn
```

The full manifest is kept in `manifests/model_weights_manifest.tsv` for optional
tasks. Download only the weights needed by the command you are running.

The default paths are:

- ALIGNN weights: `models/alignn_pretrained/*.zip`
- MatterSim checkpoint: download with differentiable MatterSim's
  `download_checkpoint` helper and pass the local path through
  `SMOACS_MATTERSIM_MODEL` or `--mattersim_model`

CrystalFormer weights are not downloaded by `scripts/download_model_weights.py`.
If you run the CrystalFormer examples, download the required files from the
[CrystalFormer GitHub page](https://github.com/deepmodeling/CrystalFormer) and
place them under `models/crystalformer/`.

### Auxiliary data

The small auxiliary files needed by the Materials Project initial-composition
generation scripts are included in this repository:

- `scripts/data_generations/density_ref_superhard.json`
- `data/intermidiate/mask_dict_super_common.npz`
- `data/intermidiate/ionic_mask_dict_super_common.npz`
- `data/intermidiate/ionic_radii_dict_super_common.npz`

Generated initial structures, run outputs, model weights, and MatterSim
reference caches are intentionally not committed.

### Smoke check

After installing dependencies and downloading the ALIGNN weights, run a minimal
ALIGNN smoke check with two candidates and ten optimization steps:

```bash
python create_dataset.py

python main.py \
  --model_name ALIGNN \
  --task bg_and_eform \
  --target_val 0.5 \
  --structure perovskite \
  --num_steps 10 \
  --b_size 2 \
  --num_candidate 2 \
  --num_workers 0 \
  --device cuda \
  --alignn_weights_dir models/alignn_pretrained \
  --exp_name smoke_perovskite_alignn
```




## Structure optimization
The main script `main.py` optimizes and evaluates structures towards specified
targeted properties under constraints. Evaluation results are output as CSV
files. Please refer to the [Output files section](#output-files) for details.

### Multi-property design of perovskite structures
Optimizing three properties—band gap, formation energy, and the tolerance factor $t$—under two
constraints: charge neutrality and preservation of perovskite structures (unit cell) .

Create the perovskite template structures before running the optimization:

```bash
python create_dataset.py
```

```bash
python main.py \
  --model_name ALIGNN \
  --task bg_and_eform \
  --target_val 0.5 \
  --structure perovskite \
  --num_steps 10 \
  --b_size 2 \
  --num_candidate 2 \
  --num_workers 0 \
  --device cuda \
  --alignn_weights_dir models/alignn_pretrained
```
or 

```bash
python main.py \
  --model_name Crystalformer \
  --task bg_and_eform \
  --target_val 0.5 \
  --structure perovskite \
  --num_steps 10 \
  --b_size 2 \
  --num_candidate 2 \
  --num_workers 0
```

### Scaling to large atomic configurations: optimization of 135-atom perovskites
Optimizing three properties—band gap, formation energy, and the tolerance factor $t$—under two
constraints: charge neutrality and preservation of large perovskite structures ($3\times3\times3$).

The same dataset creation command generates
`data/raw_data/initial_random_perovskite_3x3x3`:

```bash
python create_dataset.py
```

```bash
python main.py \
  --model_name ALIGNN \
  --task bg_and_eform \
  --target_val 4.5 \
  --structure perovskite3x3x3 \
  --num_steps 10 \
  --b_size 2 \
  --num_candidate 2 \
  --num_workers 0 \
  --device cuda \
  --alignn_weights_dir models/alignn_pretrained
```
or
```bash
python main.py \
  --model_name Crystalformer \
  --task bg_and_eform \
  --target_val 4.5 \
  --structure perovskite3x3x3 \
  --num_steps 10 \
  --b_size 2 \
  --num_candidate 2 \
  --num_workers 0
```

### Materials Project API key

Materials Project-based initial-structure generation and MP-backed convex-hull
workflows require a Materials Project API key. Export it once in the shell, and
pass it to scripts through the environment or with `--mp_api_key "$MP_API_KEY"`.
Do not commit real API keys to this repository.

```bash
export MP_API_KEY="your_materials_project_api_key"
```


### Bandgap Optimization
This command corresponds to the SMOACS side of "Section 2.3 Comparison with a
Leading Generative Model" in the paper. The target band gap is set to 3.0 eV.

Prepare Materials Project-based initial structures. This step requires
`MP_API_KEY`, the oxidation masks under `data/intermidiate/`, and a density
reference JSON at `scripts/data_generations/density_ref_superhard.json`.

```bash
PYTHONPATH=. \
python scripts/data_generations/generate_initials_mp_nsites_range.py \
  --n 256 \
  --seed 0 \
  --nsites_min 2 \
  --nsites_max 30 \
  --batch_size 128 \
  --density_ref scripts/data_generations/density_ref_superhard.json \
  --mask_paths data/intermidiate/mask_dict_super_common.npz \
  --radii_paths data/intermidiate/ionic_radii_dict_super_common.npz \
  --out_dir data/raw_data/initial_mp_nsites2_30 \
  --pkl_dir data/raw_data
```

Run SMOACS with the Section 2.3 hyperparameters:

```bash
python main.py \
  --model_name ALIGNN \
  --task bg_and_eform \
  --target_val 3.0 \
  --e_form_coef 0.0 \
  --initial_structures_path data/raw_data/initial_mp_nsites2_30__mask_dict_super_common.pkl \
  --optimizer grad_sa_hybrid \
  --num_steps 600 \
  --sa_start_step 200 \
  --sa_t0 1.3 \
  --sa_tmin 6.4e-5 \
  --sa_coords_sigma 4.0e-3 \
  --sa_lattice_sigma 4.5e-4 \
  --sa_angle_sigma 2.5e-3 \
  --allow_lr_override \
  --coords_lr 9.7e-3 \
  --lattice_lr 2.3e-6 \
  --atom_lr 2.6e-3 \
  --b_size 128 \
  --copy_mutation_override "copy_mutation___steps_cm-150-300-450__mutation_noise-0.00053__top_ratio-0.20__group_C_use_rate-0.020__atom_dist_init-True" \
  --alignn_weights_dir models/alignn_pretrained \
  --exp_name section_2_3_bandgap_3ev
```

### Half metal proposal
This command corresponds to "Section 2.4 Constrained SMOACS--DFT search for
half-metal candidates" in the paper.

Prepare Materials Project-based initial structures with the half-metal seed
set. This step requires `MP_API_KEY`, the oxidation masks under
`data/intermidiate/`, and a density reference JSON at
`scripts/data_generations/density_ref_superhard.json`.

```bash
PYTHONPATH=. \
python scripts/data_generations/generate_initials_mp_nsites_range.py \
  --n 256 \
  --seed 0 \
  --nsites_min 2 \
  --nsites_max 30 \
  --batch_size 150 \
  --half_metal_samples \
  --density_ref scripts/data_generations/density_ref_superhard.json \
  --mask_paths data/intermidiate/mask_dict_super_common.npz \
  --radii_paths data/intermidiate/ionic_radii_dict_super_common.npz \
  --out_dir data/raw_data/initial_mp_nsites2_30_half_metal \
  --pkl_dir data/raw_data
```

This writes
`data/raw_data/initial_mp_nsites2_30__mask_dict_super_common.pkl`. The level5
half-metal element restriction is applied at optimization time through
`configs/half_metal_level5_forbidden.yaml`, so the initial pkl does not need a
separate baked-in level5 mask. For ALIGNN pkl inputs, SMOACS reapplies the
generated forbidden-element mask to precomputed element masks during loading.

The forbidden elements are introduced as an optimization-time mask constraint.
Initial-structure generation uses the base oxidation mask:

```bash
--mask_paths data/intermidiate/mask_dict_super_common.npz
```

The SMOACS run then passes the same base mask and a config file:

```bash
--mask_dict_path data/intermidiate/mask_dict_super_common.npz \
--config configs/half_metal_level5_forbidden.yaml
```

The config adds the level5 forbidden list:

```yaml
element_constraints:
  mask_dict_path: data/intermidiate/mask_dict_super_common.npz
  forbidden_elements:
    - Li
    - Sc
    - Co
    - Ni
    - Ga
    - Y
    - Cd
    - La
    - Ce
    - Pr
    - Nd
    - Pm
    - Sm
    - Eu
    - Gd
    - Tb
    - Dy
    - Ho
    - Er
    - Tm
    - Yb
    - Lu
    - Pt
    - Hg
    - Pb
```

At runtime, `main.py` writes a derived mask under
`data/intermidiate/generated_masks/` by setting those element indices to zero
for every oxidation-state vector in the base `.npz`. With ALIGNN `.pkl`
initial structures, the loader also applies the same forbidden indices to the
precomputed `atom_mask_for_all_ox_states` and `radii_for_all_ox_states`. To use
a different forbidden list, copy `configs/half_metal_level5_forbidden.yaml`,
edit `forbidden_elements`, and pass the new file with `--config`.

Run SMOACS with the Section 2.4 hyperparameters:

The CLI default for `--element_count_l1_weight` is `1.0`. The penalty is active
only when `--use_element_count_l1_loss` is passed. The half-metal proposal
setting below uses `1000`.

```bash
MATTERSIM_LOGURU_ENQUEUE=0 \
FORCE_DATALOADER_SINGLE_PROCESS=1 \
python main.py \
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
  --magmom_checkpoint jv_magmom_alignn \
  --initial_structures_path data/raw_data/initial_mp_nsites2_30__mask_dict_super_common.pkl \
  --mask_dict_path data/intermidiate/mask_dict_super_common.npz \
  --config configs/half_metal_level5_forbidden.yaml \
  --optimizer grad_sa_hybrid \
  --num_steps 650 \
  --grad_sa_hybrid_grad_steps 300 \
  --sa_t0 0.18 \
  --sa_tmin 1.2e-3 \
  --sa_trials 1 \
  --sa_coords_sigma 8.8e-3 \
  --sa_lattice_sigma 1.0e-3 \
  --sa_angle_sigma 4.3e-2 \
  --allow_lr_override \
  --coords_lr 2.4e-6 \
  --lattice_lr 1.1e-6 \
  --atom_lr 4.1e-4 \
  --b_size 150 \
  --copy_mutation_override "copy_mutation___steps_cm-162-324-486__mutation_noise-0.000000012__top_ratio-0.20__group_C_use_rate-0.020__atom_dist_init-True" \
  --abc_min 2 \
  --abc_max 20 \
  --energy_coef 1.0 \
  --energy_every 1 \
  --mattersim_model /path/to/mattersim-v1.0.0-5M.pth \
  --use_element_count_l1_loss \
  --w_mag_int 1 \
  --element_count_l1_weight 1000 \
  --alignn_weights_dir models/alignn_pretrained \
  --exp_name section_2_4_half_metal
```




## Output files
| file path | description |
| ---- | ---- |
|`results/{exp_name}/result_summary.csv`| summary scores|
| `results/{exp_name}/result.csv`| scores for each optimized crystal structures |
| `results/{exp_name}/result.npz`| detailed scores with numpy |
|`results/{exp_name}/poscar/*.vasp`| optimized crystal structures written as POSCAR files|
| `results/{exp_name}/history_img.png`| an image file of optimization history. (if you optimize both band gap and formation energy) |

### Details for `result_summary.csv`


This file provides a summary of the optimization performance across all generated structures, including metrics such as success rates, uniqueness, and novelty.

| Column | Description |
|--------|-------------|
| `success` | Ratio of structures that satisfy all evaluation criteria (see `success` in the main result file). |
| `*_success` | Ratio of structures that satisfy the criteria of each peoperty or constraint. |
| `tolerance_success` | Ratio of structures that satisfy the tolerance factor constraint. |
| `perov_success` | Ratio of perovskite structures that pass the geometry and coordinate checks. |
| `Tc_above_5.0`, `Tc_above_10.0`, ... | Ratio of structures whose predicted Tc exceeds the specified threshold (used in maximization mode). |
| `is_neutral_cmn` | Ratio of structures that are electrically neutral. |
| `minbond_less_than_0.5` | Ratio of structures with all bond lengths ≥ 0.5Å. |
| `success_nuique_rate` | Fraction of successful structures that are also structurally unique. |
| `newly_megnet_success` | Number of successful structures that are novel (not present in the reference MEGNet dataset). |

This summary helps users understand the effectiveness of the optimization process and the quality of generated candidates at a glance.


### Details for `result.csv`
A file for evaluation result for each optimized structures including successeeded and failed ones.
| column | description |
| ---- | ---- |
| `original_fnames`| file names of initial structures in the directory of `dir_path` |
| `batch_abc` | values of lengths of crystal vectors $a,b,c$ |
| `batch_angle`| values of angles of crystal vector $\alpha,\beta,\gamma$|
| `*_onehot` | optimized values with one-hot atomic distributions. e.g., `bandgap_onehot`|
| `loss_*_onehot` |loss values after optimization with one-hot atomic distributions. e.g., `loss_bandgap_onehot`|
| `*_success` | whether to meet its criterion|
| `Tc_above_5.0`, `Tc_above_10.0`, ... |  whether predicted Tc exceeds the specified threshold (used in maximization mode). |
| `is_neutral_cmn`  | electrical neutrality|
| `minbond_less_than_0.5` | whether the all bonds are longer than 0.5Å |
| `ox_states`  | oxidation numbers (available only if the structure is electrically neutral). The order is the same as `elements`.|
| `perov_success`  | whether to  meet both (85°<$\alpha,\beta,\gamma$< 95°) and `perov_coords`|
| `tolerance_success`  | whether tolerance value $t$ meets its criterion|
| `perov_coords`  | whether the displacements of fractional coordinates are within $\epsilon$ (0.15) |
| `success`  | overall success. If the following conditions are met at the same time: `*_success` and `valid_structure` for non-perovskite optimization; `*_success`, `valid_structure` and `perov_success` for perovskite optimization|

## MatterSim convex-hull evaluation from SMOACS POSCAR outputs

The helper `scripts/prepare_mattersim_poscars_from_smoacs_results.sh` prepares
the POSCAR directory layout expected by
`scripts/calculate_mattersim_convex_hull_energy.py`.
The example below runs a small half-metal proposal with two candidates and then
evaluates the resulting chemical systems with the MP-only MatterSim convex-hull
workflow. It uses `--element_count_l1_weight 1000` only for this half-metal
proposal run; the general CLI default remains `1.0`.

```bash
export MP_API_KEY="your_materials_project_api_key"
export SMOACS_MATTERSIM_MODEL="$HOME/.local/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth"

EXP_NAME=half_metal_c2_s50_el1000
SMOACS_EXP_NAME="$EXP_NAME" \
SMOACS_NUM_STEPS=50 \
SMOACS_GRAD_STEPS=1 \
SMOACS_BATCH_SIZE=2 \
SMOACS_NUM_CANDIDATE=2 \
SMOACS_ELEMENT_COUNT_L1_WEIGHT=1000 \
scripts/reproduce_smoacs/half_metal_search.sh
```

If the initial-structure pkl, ALIGNN weights, or MatterSim checkpoint are stored
outside the repository, set `HALF_METAL_INITIAL_STRUCTURES_PATH`,
`HALF_METAL_ALIGNN_WEIGHTS_DIR`, or `SMOACS_MATTERSIM_MODEL` before running the
script.

```bash
scripts/prepare_mattersim_poscars_from_smoacs_results.sh \
  --condition "$EXP_NAME" \
  --out-root to_relax_poscars
```

The MP-only convex-hull workflow follows the same stage order and directory
parameters used in the paper workflow:
Stage1 `prefetch_MP`, Stage2 `MatterSim_phase1`, then `export_MP_only`.
For multiple proposal directories, Stage2 can be run in parallel; the commands
below show the single-directory version for the half-metal example.

```bash
python scripts/calculate_mattersim_convex_hull_energy.py \
  -stage prefetch_MP \
  --mp_api_key "$MP_API_KEY" \
  -p_dir "to_relax_poscars/$EXP_NAME" \
  -s_dir "DFT_POSCARS/$EXP_NAME" \
  -r_dir "prop_relaxed_poscars/$EXP_NAME" \
  -rr_dir "result_dir/$EXP_NAME" \
  --cache_dir cache

python scripts/calculate_mattersim_convex_hull_energy.py \
  -stage MatterSim_phase1 \
  --mp_api_key "$MP_API_KEY" \
  -p_dir "to_relax_poscars/$EXP_NAME" \
  -s_dir "DFT_POSCARS/$EXP_NAME" \
  -r_dir "prop_relaxed_poscars/$EXP_NAME" \
  -rr_dir "result_dir/$EXP_NAME" \
  --cache_dir cache

python scripts/calculate_mattersim_convex_hull_energy.py \
  -stage export_MP_only \
  --mp_api_key "$MP_API_KEY" \
  -p_dir "to_relax_poscars/$EXP_NAME" \
  -s_dir "DFT_POSCARS/$EXP_NAME" \
  -r_dir "prop_relaxed_poscars/$EXP_NAME" \
  -rr_dir "result_dir/$EXP_NAME" \
  --cache_dir cache
```

To include Alexandria Materials Database references, continue with Stage3 and
Stage4 after the MP stages above. `--alexandria_base_url` defaults to
`https://alexandria.icams.rub.de/pbe`, so it only needs to be specified when
using a different Alexandria endpoint.

```bash
python scripts/calculate_mattersim_convex_hull_energy.py \
  -stage prefetch_Alex \
  -p_dir "to_relax_poscars/$EXP_NAME" \
  -s_dir "DFT_POSCARS/$EXP_NAME" \
  -r_dir "prop_relaxed_poscars/$EXP_NAME" \
  -rr_dir "result_dir/$EXP_NAME" \
  --cache_dir cache

python scripts/calculate_mattersim_convex_hull_energy.py \
  -stage MatterSim_phase2 \
  -p_dir "to_relax_poscars/$EXP_NAME" \
  -s_dir "DFT_POSCARS/$EXP_NAME" \
  -r_dir "prop_relaxed_poscars/$EXP_NAME" \
  -rr_dir "result_dir/$EXP_NAME" \
  --cache_dir cache
```
