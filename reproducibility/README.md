# Reproducing the Paper: Results Sections 2.1–2.4

This directory contains the run scripts, initial structures, masks, and DFT inputs for each section.
See the [repository README](../README.md) for environment setup and model preparation.
Download the fixed inputs from
[reproducibility.tar.gz](https://drive.google.com/file/d/1eCET1IfM5W-HxZ15FSpY1xkFAZDyGvz9/view?usp=sharing)
and extract the data directories using the command in the repository README.
Use the scripts in this Git checkout.
The directory belongs at `<repo_root>/reproducibility/`.

## Example

This example runs SMOACS optimization for Section 2.3 (five target band gaps from 2.5 to 4.5 eV),
then relaxes the structures proposed for the 2.5 eV target with MatterSim and evaluates
$\mathrm{E_{hull}}$ using Materials Project and Alexandria reference phases.
The section scripts change to the repository root internally, so they can be launched from their section directories.
Set your Materials Project API key in the `MP_API_KEY` environment variable before fetching reference data for MatterSim.

```bash
cd reproducibility/section_2_3
bash run_section_2_3.sh # SMOACS optimization.
python ../prepare_mattersim_inputs.py section_2_3 section_2_3_bg2p5
bash ../run_mattersim.sh section_2_3 section_2_3_bg2p5 # MatterSim relaxation and E_hull eval.
```

`run_section_2_3.sh` runs all five targets sequentially. Apply the two MatterSim commands to each condition by changing the experiment name.
SMOACS outputs are saved under `<repo_root>/results/<exp_name>/`; MatterSim outputs are saved under `mattersim/` in each section directory.

In each section, `templates_masks/` contains the initial structure templates and masks, and `run_section_*.sh` runs SMOACS optimization.
Optimized structures are saved in POSCAR format under `results/<exp_name>/poscar/` at the repository root.

## Minimal execution check

Set `SMOACS_DEVICE=cpu` for CPU optimization; the default is `cuda`.
Crystalformer uses its existing PyTorch implementation on the CPU.

The same four section scripts support `SMOACS_SMOKE_TEST=1`. Each uses two
candidates, batch size two, and ten steps, keeping the original inputs, models,
loss coefficients, and seed 42. All 35 combinations run once (16 + 8 + 5 + 6);
Section 2.1 uses only seed 42 in this mode and skips the four-run mean.
The experiment name receives a `__smoke` suffix, so full-run outputs are separate.
Run this example from the repository root after preparing the environment,
models, and inputs:

```bash
export SMOACS_SMOKE_TEST=1
export SMOACS_DEVICE=cpu
cd reproducibility/section_2_3
bash run_section_2_3.sh
python ../prepare_mattersim_inputs.py section_2_3 section_2_3_bg2p5__smoke
bash ../run_mattersim.sh section_2_3 section_2_3_bg2p5__smoke
```

Set `MP_API_KEY` before the MatterSim command. Use a fresh output directory when
repeating the check. Unset `SMOACS_SMOKE_TEST` in the shell before full runs.
The schedules below use zero-based steps:

| Section | Smoke mutation steps | Smoke hybrid gradient steps |
| --- | --- | --- |
| 2.1 | 5, 7 | Not a hybrid optimizer |
| 2.2 | 2, 5, 7; Crystalformer unit cell: 5, 7 | Not a hybrid optimizer |
| 2.3 / 2.4 | 2, 5, 7 | 5 |

`run_mattersim.sh` uses a relaxation limit of ten steps in smoke mode and records
that value in `run_environment.json`; the default is 400. Inspect relaxation
status before using hull energies. Section 2.4 may select zero structures at this scale;
the existing selection criteria are retained and the MatterSim wrapper requires
at least one prepared structure. Small candidate sets can also leave no actual
copy/mutation targets even when the scheduled operation is called.

## Directories and Files

```text
<repo_root>/
├── results/                       # Created when SMOACS runs
│   └── <exp_name>/
│       └── poscar/
└── reproducibility/
    ├── README.md
    ├── experiment_manifest.md
    ├── prepare_mattersim_inputs.py
    ├── run_mattersim.sh
    ├── section_2_1/
    │   ├── run_section_2_1.sh
    │   ├── section_2_1_main.py
    │   ├── summarize_runs.py
    │   └── templates_masks/
    │       ├── initial_structures/
    │       │   ├── template_order.txt
    │       │   └── *.vasp                 # 256 initial templates
    │       └── masks/
    │           ├── initial_masks.npz
    │           └── mask_dict_super_common.npz
    ├── section_2_2/
    │   ├── run_section_2_2.sh
    │   └── templates_masks/
    ├── section_2_3/
    │   ├── run_section_2_3.sh
    │   └── templates_masks/
    └── section_2_4/
        ├── run_section_2_4.sh
        ├── templates_masks/
        └── half-metal_candidate_dft_inputs/
```

The following files are written to `results/`. See the README.md at the repository root for details.

| File path | Description |
| --- | --- |
| `results/<exp_name>/result_summary.csv` | Aggregate evaluation metrics, success rates, and other run-level statistics |
| `results/<exp_name>/result.csv` | Evaluation results for each candidate, associated with its initial structure ID |
| `results/<exp_name>/result.npz` | Arrays containing optimization results |
| `results/<exp_name>/poscar/*.vasp` | Structures immediately after SMOACS optimization; POSCAR is the default output format |
| `results/<exp_name>/history_img.png` | Plot of the optimization history |
| `results/<exp_name>/opt_timing.jsonl` | Records of data counts, processing times, and related information (Sections 2.2–2.4) |
| `results/<exp_name>/run_config.json` | Effective settings for the Section 2.1 ablation run |

| File | Purpose |
| --- | --- |
| [experiment_manifest.md](experiment_manifest.md) | Mapping of sections, experimental conditions, inputs, seeds, run scripts, and output locations |
| `run_section_2_*.sh` in each section | SMOACS arguments, including candidate count, batch size, seed, learning rates, loss coefficients, and optimizer |

## Section 2.1: Crystalformer Ablation Study

Run the following commands from the repository root after preparing the Crystalformer models
described in the root README:

```bash
cd reproducibility/section_2_1
bash run_section_2_1.sh
```

[run_section_2_1.sh](section_2_1/run_section_2_1.sh) calls
[section_2_1_main.py](section_2_1/section_2_1_main.py) for each condition.
It resolves the repository root automatically, so it can be launched from the section directory.
The script runs each of the 16 combinations below four times, using 128 candidates,
a batch size of 128, and 200 optimization steps per run (64 runs total).
This package assigns seeds 42, 43, 44, and 45 to the four runs and reports the
arithmetic mean of their run-level metrics, following Table 1.
The target band gap is 2.5 ± 0.04 eV;
formation energy is optimized as well, with a success criterion of `< -0.5 eV/atom`.
The atom-distribution, lattice, and coordinate learning rates are 0.007, 0.002, and 0.005.

| Initialization | Initial lattice and coordinates | Oxidation-state mask | Initial element distribution |
| --- | --- | --- | --- |
| `random-random` | Randomized | Disabled | Uniform |
| `random-AtomicMask` | Randomized | Enabled | Uniform |
| `Struct-random` | From templates | Disabled | Uniform |
| `Struct-AtomicMask` | From templates | Enabled | Uniform |

Each initialization is combined with `grad` (Adam), `grad_sa` (Adam + SA),
`mutation` (Adam + mutation), and `mutation_sa` (Adam + SA + mutation).
SA uses `T0=0.1`, `alpha=0.9`, and `noise_scale=0.1`. Mutation uses zero-based
steps 100 and 150, noise 0.00007, `top_ratio=0.10`, `group_C_use_rate=0.05`,
and `atom_dist_init=True`. Here, `group_C_use_rate` is the fraction of the remaining
group that skips mutation.

To run one repetition of a single condition, use the following command from `reproducibility/section_2_1/`.
This example uses template geometry, oxidation-state masks, and Adam + SA + mutation:

```bash
python section_2_1_main.py \
    --initialization Struct-AtomicMask \
    --method mutation_sa \
    --num_candidate 128 \
    --b_size 128 \
    --num_steps 200 \
    --mutation_steps 100 150 \
    --seed 42 \
    --exp_name section_2_1_Struct-AtomicMask_mutation_sa__seed42
```

The 256 POSCAR templates are in `templates_masks/initial_structures/`; each run uses
the first 128 in the order recorded in `template_order.txt`.
The supplied templates are unchanged copies of the first 256
files in the original directory read order of
`data/raw_data/initial_candidates_from_megnet_max_10atoms_neutral_super_common/`.
The associated masks are frozen in `templates_masks/masks/initial_masks.npz`, with
template IDs and site counts stored alongside the arrays. During preparation, element
order follows the POSCAR species order and oxidation-state choices are sorted before
retaining up to 10 neutral patterns. Masks are loaded directly during optimization.

Seeds are reset before each run, and all four groups start from uniform element
distributions. For the randomized groups, a given seed produces the same initial geometry across
the four update methods. Randomization retains each template's atom count and lattice
scaling; it does not generate structures independently of all template information.

Outputs are saved directly under
`results/section_2_1_<initialization>_<method>__seed<seed>/`. Each run writes its configuration,
evaluation results, summary, POSCAR structures, and optimization history.
`run_config.json` records the selected template IDs and effective settings;
`result_summary.csv` contains the summary for that run.
The script stops if an output directory already exists.

After all 64 runs finish, [summarize_runs.py](section_2_1/summarize_runs.py)
reads the four `result_summary.csv` files per condition and writes
`results/section_2_1_mean.csv` (16 rows). The `*_mean` columns are arithmetic
means across the four runs; success rates are fractions between 0 and 1.
Run configurations are checked before averaging, and missing runs cause an error.
Undefined metrics remain undefined in the mean rather than being dropped.
To generate this file separately after completing all runs, use the following command
from the repository root. An existing mean file is not overwritten.

```bash
python reproducibility/section_2_1/summarize_runs.py
```

## Section 2.2: Multi-property Optimization of Perovskites

The eight conditions use the first 128 templates in each of the author-selected
unit-cell and 3×3×3 input directories, in `template_order.txt` order.
The script explicitly sets the following learning rates, including the manuscript
values for ALIGNN unit-cell coordinates and Crystalformer 3×3×3 optimization:

| Model / cell | Lattice learning rate | Coordinate learning rate | Atom-distribution learning rate |
| --- | --- | --- | --- |
| ALIGNN / unit cell | 9.0 | 0.002 | 0.00005 |
| Crystalformer / unit cell | 0.002 | 0.005 | 0.007 |
| ALIGNN / 3×3×3 | 0.01 | 0.000667 | 0.00004 |
| Crystalformer / 3×3×3 | 0.6 | 0.00067 (6.7 × 10⁻⁴) | 0.002 |

```bash
cd reproducibility/section_2_2
bash run_section_2_2.sh
```

## Section 2.3: Band Gap Optimization and Comparison

Run five conditions (2.5 / 3.0 / 3.5 / 4.0 / 4.5 eV), each with 512 candidates, a batch size of 128, and seed 42.
Run the following commands from the repository root. Set your Materials Project API key in
the `MP_API_KEY` environment variable beforehand. After SMOACS optimization, the proposed
structures for all five conditions are relaxed sequentially with MatterSim, and $E_{\mathrm{hull}}$
is calculated using MP + Alexandria reference phases.

```bash
set -e
cd reproducibility/section_2_3
bash run_section_2_3.sh # SMOACS optimization

for target in 2.5 3.0 3.5 4.0 4.5; do
    exp_name="section_2_3_bg${target/./p}"
    python ../prepare_mattersim_inputs.py section_2_3 "$exp_name" # Prepare proposed structures
    bash ../run_mattersim.sh section_2_3 "$exp_name" # MatterSim relaxation and E_hull calculation
done
```

## Section 2.4: Half-metal Candidate Search

[run_section_2_4.sh](section_2_4/run_section_2_4.sh) runs six conditions combining
element-count loss weights of 100 / 1000 with integer magnetic moment loss weights of 1 / 10 / 100.
The initial structures and oxidation-state masks are combined in a single file,
`templates_masks/initial_mp_nsites2_30__mask_dict_super_common_level5.pkl`.
Run the following commands from the repository root. Set your Materials Project API key in
the `MP_API_KEY` environment variable beforehand. After SMOACS optimization, proposed structures
meeting the selection criteria below are prepared for all six conditions, relaxed with MatterSim,
and evaluated for $E_{\mathrm{hull}}$ using MP + Alexandria reference phases.

```bash
set -e
cd reproducibility/section_2_4
bash run_section_2_4.sh # SMOACS optimization

for mag_weight in 1 10 100; do
    for element_weight in 100 1000; do
        exp_name="section_2_4_best_trial00010__elm-w${element_weight}__level5__w-mag-int${mag_weight}"
        python ../prepare_mattersim_inputs.py section_2_4 "$exp_name" # Prepare selected proposed structures
        bash ../run_mattersim.sh section_2_4 "$exp_name" # MatterSim relaxation and E_hull calculation
    done
done
```

The specified PKL contains 10,050 entries; the first 8,192 are used.
The final batch of 92 candidates is retained.

### DFT Inputs for the Three Included Candidates

Each `half-metal_candidate_dft_inputs/<candidate_id>/` directory contains `relax1/`, `relax2/`,
`static/`, and `dos/`, with POSCAR, INCAR, KPOINTS, and POTCAR TITEL information for each stage.
The POTCAR files themselves are not included. Running DFT requires the outputs from the preceding
stage, the POTCAR files, and a VASP environment.

## MatterSim Convex-hull Evaluation: Overview, Inputs, and Outputs

`scripts/calculate_mattersim_convex_hull_energy.py` relaxes the POSCAR structures proposed
by SMOACS with MatterSim and calculates convex-hull energies relative to reference phases
retrieved from Materials Project (MP) and Alexandria. It also records the uniqueness of the
relaxed structures and their novelty relative to the reference phases. This script does not
perform DFT calculations or determine half-metallicity from DOS.

For Sections 2.3 and 2.4, first copy the structures from `results/<exp_name>/poscar/`
to the experiment-specific `to_relax_poscars/` directory, then run the four MP + Alexandria stages.
Below, `<S>` denotes `section_2_3` or `section_2_4`, `<E>` is the experiment name specified
by SMOACS's `--exp_name`, and `<M>` denotes `reproducibility/<S>/mattersim`.

| Input/output | File or directory | Description and corresponding argument |
| --- | --- | --- |
| Source | `results/<E>/result.csv`, `results/<E>/poscar/*.vasp` | SMOACS optimization results and proposed structures |
| Input | `<M>/to_relax_poscars/<E>/*.vasp` | Proposed structures copied without modification for MatterSim. `-p_dir` |
| Input preparation record | `reproducibility/<S>/screening_logs/<E>_mattersim_inputs.csv` | Selection results, original IDs, MatterSim IDs, destination paths, and SHA-256 hashes for all SMOACS rows |
| Output | `<M>/relaxed_poscars/<E>/*_relaxed.vasp` | Proposed structures relaxed with MatterSim. `-r_dir` |
| Output | `<M>/results/<E>/relax_results.csv` | Includes `fname`, `e_hull_mp`, `e_hull_mp_alex`, `structural_novelty`, and `unique_structure`. `-rr_dir` |
| Output | `<M>/DFT_POSCARS/<E>/<proposal_id>/` | POSCAR files for proposed structures and reference phases selected based on stability, plus `material_summary.csv`. `-s_dir`. These are not complete DFT inputs or DFT results |
| Record | `<M>/cache/<E>/screening_results.jsonl` | First-stage hull evaluation using MP references. `--cache_dir` |
| Record | `<M>/cache/<E>/proposal_relax_records.jsonl` | Relaxation records for proposed structures, including failure status |
| Reference retrieval records | `mp_requests_cache.sqlite`, `alex_requests_cache.sqlite`, `*_fetch_done.json`, and `chemsys_to_fetch.json` under `<M>/cache/<E>/` | Retrieved reference phases, chemical systems already fetched, and targets for Alexandria retrieval |
| Reference relaxation data | `<M>/reference/<E>/` | Relaxed reference structures, `relax_results.json`, `relax_results_*.json.gz` for each chemical system, and `DFT_POSCARS/` for reference selection. `--global_relax_dir` |
| Run records | `<M>/results/<E>/run_environment.json`, `prefetch_MP.log`, `MatterSim_phase1.log`, `prefetch_Alex.log`, `MatterSim_phase2.log` | Environment, model, and script identification information, plus standard output and standard error for each stage |
| Progress and timing | `<M>/results/<E>/logs/time_logs/`, `<M>/results/<E>/timing_logs/` | Records written to the working directory by the underlying script |

## Preparing SMOACS Outputs and Running MatterSim

Confirm that SMOACS optimization has completed and that `results/<E>/result.csv` and `poscar/` exist.
`run_section_2_3.sh` / `run_section_2_4.sh` perform optimization only; run the post-processing below separately.
Follow the repository root README to prepare the environment and the 5M model.
The following example processes the 2.5 eV condition in Section 2.3 from the repository root.

```bash
python reproducibility/prepare_mattersim_inputs.py \
  section_2_3 section_2_3_bg2p5

# Set your Materials Project API key in the MP_API_KEY environment variable before running.
bash reproducibility/run_mattersim.sh \
  section_2_3 section_2_3_bg2p5
```

You can also run these commands from the section directory:

```bash
cd reproducibility/section_2_3
python ../prepare_mattersim_inputs.py section_2_3 section_2_3_bg2p5
bash ../run_mattersim.sh section_2_3 section_2_3_bg2p5
```

An example for Section 2.4 is shown below.

```bash
python reproducibility/prepare_mattersim_inputs.py \
  section_2_4 section_2_4_best_trial00010__elm-w1000__level5__w-mag-int10
bash reproducibility/run_mattersim.sh \
  section_2_4 section_2_4_best_trial00010__elm-w1000__level5__w-mag-int10
```

`prepare_mattersim_inputs.py` checks that all CSV rows match the POSCAR files before preparing inputs.
It does not re-relax or standardize structures, or change site ordering.
It adds `__<E>` to the original `Optimized_based_on_<original_fnames>` name to create the MatterSim
filename and records its mapping to the original template-derived ID in a CSV file.

- Section 2.3: Prepare all proposed structures without additional filtering by band gap or element count.
- Section 2.4: Stage 1 requires the saved `bandgap_onehot <= 0.05 eV` and an element count of `<= 5`.
  Only structures that also have a saved `minbond_less_than_0.5 == True` are prepared.
  The saved decision in the original CSV is used
  without recalculating the distance check. The two Stage 1 criteria and distance-based exclusions
  are recorded in separate columns of the input preparation record.
  For the original six level5 experiments, 21,712 of 49,152 candidates passed Stage 1;
  excluding 118 based on distance left 21,594 MatterSim inputs.

Input preparation stops if the destination or input preparation record already exists.
The calculation wrapper also stops if calculation outputs, caches, or reference relaxation
data already exist for that experiment name. Use a new `--exp_name` for new run conditions
to avoid mixing outputs under the same name. The wrapper launches Python directly;
it does not submit Slurm jobs or send email or external chat notifications.
Use `PYTHON=/path/to/python` to select the Python interpreter.

`run_mattersim.sh` calls `scripts/calculate_mattersim_convex_hull_energy.py` in the following
order, using the same input and output paths for each stage.

| Order | `--stage` | Operation |
| --- | --- | --- |
| 1 | `prefetch_MP` | Retrieve MP reference phases for the chemical systems of the proposed structures |
| 2 | `MatterSim_phase1` | Relax proposed structures and MP reference phases, then evaluate MP hull energies. Pass chemical systems with MP hull `< 0.1 eV/atom` to the next stage |
| 3 | `prefetch_Alex` | Retrieve Alexandria reference phases for the selected chemical systems |
| 4 | `MatterSim_phase2` | Evaluate MP + Alexandria hull energies, uniqueness, and novelty; output the evaluation CSV and a list of structures for DFT |

The `fname` column in `relax_results.csv` corresponds to `mattersim_fname` in the input preparation record.
Compute S.U.N. statistics using `e_hull_mp_alex < 0.1 eV/atom`, `unique_structure == True`,
and `structural_novelty == 1`, retaining the original number of proposed structures as the denominator.
Because `relax_results.csv` does not list all proposals rejected at the MP stage, join it with the
input preparation record, `screening_results.jsonl`, and `proposal_relax_records.jsonl` to track
outcomes for the full set. Do not automatically classify missing rows or NaN values as unstable.
Structures in `DFT_POSCARS/` are selected based on stability; their inclusion does not imply
that they satisfy all S.U.N. criteria.

Uniqueness is evaluated within the set of proposed structures supplied to a single run.
Splitting one experiment across multiple `-p_dir` inputs changes that set, so supply all
structures for each condition in a section together.

For comparisons using MP alone, use `all_MP_only` or `export_MP_only` in the underlying script.

## MatterSim Model and Calculation Settings

The hull script uses the standard 5M checkpoint at
`~/.local/mattersim/pretrained_models/mattersim-v1.0.0-5M.pth`, BFGS,
ExpCellFilter, and `constrain_symmetry=True` for relaxation. The wrapper
explicitly sets `--steps_max 400`, or 10 with `SMOACS_SMOKE_TEST=1`.
The `SMOACS_MATTERSIM_MODEL` environment variable used during optimization must
point to this same standard checkpoint for this reproduction workflow.
The force convergence threshold is not specified in the call; the default for `Relaxer.relax`
in the inspected differentiable-MatterSim implementation is `fmax=0.01`.
Reference phases are retrieved using a threshold of `0.3 eV/atom` for database-reported hull energies;
the candidate selection threshold is `< 0.1 eV/atom`.
