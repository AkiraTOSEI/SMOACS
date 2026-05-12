import argparse
import copy
import sys
from pathlib import Path
import os
from typing import Optional
import numpy as np

import src.main_experiment as _main_experiment_module
from src.main_experiment import main_experiment
from src.smoacs_opt_eval.alignn_graph_update import calculate_update_steps
from src.spg_wp.P321 import P321Wyckoff
from src.spg_wp.Fbar43m import Fbar43mWyckoff
from src.spg_wp.Amm2 import Amm2Wyckoff
from src.spg_wp.R3m import R3mWyckoff
from src.spg_wp.P6_3m import P63mWyckoff
from src.spg_wp.P4mmm import P4mmmWyckoff
from src.spg_wp.P2m import P2mWyckoff
from src.spg_wp.wyckoff_expander import WyckoffExpander
import pandas as pd
from src.utils.common import get_master_dir
from src.utils.config_loader import load_config
from src.utils.element_constraints import (
    get_element_constraints,
    select_mask_dict_path,
    write_forbidden_element_mask,
)
from src.utils.eform_only_mode import (
    EFORM_ONLY_TASK,
    LEGACY_EFORM_ONLY_TASK,
    build_eform_only_alignn_config,
    validate_eform_only_args,
)


# Optional workaround: force all DataLoaders to single-process to avoid SemLock permission errors
if os.environ.get("FORCE_DATALOADER_SINGLE_PROCESS") in {"1", "true", "True"}:
    import torch.utils.data.dataloader as _torch_dataloader

    _orig_dataloader_init = _torch_dataloader.DataLoader.__init__

    def _patched_dataloader_init(self, *args, **kwargs):
        # enforce single-process workers
        kwargs["num_workers"] = 0
        kwargs["persistent_workers"] = False
        # prefetch_factor is invalid when num_workers=0
        kwargs.pop("prefetch_factor", None)
        return _orig_dataloader_init(self, *args, **kwargs)

    _torch_dataloader.DataLoader.__init__ = _patched_dataloader_init
    print("[info] FORCE_DATALOADER_SINGLE_PROCESS enabled: num_workers forced to 0")


def _patch_max_ox_states(max_ox_patterns: int) -> None:
    """Force max_ox_states at data-loading time without editing src/main_experiment.py."""
    original_load_initial_data = _main_experiment_module.load_initial_data

    # Guard against accidental double wrapping in the same process.
    if getattr(original_load_initial_data, "_is_max_ox_states_patched", False):
        return

    def _wrapped_load_initial_data(*args, **kwargs):
        kwargs["max_ox_states"] = max_ox_patterns
        settings_dict = kwargs.get("settings_dict")
        if isinstance(settings_dict, dict):
            settings_dict["max_ox_states"] = max_ox_patterns
        return original_load_initial_data(*args, **kwargs)

    _wrapped_load_initial_data._is_max_ox_states_patched = True
    _main_experiment_module.load_initial_data = _wrapped_load_initial_data


def _resize_last_dim(arr: np.ndarray, target: int) -> np.ndarray:
    current = int(arr.shape[-1])
    if current == target:
        return arr
    if current > target:
        return arr[..., :target]
    pad_shape = arr.shape[:-1] + (target - current,)
    pad = np.zeros(pad_shape, dtype=arr.dtype)
    return np.concatenate([arr, pad], axis=-1)


def _adjust_input_dataframe_ox_patterns(data_list, max_ox_patterns: int) -> None:
    """Resize oxidation-related arrays in input_dataframe records to max_ox_patterns."""
    if not data_list:
        return

    target = int(max_ox_patterns)
    keys = (
        "atom_mask_for_all_ox_states",
        "radii_for_all_ox_states",
        "ox_states_used_mask",
    )

    before = None
    for key in keys:
        sample = data_list[0].get(key)
        if sample is None:
            continue
        sample = np.asarray(sample)
        if sample.ndim > 0:
            before = int(sample.shape[-1])
            break

    for row in data_list:
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            arr = np.asarray(value)
            if arr.ndim == 0:
                continue
            row[key] = _resize_last_dim(arr, target)

    if before is not None and before != target:
        print(
            f"[info] input_dataframe oxidation pattern width adjusted: {before} -> {target}"
        )


# hyperparameter list
condition_dict = {
    "ALIGNN_supercon": {  # ALIGNN superconductor
        "coord_lr": 0.002,
        "lattice_lr": 2e-04,
        "atom_lr": 2e-05,
        "copy_mutation": "copy_mutation___steps_cm-50-100-150__mutation_noise-0.00007__top_ratio-0.31__group_C_use_rate-0.02__atom_dist_init-True",
        "ALIGNN_update_step": calculate_update_steps(num_steps=200, num_g_upd=12),
        "model_name": "ALIGNN",
    },
    "Crystalformer_wide_bg": {  # Crystalformer wide band gap, Table 3.
        "coord_lr": 0.005,
        "lattice_lr": 0.002,
        "atom_lr": 0.007,
        "copy_mutation": "copy_mutation___steps_cm-100-150__mutation_noise-0.00007__top_ratio-0.1__group_C_use_rate-0.05__atom_dist_init-True",
        "ALIGNN_update_step": [],
        "model_name": "Crystalformer",
    },
    "ALIGNN_wide_bg": {  # ALIGNN wide band gap, Table 3.
        "lattice_lr": 7e-04,
        "coord_lr": 7e-04,
        "atom_lr": 3e-04,
        "copy_mutation": "copy_mutation___steps_cm-50-100-150__mutation_noise-0.00001__top_ratio-0.3__group_C_use_rate-0.2__atom_dist_init-True",
        "ALIGNN_update_step": calculate_update_steps(num_steps=200, num_g_upd=46),
        "model_name": "ALIGNN",
    },
    "ALIGNN_spg1": {  # ALIGNN space group, Amm2条件付きのときによかった設定
        "lattice_lr": 0.008,
        "coord_lr": 0.0002,
        "atom_lr": 0.0002,
        "limit_coords_displacement": None,
        "copy_mutation": "copy_mutation___steps_cm-None",
        "ALIGNN_update_step": calculate_update_steps(num_steps=200, num_g_upd=200),
        "model_name": "ALIGNN",
    },
    "ALIGNN_spg2": {  # ALIGNN space group, Band gap条件付きのときによかった設定
        "atom_lr": 0.00030,
        "lattice_lr": 0.0007,
        "coord_lr": 0.00002,
        "copy_mutation": "copy_mutation___steps_cm-None",
        "ALIGNN_update_step": calculate_update_steps(num_steps=200, num_g_upd=46),
        "limit_coords_displacement": 0.04,
    },
    "ALIGNN_spg3": {  # ALIGNN space group, Band gap条件付きのときによかった設定でlimit_coords_displacementなし
        "atom_lr": 0.00030,
        "lattice_lr": 0.0007,
        "coord_lr": 0.00002,
        "copy_mutation": "copy_mutation___steps_cm-None",
        "ALIGNN_update_step": calculate_update_steps(num_steps=200, num_g_upd=46),
        "limit_coords_displacement": None,
    },
    "ALIGNN_spg_gap": {  # ALIGNN space group, Table 3.
        "lattice_lr": 7e-04,
        "coord_lr": 7e-04,
        "atom_lr": 3e-04,
        "copy_mutation": "copy_mutation___steps_cm-50-100-150__mutation_noise-0.00001__top_ratio-0.3__group_C_use_rate-0.2__atom_dist_init-True",
        "ALIGNN_update_step": calculate_update_steps(num_steps=200, num_g_upd=46),
        "model_name": "ALIGNN",
    },
    "Crystalformer_perov": {  # random perovskite Crystalformer, Table 4.
        "coord_lr": 0.005,
        "lattice_lr": 0.002,
        "atom_lr": 0.007,
        "copy_mutation": "copy_mutation___steps_cm-100-150__mutation_noise-0.00007__top_ratio-0.1__group_C_use_rate-0.05__atom_dist_init-True",
        "ALIGNN_update_step": calculate_update_steps(num_steps=200, num_g_upd=41),
        "model_name": "Crystalformer",
    },
    "ALIGNN_perov": {  # random perovskite ALIGNN, Table 4.
        "coord_lr": 0.0002,
        "lattice_lr": 9.0,
        "atom_lr": 5e-05,
        "copy_mutation": "copy_mutation___steps_cm-50-100-150__mutation_noise-0.00001__top_ratio-0.87__group_C_use_rate-0.59__atom_dist_init-True",
        "ALIGNN_update_step": calculate_update_steps(num_steps=200, num_g_upd=41),
        "model_name": "ALIGNN",
    },
    "Crystalformer_perov3x3x3": {  # Crystalfomer perov 3x3x3, Table 5.
        "coord_lr": 0.003,
        "lattice_lr": 0.002,
        "atom_lr": 0.00006,
        "copy_mutation": "copy_mutation___steps_cm-50-100-150__mutation_noise-0.00001__top_ratio-0.3__group_C_use_rate-0.2__atom_dist_init-True",
        "ALIGNN_update_step": calculate_update_steps(num_steps=200, num_g_upd=41),
        "model_name": "Crystalformer",
    },
    "ALIGNN_perov3x3x3": {  # ALIGNN perov 3x3x3, Table 5.
        "coord_lr": 0.000667,
        "lattice_lr": 0.01,
        "atom_lr": 0.00004,
        "copy_mutation": "copy_mutation___steps_cm-50-100-150__mutation_noise-0.000003__top_ratio-0.3__group_C_use_rate-0.2__atom_dist_init-True",
        "ALIGNN_update_step": calculate_update_steps(num_steps=200, num_g_upd=41),
        "model_name": "ALIGNN",
    },
    "ALIGNN_bg_e_hull": {  # ALIGNN , BG and E_hull. Table 6.
        "coord_lr": 7e-04,
        "lattice_lr": 7e-04,
        "atom_lr": 3e-04,
        "limit_coords_displacement": None,
        "copy_mutation": "copy_mutation___steps_cm-120__mutation_noise-0.000003__top_ratio-0.48__group_C_use_rate-0.02__atom_dist_init-True",
        "ALIGNN_update_step": calculate_update_steps(num_steps=200, num_g_upd=46),
        "model_name": "ALIGNN",
    },
    "ALIGNN_BM_e_form": {  # ALIGNN , Bluk Modulus and E_f. Table 7.
        "coord_lr": 3e-04,
        "lattice_lr": 0.001,
        "atom_lr": 2e-05,
        "copy_mutation": "copy_mutation___steps_cm-50-100__mutation_noise-0.000003__top_ratio-0.3__group_C_use_rate-0.1__atom_dist_init-True",
        "ALIGNN_update_step": calculate_update_steps(num_steps=200, num_g_upd=46),
        "model_name": "ALIGNN",
    },
}


def float_or_max(value):
    if value == "maximization":
        return value
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Value must be a float or 'maximization', but got '{value}'"
        )


# for condition_dict in condition_dict_list[1:]:

parser = argparse.ArgumentParser(
    description="Structure optimization script",
    allow_abbrev=False,
)

parser.add_argument(
    "--task",
    type=str,
    choices=[
        "bg_and_eform",
        "eform_only",
        "bg_and_eform2",
        "bg_and_eform3",
        "tc_and_efrom",
        "bg_and_ehull",
        "B_and_eform",
        "superhard",
    ],
    default="bg_and_eform",
    help=(
        "Task to perform: bg_and_eform, eform_only, tc_and_efrom, bg_and_ehull, "
        "B_and_eform"
    ),
)
parser.add_argument(
    "--model_name",
    type=str,
    choices=["ALIGNN", "Crystalformer"],
    required=True,
    help="Path to the pretrained model",
)

parser.add_argument(
    "--target_val",
    type=float_or_max,
    default=None,
    help="Main target value such as band gap, Tc or bulk modulus(float) or 'maximization'.",
)

parser.add_argument(
    "--training_margin",
    type=float,
    default=0.04,
    help="training margin for main target such as bandgap, Tc or bulk modulus",
)
parser.add_argument(
    "--eval_margin",
    type=float,
    default=2.0,
    help="evaluation margin for main target such as bandgap, Tc or bulk modulus",
)

parser.add_argument(
    "--e_form_coef",
    type=float,
    default=1.0,
    help="loss coefficient for formation energy or E_hull",
)

parser.add_argument(
    "--e_form_criteria",
    type=float,
    default=1.0,
    help="criteria for formation energy or E_hull",
)

parser.add_argument(
    "--abc_min",
    type=float,
    default=2.0,
    help="minimum lattice length clamp for abc optimization",
)
parser.add_argument(
    "--abc_max",
    type=float,
    default=10.0,
    help="maximum lattice length clamp for abc optimization",
)

parser.add_argument(
    "--structure",
    type=str,
    default=None,
    help="structure: None, perovskite or perovskite3x3x3  (default: None)",
)

parser.add_argument(
    "--b_size", type=int, default=256, help="batch size for crystal optimization"
)

parser.add_argument(
    "--num_candidate",
    type=int,
    default=256,
    help="total number of candidates for crystal optimization",
)
parser.add_argument(
    "--num_workers",
    type=int,
    default=4,
    help="number of DataLoader workers (default: 4)",
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=200,
    help="total number of optimization steps",
)
parser.add_argument(
    "--max_ox_patterns",
    type=int,
    default=100,
    help="maximum oxidation patterns (mapped to internal max_ox_states, default: 100)",
)
parser.add_argument(
    "--ox_solution_mode",
    type=str,
    choices=["strict_unique", "up_to_k"],
    default="strict_unique",
    help=(
        "oxidation-state sampling mode. Both modes use legacy count semantics: "
        "sample up to max_ox_patterns unique neutral assignments and pad the rest "
        "with dummy entries. 'up_to_k' is kept as a compatibility alias."
    ),
)

parser.add_argument(
    "--space_group",
    type=int,
    default=-1,
    help="space group No. for crystal optimization (default: -1 means no space group constraint)",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda",
    help="device for computation (cuda or cpu)",
)
parser.add_argument(
    "--optimizer",
    type=str,
    default="gradient",
    choices=["gradient", "sa", "grad_sa_hybrid"],
    help="optimization method: gradient, sa, or grad_sa_hybrid (grad-only prefix + per-step grad->sa suffix)",
)
parser.add_argument(
    "--sa_start_step",
    type=int,
    default=None,
    help="global step at which SA starts for --optimizer grad_sa_hybrid (default: num_steps//2)",
)
parser.add_argument(
    "--grad_sa_hybrid_grad_steps",
    type=int,
    default=None,
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--seed",
    type=int,
    default=None,
    help="random seed for reproducibility (default: None)",
)
parser.add_argument(
    "--config",
    type=str,
    default=None,
    help="path to config file (yaml/json) for multi-predictor and loss DSL",
)
parser.add_argument(
    "--energy_coef",
    type=float,
    default=0.0,
    help="MatterSim energy loss coefficient (default: 0.0, disabled)",
)
parser.add_argument(
    "--energy_every",
    type=int,
    default=1,
    help="Compute energy loss every N steps (default: 1)",
)
parser.add_argument(
    "--use_element_count_l1_loss",
    action="store_true",
    help="enable per-crystal element-count L1 regularization",
)
parser.add_argument(
    "--element_count_l1_weight",
    type=float,
    default=1.0,
    help="weight for per-crystal element-count L1 regularization (default: 1.0)",
)
parser.add_argument(
    "--mattersim_model",
    type=str,
    default=None,
    help="Path to MatterSim checkpoint (optional, defaults to submodule weights)",
)
parser.add_argument(
    "--bg_target",
    type=float,
    default=None,
    help="bandgap target for custom loss (default: None)",
)
parser.add_argument(
    "--magmom_tol",
    type=float,
    default=0.05,
    help="tolerance for magmom integerization loss (default: 0.05)",
)
parser.add_argument(
    "--magmom_eps",
    type=float,
    default=0.01,
    help="minimum magmom value before applying sign penalty (default: 0.01)",
)
parser.add_argument(
    "--w_bg",
    type=float,
    default=1.0,
    help="weight for bandgap term in custom loss (default: 1.0)",
)
parser.add_argument(
    "--w_mag",
    type=float,
    default=1.0,
    help="weight for magmom term in custom loss (default: 1.0)",
)
parser.add_argument(
    "--w_mag_int",
    type=float,
    default=None,
    help="weight for magmom integerization term (default: --w_mag)",
)
parser.add_argument(
    "--w_mag_sign",
    type=float,
    default=None,
    help="weight for magmom sign term (default: --w_mag)",
)
parser.add_argument(
    "--w_E",
    type=float,
    default=None,
    help="alias for --energy_coef (if set and --energy_coef not provided)",
)
parser.add_argument(
    "--magmom_checkpoint",
    type=str,
    default="jv_magmom_alignn",
    help="ALIGNN checkpoint name for magmom predictor",
)
parser.add_argument(
    "--use_magmom_loss",
    action="store_true",
    help="enable magmom integerization loss in custom loss mode",
)
parser.add_argument(
    "--poscar_magmom_min",
    type=float,
    default=None,
    help="only save POSCARs when predicted magmom >= this value (default: None)",
)
parser.add_argument(
    "--initial_structures_path",
    type=str,
    default=None,
    help="path to initial structures (directory of .vasp files or a .pkl file)",
)
parser.add_argument(
    "--init_dir_path",
    dest="initial_structures_path",
    type=str,
    default=None,
    help="alias of --initial_structures_path",
)
parser.add_argument(
    "--mask_dict_path",
    type=str,
    default=None,
    help="path to oxidation mask dict .npz file",
)
parser.add_argument(
    "--mask_path",
    dest="mask_dict_path",
    type=str,
    default=None,
    help="alias of --mask_dict_path",
)
parser.add_argument(
    "--G_coef",
    type=float,
    default=1.0,
    help="loss coefficient for shear modulus G (maximize)",
)
parser.add_argument(
    "--B_coef",
    type=float,
    default=1.0,
    help="loss coefficient for bulk modulus B (maximize)",
)
parser.add_argument(
    "--k_min",
    type=float,
    default=1.0,
    help="lower bound for k = G/B penalty",
)
parser.add_argument(
    "--k_max",
    type=float,
    default=1.5,
    help="upper bound for k = G/B penalty",
)
parser.add_argument(
    "--k_coef",
    type=float,
    default=1.0,
    help="loss coefficient for k = G/B penalty",
)
parser.add_argument(
    "--alignn_weights_dir",
    type=str,
    default=None,
    help="directory containing ALIGNN pretrained zip files",
)
parser.add_argument(
    "--sa_t0",
    type=float,
    default=1.0,
    help="SA initial temperature",
)
parser.add_argument(
    "--sa_tmin",
    type=float,
    default=1e-3,
    help="SA minimum temperature",
)
parser.add_argument(
    "--sa_trials",
    type=int,
    default=1,
    help="SA trials per temperature step",
)
parser.add_argument(
    "--sa_coords_sigma",
    type=float,
    default=1e-3,
    help="SA proposal sigma for frac coords",
)
parser.add_argument(
    "--sa_lattice_sigma",
    type=float,
    default=1e-3,
    help="SA proposal sigma for lattice",
)
parser.add_argument(
    "--sa_angle_sigma",
    type=float,
    default=1e-3,
    help="SA proposal sigma for lattice angles",
)
parser.add_argument(
    "--sa_cache_size",
    type=int,
    default=0,
    help="SA evaluation cache size (0 disables)",
)
parser.add_argument(
    "--sa_per_sample_accept",
    dest="sa_per_sample_accept",
    action="store_true",
    default=True,
    help="accept/reject SA proposals per sample instead of per mini-batch",
)
parser.add_argument(
    "--no_sa_per_sample_accept",
    "--no-sa_per_sample_accept",
    "--no-sa-per-sample-accept",
    dest="sa_per_sample_accept",
    action="store_false",
    help="accept/reject SA proposals per mini-batch",
)
parser.add_argument(
    "--exp_name",
    type=str,
    default=None,
    help="override experiment name for results directory (optional)",
)
parser.add_argument(
    "--allow_lr_override",
    action="store_true",
    help="allow overriding task preset learning rates with --lattice_lr/--coords_lr/--atom_lr",
)
parser.add_argument(
    "--lattice_lr",
    type=float,
    default=None,
    help="override lattice learning rate (requires --allow_lr_override)",
)
parser.add_argument(
    "--coords_lr",
    type=float,
    default=None,
    help="override coords learning rate; mapped to internal coord_lr (requires --allow_lr_override)",
)
parser.add_argument(
    "--atom_lr",
    type=float,
    default=None,
    help="override atom learning rate (requires --allow_lr_override)",
)
parser.add_argument(
    "--copy_mutation_override",
    type=str,
    default=None,
    help="override task preset copy_mutation string",
)

args = parser.parse_args()
model_name = args.model_name
task = args.task
target_val = args.target_val
training_margin = args.training_margin
eval_margin = args.eval_margin
e_from_coef = args.e_form_coef
e_from_criteria = args.e_form_criteria
abc_min = args.abc_min
abc_max = args.abc_max
structure = args.structure
b_size = args.b_size
num_candidate = args.num_candidate
num_workers = args.num_workers
num_steps = args.num_steps
max_ox_patterns = args.max_ox_patterns
ox_solution_mode = args.ox_solution_mode
target_space_group_no = args.space_group
config_path = args.config
config = load_config(config_path) if config_path else None
device = args.device
optimizer = args.optimizer
sa_start_step_arg = args.sa_start_step
legacy_sa_start_step_arg = args.grad_sa_hybrid_grad_steps
seed = args.seed
energy_coef = args.energy_coef
energy_every = args.energy_every
mattersim_model = args.mattersim_model
use_element_count_l1_loss = args.use_element_count_l1_loss
element_count_l1_weight = args.element_count_l1_weight
G_coef = args.G_coef
B_coef = args.B_coef
k_min = args.k_min
k_max = args.k_max
k_coef = args.k_coef
alignn_weights_dir = args.alignn_weights_dir
exp_name_override = args.exp_name
allow_lr_override = args.allow_lr_override
override_lattice_lr = args.lattice_lr
override_coords_lr = args.coords_lr
override_atom_lr = args.atom_lr
copy_mutation_override = args.copy_mutation_override
bg_target = args.bg_target
magmom_tol = args.magmom_tol
magmom_eps = args.magmom_eps
w_bg = args.w_bg
w_mag = args.w_mag
w_mag_int = args.w_mag_int
w_mag_sign = args.w_mag_sign
w_E = args.w_E
magmom_checkpoint = args.magmom_checkpoint
use_magmom_loss = args.use_magmom_loss
poscar_magmom_min = args.poscar_magmom_min

override_values = [override_lattice_lr, override_coords_lr, override_atom_lr]
override_requested = any(v is not None for v in override_values)
if override_requested and not allow_lr_override:
    parser.error(
        "--allow_lr_override is required when using --lattice_lr/--coords_lr/--atom_lr"
    )
if allow_lr_override and not all(v is not None for v in override_values):
    parser.error(
        "When --allow_lr_override is set, all of --lattice_lr, --coords_lr, --atom_lr must be provided."
    )
if max_ox_patterns < 1:
    parser.error("--max_ox_patterns must be a positive integer.")
if abc_min <= 0.0:
    parser.error("--abc_min must be > 0.")
if abc_max < abc_min:
    parser.error("--abc_max must be >= --abc_min.")
if args.sa_t0 <= 0.0:
    parser.error("--sa_t0 must be > 0.")
if args.sa_tmin <= 0.0:
    parser.error("--sa_tmin must be > 0.")
if args.sa_tmin > args.sa_t0:
    parser.error("--sa_tmin must be <= --sa_t0.")
if args.sa_trials < 1:
    parser.error("--sa_trials must be >= 1.")
if args.sa_cache_size < 0:
    parser.error("--sa_cache_size must be >= 0.")
for sigma_name, sigma_val in (
    ("--sa_coords_sigma", args.sa_coords_sigma),
    ("--sa_lattice_sigma", args.sa_lattice_sigma),
    ("--sa_angle_sigma", args.sa_angle_sigma),
):
    if sigma_val < 0.0:
        parser.error(f"{sigma_name} must be >= 0.")

if sa_start_step_arg is not None and legacy_sa_start_step_arg is not None:
    parser.error("Specify only one of --sa_start_step or --grad_sa_hybrid_grad_steps.")
if legacy_sa_start_step_arg is not None:
    sa_start_step_arg = legacy_sa_start_step_arg

if optimizer == "grad_sa_hybrid":
    if sa_start_step_arg is None:
        sa_start_step = num_steps // 2
    else:
        sa_start_step = sa_start_step_arg
    if sa_start_step < 0:
        parser.error("--sa_start_step must be >= 0.")
    if sa_start_step >= num_steps:
        parser.error(
            "--sa_start_step must be < --num_steps for optimizer=grad_sa_hybrid."
        )
    grad_sa_hybrid_sa_steps = num_steps - sa_start_step
    grad_sa_hybrid_config = {
        "grad_only_steps": int(sa_start_step),
        "sa_start_step": int(sa_start_step),
        "grad_sa_hybrid_grad_steps": int(sa_start_step),
        "grad_sa_hybrid_sa_steps": int(grad_sa_hybrid_sa_steps),
    }
else:
    if sa_start_step_arg is not None:
        parser.error(
            "--sa_start_step/--grad_sa_hybrid_grad_steps requires "
            "--optimizer grad_sa_hybrid."
        )
    grad_sa_hybrid_config = None

if task == LEGACY_EFORM_ONLY_TASK:
    raise ValueError(
        f"--task {LEGACY_EFORM_ONLY_TASK} is removed. Use --task {EFORM_ONLY_TASK}."
    )
validate_eform_only_args(
    task=task,
    target_space_group_no=target_space_group_no,
    target_val=target_val,
)
if task == EFORM_ONLY_TASK:
    if model_name != "ALIGNN":
        raise ValueError("--task eform_only supports only --model_name ALIGNN.")
    if config is not None:
        raise ValueError("--task eform_only does not allow --config.")
    config = build_eform_only_alignn_config(alignn_weights_dir=alignn_weights_dir)
    print(
        "[info] eform_only mode enabled: using eform predictor only (no bandgap "
        "predictor)."
    )

_patch_max_ox_states(max_ox_patterns=max_ox_patterns)
print(f"[info] max_ox_patterns={max_ox_patterns} (applied as max_ox_states)")

sa_config = {
    "t0": args.sa_t0,
    "tmin": args.sa_tmin,
    "trials_per_step": args.sa_trials,
    "coords_sigma": args.sa_coords_sigma,
    "lattice_sigma": args.sa_lattice_sigma,
    "angle_sigma": args.sa_angle_sigma,
    "seed": seed,
    "cache_size": args.sa_cache_size,
    "per_sample_accept": args.sa_per_sample_accept,
}

if config and "energy" in config:
    energy_cfg = config["energy"] or {}
    if "--energy_coef" not in sys.argv and "coef" in energy_cfg:
        energy_coef = float(energy_cfg["coef"])
    if "--energy_every" not in sys.argv and "every" in energy_cfg:
        energy_every = int(energy_cfg["every"])
    if "--mattersim_model" not in sys.argv and "model" in energy_cfg:
        mattersim_model = energy_cfg["model"]
if config and "element_count_l1" in config:
    elem_cfg = config["element_count_l1"] or {}
    if "--use_element_count_l1_loss" not in sys.argv and "enabled" in elem_cfg:
        use_element_count_l1_loss = bool(elem_cfg["enabled"])
    if "--element_count_l1_weight" not in sys.argv and "weight" in elem_cfg:
        element_count_l1_weight = float(elem_cfg["weight"])
if w_E is not None and "--energy_coef" not in sys.argv:
    energy_coef = float(w_E)
initial_structures_path = args.initial_structures_path
mask_dict_path = args.mask_dict_path
element_constraints = get_element_constraints(config)
forbidden_elements = element_constraints["forbidden_elements"]
mask_dict_path = select_mask_dict_path(mask_dict_path, element_constraints)


def _resolve_path_arg(value: str, arg_name: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.exists():
        if "intermidiat" in value:
            alt_path = (
                Path(value.replace("intermidiat", "intermidiate"))
                .expanduser()
                .resolve()
            )
            if alt_path.exists():
                return alt_path
        raise ValueError(f"{arg_name} does not exist: {path}")
    return path


def _infer_space_group_no_from_info_pkl(path: Optional[Path]) -> Optional[int]:
    if path is None or not path.is_file() or path.suffix != ".pkl":
        return None

    name = path.name
    prefix = "initial_MP_based_"
    suffix = "_info.pkl"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        return None

    spg_key = name[len(prefix) : -len(suffix)]
    mapping = {
        "Amm2": 38,
        "P2m": 10,
        "Fbar43m": 216,
        "R3m": 160,
        "P6_3m": 176,
        "P321": 150,
        "P4mmm": 123,
    }
    if spg_key not in mapping:
        raise ValueError(
            "Unsupported space-group info pkl name. "
            f"Expected one of {sorted(mapping)}, got: {path.name}"
        )
    return mapping[spg_key]


if task == "superhard":
    if initial_structures_path is None:
        raise ValueError(
            "--initial_structures_path is required for task=superhard. "
            "Model weights and large initial-structure datasets are not bundled "
            "with the public repository."
        )
    if mask_dict_path is None:
        raise ValueError(
            "--mask_dict_path is required for task=superhard. "
            "Provide a generated oxidation-mask .npz file."
        )
    if "--e_form_coef" not in sys.argv:
        e_from_coef = 0.0
    if "--energy_coef" not in sys.argv:
        energy_coef = 1.0
    if config is None:
        k_eps = 1e-8
        k_expr = (
            f"relu({k_min} - safe_div(pred.G, pred.B, {k_eps}))**2 + "
            f"relu(safe_div(pred.G, pred.B, {k_eps}) - {k_max})**2"
        )
        config = {
            "predictors": [
                {
                    "name": "G",
                    "type": "alignn",
                    "checkpoint": "jv_shear_modulus_gv_alignn",
                    "params": {"weights_dir": alignn_weights_dir},
                },
                {
                    "name": "B",
                    "type": "alignn",
                    "checkpoint": "jv_bulk_modulus_kv_alignn",
                    "params": {"weights_dir": alignn_weights_dir},
                },
            ],
            "loss": {
                "terms": [
                    {"name": "loss_G", "expr": "-pred.G", "weight": G_coef},
                    {"name": "loss_B", "expr": "-pred.B", "weight": B_coef},
                    {"name": "loss_k", "expr": k_expr, "weight": k_coef},
                ]
            },
            "output": {
                "primary_metrics": {
                    "bandgap_onehot": "pred.G",
                    "ef_onehot": "pred.B",
                }
            },
        }

if (config is None or not config.get("predictors")) and (
    bg_target is not None or use_magmom_loss
):
    existing_config = dict(config or {})
    if any(key in existing_config for key in ("loss", "output")):
        raise ValueError(
            "A config with loss/output but without predictors is not supported. "
            "Provide a complete predictor config, or use only top-level runtime "
            "sections such as element_constraints, energy, and element_count_l1."
        )
    if bg_target is None:
        bg_target = 0.0
    if "--e_form_coef" not in sys.argv:
        e_from_coef = 0.0
    if w_mag_int is None:
        w_mag_int = w_mag
    if w_mag_sign is None:
        w_mag_sign = w_mag
    magmom_pos = "max(pred.magmom, 0.0)"
    magmom_dist = f"abs({magmom_pos} - round({magmom_pos}))"
    magmom_loss_int = f"relu({magmom_dist} - {magmom_tol})"
    magmom_loss_sign = f"relu({magmom_eps} - pred.magmom)"
    config = {
        "predictors": [
            {
                "name": "bandgap",
                "type": "alignn",
                "checkpoint": "mp_gappbe_alignnn",
                "params": {"weights_dir": alignn_weights_dir},
            },
            {
                "name": "eform",
                "type": "alignn",
                "checkpoint": "mp_e_form_alignnn",
                "params": {"weights_dir": alignn_weights_dir},
            },
            {
                "name": "magmom",
                "type": "alignn",
                "checkpoint": magmom_checkpoint,
                "params": {"weights_dir": alignn_weights_dir},
            },
        ],
        "loss": {
            "terms": [
                {
                    "name": "loss_bg",
                    "expr": f"mae(pred.bandgap, {bg_target})",
                    "weight": w_bg,
                },
                {
                    "name": "loss_magmom_int",
                    "expr": magmom_loss_int,
                    "weight": w_mag_int,
                },
                {
                    "name": "loss_magmom_sign",
                    "expr": magmom_loss_sign,
                    "weight": w_mag_sign,
                },
            ]
        },
        "output": {
            "primary_metrics": {
                "bandgap_onehot": "pred.bandgap",
                "ef_onehot": "pred.eform",
            }
        },
    }
    for key, value in existing_config.items():
        if key not in {"predictors", "loss", "output"}:
            config[key] = value

initial_info_pkl_path: Optional[Path] = None
if initial_structures_path is not None:
    initial_structures_path = _resolve_path_arg(
        initial_structures_path, "--initial_structures_path"
    )
    if not (
        initial_structures_path.is_dir()
        or (
            initial_structures_path.is_file()
            and initial_structures_path.suffix == ".pkl"
        )
    ):
        raise ValueError(
            "--initial_structures_path must be a directory or a .pkl file: "
            f"{initial_structures_path}"
        )
    inferred_space_group_no = _infer_space_group_no_from_info_pkl(initial_structures_path)
    if inferred_space_group_no is not None:
        initial_info_pkl_path = initial_structures_path
        if target_space_group_no > 0 and target_space_group_no != inferred_space_group_no:
            raise ValueError(
                "--space_group does not match --initial_structures_path info pkl: "
                f"{target_space_group_no} vs {inferred_space_group_no}"
            )
        if target_space_group_no <= 0:
            target_space_group_no = inferred_space_group_no
            print(
                "[info] Inferred --space_group from info pkl: "
                f"{target_space_group_no} ({initial_structures_path.name})"
            )
    initial_structures_path = str(initial_structures_path)

if mask_dict_path is not None:
    mask_dict_path = _resolve_path_arg(mask_dict_path, "--mask_dict_path")
    if not (mask_dict_path.is_file() and mask_dict_path.suffix == ".npz"):
        raise ValueError("--mask_dict_path must be a .npz file: " f"{mask_dict_path}")
    if forbidden_elements:
        generated_mask_dir = (
            Path(get_master_dir()) / "data" / "intermidiate" / "generated_masks"
        )
        mask_dict_path = write_forbidden_element_mask(
            base_mask_path=mask_dict_path,
            forbidden_elements=forbidden_elements,
            out_dir=generated_mask_dir,
        )
        print(
            "[info] element_constraints.forbidden_elements applied: "
            f"{forbidden_elements} -> {mask_dict_path}"
        )
    mask_dict_path = str(mask_dict_path)

# parameter settings from arguments
## loss settings for the main target property

if target_val is None:
    bg_loss_mode = "no_loss"
elif isinstance(target_val, float):
    bg_loss_mode = "band"
elif target_val == "maximization":
    bg_loss_mode = "maximization"


## dataset, task and hyper parameters settings
initial_dataset = "megnet"
if model_name == "ALIGNN" and task == "tc_and_efrom":
    condition_dict = condition_dict["ALIGNN_supercon"]
    initial_dataset = "jarvis_supercon"
elif (
    model_name == "ALIGNN"
    and task == "bg_and_eform"
    and target_space_group_no > 0
    and target_val is not None
):
    condition_dict = condition_dict["ALIGNN_spg_gap"]
    print("Debug print: ALIGNN_spg_gap condition is selected.")
elif (
    model_name == "ALIGNN"
    and task == "eform_only"
    and target_space_group_no > 0
    and target_val is None
):
    condition_dict = condition_dict["ALIGNN_spg1"]
    print("Debug print: ALIGNN_spg1 condition is selected for eform_only.")
elif (
    model_name == "ALIGNN"
    and task == "bg_and_eform2"
    and target_space_group_no > 0
    and target_val is None
):
    condition_dict = condition_dict["ALIGNN_spg2"]
    # task = 'bg_and_eform'
    print("Debug print: ALIGNN_spg2 condition is selected.")
elif (
    model_name == "ALIGNN"
    and task == "bg_and_eform3"
    and target_space_group_no > 0
    and target_val is None
):
    condition_dict = condition_dict["ALIGNN_spg3"]
    # task = 'bg_and_eform'
    print("Debug print: ALIGNN_spg3 condition is selected.")
elif model_name == "ALIGNN" and task == "bg_and_eform" and structure is None:
    condition_dict = condition_dict["ALIGNN_wide_bg"]
elif model_name == "Crystalformer" and task == "bg_and_eform" and structure is None:
    condition_dict = condition_dict["Crystalformer_wide_bg"]
elif model_name == "ALIGNN" and task == "bg_and_eform" and structure == "perovskite":
    condition_dict = condition_dict["ALIGNN_perov"]
elif (
    model_name == "Crystalformer"
    and task == "bg_and_eform"
    and structure == "perovskite"
):
    condition_dict = condition_dict["Crystalformer_perov"]
elif (
    model_name == "ALIGNN"
    and task == "bg_and_eform"
    and structure == "perovskite3x3x3"
):
    condition_dict = condition_dict["ALIGNN_perov3x3x3"]
elif (
    model_name == "Crystalformer"
    and task == "bg_and_eform"
    and structure == "perovskite3x3x3"
):
    condition_dict = condition_dict["Crystalformer_perov3x3x3"]
elif (
    model_name == "ALIGNN" and task == "bg_and_ehull" and structure == "perovskite3x3x3"
):
    condition_dict = condition_dict["ALIGNN_perov3x3x3"]
elif (
    model_name == "Crystalformer"
    and task == "bg_and_ehull"
    and structure == "perovskite3x3x3"
):
    condition_dict = condition_dict["Crystalformer_perov3x3x3"]
elif model_name == "ALIGNN" and task == "bg_and_ehull":
    condition_dict = condition_dict["ALIGNN_bg_e_hull"]
elif model_name == "ALIGNN" and task == "B_and_eform":
    condition_dict = condition_dict["ALIGNN_BM_e_form"]
elif model_name == "ALIGNN" and task == "superhard":
    condition_dict = condition_dict["ALIGNN_BM_e_form"]
else:
    raise NotImplementedError(
        f"Task {task} is not implemented for model {model_name} with structure {structure}"
    )

if allow_lr_override:
    condition_dict["lattice_lr"] = override_lattice_lr
    condition_dict["coord_lr"] = override_coords_lr
    condition_dict["atom_lr"] = override_atom_lr
    print(
        "[info] Learning-rate override enabled: "
        f"lattice_lr={condition_dict['lattice_lr']}, "
        f"coord_lr={condition_dict['coord_lr']}, "
        f"atom_lr={condition_dict['atom_lr']}"
    )
if copy_mutation_override is not None:
    condition_dict["copy_mutation"] = copy_mutation_override
    print(
        "[info] copy_mutation override enabled: " f"{condition_dict['copy_mutation']}"
    )


## structure settings
if structure == "perovskite3x3x3":
    structure = "perovskite"
    perov_size = "3x3x3"
    limit_coords_displacement = 0.05
elif structure == "perovskite":
    perov_size = None
    limit_coords_displacement = 0.15
elif target_space_group_no > 0:
    perov_size = None
    limit_coords_displacement = condition_dict.get("limit_coords_displacement")
else:
    perov_size = None
    limit_coords_displacement = None

if num_steps != 200 and condition_dict.get("ALIGNN_update_step"):
    num_g_upd = len(condition_dict["ALIGNN_update_step"])
    if num_g_upd > 0:
        condition_dict["ALIGNN_update_step"] = calculate_update_steps(
            num_steps=num_steps, num_g_upd=num_g_upd
        )


if target_space_group_no > 0:
    if target_space_group_no == 38:
        sg_wyckoff = Amm2Wyckoff.build()
        target_space_group = "Amm2"
    elif target_space_group_no == 150:
        sg_wyckoff = P321Wyckoff.build()
        target_space_group = "P321"
        spg_number = 150
    elif target_space_group_no == 216:
        sg_wyckoff = Fbar43mWyckoff.build()
        target_space_group = "Fbar43m"
        spg_number = 216
    elif target_space_group_no == 160:
        sg_wyckoff = R3mWyckoff.build()
        target_space_group = "R3m"
    elif target_space_group_no == 176:
        sg_wyckoff = P63mWyckoff.build()
        target_space_group = "P6_3/m"
    elif target_space_group_no == 123:
        sg_wyckoff = P4mmmWyckoff.build()
        target_space_group = "P4/mmm"
    elif target_space_group_no == 10:
        sg_wyckoff = P2mWyckoff.build()
        target_space_group = "P2m"
    else:
        raise NotImplementedError(
            f"Space group No. {target_space_group_no} is not implemented yet."
        )

    wyckoff_expander = WyckoffExpander(
        WYCKOFF_TABLE=sg_wyckoff.wyckoff_table,
        dup_positions=sg_wyckoff.dup_pos,
        wyckoff_site=sg_wyckoff.wyckoff_site,
        spg_number=target_space_group_no,
        spg_name=target_space_group,
        num_equivalent_sites=sg_wyckoff.num_equivalent_sites,
        device=device,
    )
    if initial_info_pkl_path is not None:
        spg_df = pd.read_pickle(initial_info_pkl_path)
    else:
        spg_df = pd.read_pickle(
            os.path.join(
                get_master_dir(),
                f'data/raw_data/initial_MP_based_{target_space_group.replace("/","")}_info.pkl',
            )
        )
    data_list = spg_df.to_dict(orient="records")
    _adjust_input_dataframe_ox_patterns(data_list, max_ox_patterns)
else:
    wyckoff_expander = None
    data_list = None
    target_space_group = None

## experiment name
exp_name = f"{model_name}__{task}__{target_val}"
if target_space_group is not None:
    exp_name += f"__spg-{target_space_group_no}"
if exp_name_override:
    exp_name = exp_name_override

result_dir = main_experiment(
    model_name=args.model_name,  #'Crystalformer' or 'ALIGNN' or 'Both
    exp_name=exp_name,
    initial_dataset=initial_dataset,
    num_steps=num_steps,
    copy_mutation=condition_dict["copy_mutation"],
    bg_margin_for_train=training_margin,  # margin for training
    bg_margin_for_eval=eval_margin,  # margin for evaluation
    num_candidate=num_candidate,  # 1024, # max 4096
    num_batch_crystal=b_size,  # 1024 # max 4096
    target_bandgap=target_val,  # None: all bandgap test.
    crystal_system=structure,  #'perovskite' or None
    perovskite_size=perov_size,  # None or '2x2x2', '3x3x3', '4x4x4'
    limit_coords_displacement=limit_coords_displacement,  # 0.15 for perovskite, None for non-perovskite
    abc_range=(abc_min, abc_max),
    max_ox_states=max_ox_patterns,
    ox_solution_mode=ox_solution_mode,
    atom_lr=condition_dict["atom_lr"],
    lattice_lr=condition_dict["lattice_lr"],
    coords_lr=condition_dict["coord_lr"],
    bg_loss_mode=bg_loss_mode,
    e_form_coef=e_from_coef,
    e_form_criteria=e_from_criteria,
    task=task,
    ALIGNN_update_step=copy.deepcopy(condition_dict["ALIGNN_update_step"]),
    test_mode=False,
    wyckoff_expander=wyckoff_expander,
    input_dataframe=data_list,
    config=config,
    energy_coef=energy_coef,
    energy_every=energy_every,
    mattersim_model=mattersim_model,
    use_element_count_l1_loss=use_element_count_l1_loss,
    element_count_l1_weight=element_count_l1_weight,
    superhard_cfg=(
        {
            "G_coef": G_coef,
            "B_coef": B_coef,
            "k_coef": k_coef,
            "k_min": k_min,
            "k_max": k_max,
            "k_eps": 1e-8,
        }
        if task == "superhard"
        else None
    ),
    num_workers=num_workers,
    device=device,
    initial_structures_path=initial_structures_path,
    mask_dict_path=mask_dict_path,
    alignn_weights_dir=alignn_weights_dir,
    optimizer=optimizer,
    seed=seed,
    sa_config=sa_config if optimizer in {"sa", "grad_sa_hybrid"} else None,
    grad_sa_hybrid_config=grad_sa_hybrid_config,
    poscar_magmom_min=poscar_magmom_min,
)
