import argparse
import copy

from src.main_experiment import main_experiment
from src.smoacs_opt_eval.alignn_graph_update import calculate_update_steps

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
    "Crystalformer_perov": {  # random perovskite Crystalformer, Table 4.
        "coord_lr": 0.005,
        "lattice_lr": 0.002,
        "atom_lr": 0.007,
        "copy_mutation": "copy_mutation___steps_cm-100-150__mutation_noise-0.00007__top_ratio-0.1__group_C_use_rate-0.05__atom_dist_init-True",
        "ALIGNN_update_step": [],
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
        "ALIGNN_update_step": [],
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

parser = argparse.ArgumentParser(description="Structure optimization script")

parser.add_argument(
    "--task",
    type=str,
    choices=["bg_and_eform", "tc_and_efrom", "bg_and_ehull", "B_and_eform"],
    default="bg_and_eform",
    help="Task to perform: bg_and_eform, tc_and_efrom, bg_and_ehull, B_and_eform",
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

args = parser.parse_args()
model_name = args.model_name
task = args.task
target_val = args.target_val
training_margin = args.training_margin
eval_margin = args.eval_margin
e_from_coef = args.e_form_coef
e_from_criteria = args.e_form_criteria
structure = args.structure
b_size = args.b_size
num_candidate = args.num_candidate


# parameter settings from arguments
## loss settings for the main target property
if type(target_val) == float:
    bg_loss_mode = "band"
elif target_val == "maximization":
    bg_loss_mode = "maximization"

## structure settings
if structure == "perovskite3x3x3":
    structure = "perovskite"
    perov_size = "3x3x3"
    limit_coords_displacement = 0.05
elif structure == "perovskite":
    perov_size = None
    limit_coords_displacement = 0.15
else:
    perov_size = None
    limit_coords_displacement = None

## dataset, task and hyper parameters settings
initial_dataset = "megnet"
if model_name == "ALIGNN" and task == "tc_and_efrom":
    condition_dict = condition_dict["ALIGNN_supercon"]
    initial_dataset = "jarvis_supercon"
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
else:
    raise NotImplementedError(
        f"Task {task} is not implemented for model {model_name} with structure {structure}"
    )

## experiment name
exp_name = f"{model_name}__{task}__{target_val}"

result_dir = main_experiment(
    model_name=args.model_name,  #'Crystalformer' or 'ALIGNN' or 'Both
    exp_name=exp_name,
    initial_dataset=initial_dataset,
    num_steps=200,
    copy_mutation=condition_dict["copy_mutation"],
    bg_margin_for_train=training_margin,  # margin for training
    bg_margin_for_eval=eval_margin,  # margin for evaluation
    num_candidate=num_candidate,  # 1024, # max 4096
    num_batch_crystal=b_size,  # 1024 # max 4096
    target_bandgap=target_val,  # None: all bandgap test.
    crystal_system=structure,  #'perovskite' or None
    perovskite_size=perov_size,  # None or '2x2x2', '3x3x3', '4x4x4'
    limit_coords_displacement=limit_coords_displacement,  # 0.15 for perovskite, None for non-perovskite
    atom_lr=condition_dict["atom_lr"],
    lattice_lr=condition_dict["lattice_lr"],
    coords_lr=condition_dict["coord_lr"],
    bg_loss_mode=bg_loss_mode,
    e_form_coef=e_from_coef,
    e_form_criteria=e_from_criteria,
    task=task,
    ALIGNN_update_step=copy.deepcopy(condition_dict["ALIGNN_update_step"]),
    test_mode=False,
)
