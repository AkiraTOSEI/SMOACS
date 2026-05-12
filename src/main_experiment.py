import itertools
import json
import os
import random
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython.display import clear_output, display
from PIL import Image
from tqdm import tqdm

from src.data_loader.input_loader_pipeline import load_initial_data

# visualization and evaluation utilities
from src.evaluation.evaluation import evaluation_and_analysis

# losses and datasets
from src.losses.bandgap_loss import GapLoss
from src.losses.tolerance_loss import ToleranceLoss
from src.models.alignn_loader import Load_Pretrained_ALIGNN
from src.models.crystalformer_loader import Load_Pretrained_Crystalformers
from src.predictors import build_predictors
from src.losses.loss_composer import LossComposer
from src.energy import MatterSimEnergyModule
from src.utils.safe_expression import SafeExpressionEvaluator

# model loading and optimization (assumed to be project-specific modules)
from src.smoacs_opt_eval.alignn_eval import evaluation_for_each_batch_ALIGNN
from src.smoacs_opt_eval.alignn_optimize import optimize_solution_ALIGNN
from src.smoacs_opt_eval.crystalformer_eval import (
    evaluation_for_each_batch_Crystalformer,
)
from src.smoacs_opt_eval.crystalformer_optimize import optimize_solution_Crystalformer
from src.utils.common import get_master_dir
from src.utils.experiment_utils import (
    define_initial_dataset_dir_path,
    remove_files_in_directory,
)
from src.utils.save_optimization_result import save_optimization_results
from src.utils.schedule_utils import set_schedule

from src.spg_wp.wyckoff_expander import WyckoffExpander


def _parse_primary_metrics(config: Dict[str, Any]) -> Optional[Dict[str, str]]:
    output_cfg = config.get("output", {})
    primary = output_cfg.get("primary_metrics")
    if primary is None:
        return None
    if isinstance(primary, dict):
        return {k: _strip_pred_prefix(v) for k, v in primary.items()}
    if isinstance(primary, list):
        mapping: Dict[str, str] = {}
        for item in primary:
            if not isinstance(item, dict):
                raise ValueError("primary_metrics list must contain dict entries")
            for key, value in item.items():
                mapping[key] = _strip_pred_prefix(value)
        return mapping
    raise ValueError("primary_metrics must be dict or list of dicts")


def _strip_pred_prefix(value: str) -> str:
    if isinstance(value, str) and value.startswith("pred."):
        return value.replace("pred.", "", 1)
    return value


def _move_result_payload_to_cpu(value: Any) -> Any:
    """Detach tensors from graph and move result payload recursively to CPU."""
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _move_result_payload_to_cpu(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_result_payload_to_cpu(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_result_payload_to_cpu(v) for v in value)
    if (
        hasattr(value, "to")
        and hasattr(value, "number_of_nodes")
        and hasattr(value, "number_of_edges")
    ):
        return value.to("cpu")
    return value


def prepare_result_directories(exp_name: str):
    """
    Set up the result directories for a given experiment name.
    Returns paths to result directories and files.
    """
    result_dir = f"./results/{exp_name}"
    os.makedirs(result_dir, exist_ok=True)
    npz_path = os.path.join(result_dir, "result.npz")
    poscar_saved_dir = os.path.join(result_dir, "poscar")
    saved_gif_dir = os.path.join(result_dir, "gif_dir")
    tmp_output_dir = os.path.join(result_dir, "tmp_gif_output_dir")
    os.makedirs(poscar_saved_dir, exist_ok=True)
    os.makedirs(saved_gif_dir, exist_ok=True)
    os.makedirs(tmp_output_dir, exist_ok=True)
    history_img_path = os.path.join(result_dir, "history_img.png")

    for saved_dir_path in [poscar_saved_dir, saved_gif_dir, tmp_output_dir]:
        if os.path.exists(saved_dir_path):
            remove_files_in_directory(saved_dir_path)

    return {
        "result_dir": result_dir,
        "npz_path": npz_path,
        "poscar_saved_dir": poscar_saved_dir,
        "saved_gif_dir": saved_gif_dir,
        "tmp_output_dir": tmp_output_dir,
        "history_img_path": history_img_path,
    }
    return gap_loss_func_for_train, gap_loss_func_for_eval, gap_min_eval, gap_max_eval


def prepare_gap_loss_funcs(
    bg_loss_mode: str,
    target_bandgap: float,
    bg_margin_for_train: float,
    bg_margin_for_eval: float,
):
    """
    Create training and evaluation loss functions for band gap optimization.

    Returns:
        gap_loss_func_for_train: GapLoss object used during training
        gap_loss_func_for_eval: GapLoss object used during evaluation
        gap_min_eval: float
        gap_max_eval: float
    """
    if bg_loss_mode == "band":
        gap_min_train = target_bandgap - bg_margin_for_train
        gap_max_train = target_bandgap + bg_margin_for_train
        gap_min_eval = target_bandgap - bg_margin_for_eval
        gap_max_eval = target_bandgap + bg_margin_for_eval
        gap_loss_func_for_train = GapLoss(
            gap_min=gap_min_train, gap_max=gap_max_train, mode=bg_loss_mode
        )
        gap_loss_func_for_eval = GapLoss(
            gap_min=gap_min_eval, gap_max=gap_max_eval, mode=bg_loss_mode
        )

    elif bg_loss_mode == "minimization" or bg_loss_mode == "maximization":
        gap_loss_func_for_train = GapLoss(mode=bg_loss_mode)
        gap_loss_func_for_eval = GapLoss(mode=bg_loss_mode)
        gap_min_eval, gap_max_eval = (
            0.0,
            0.0,
        )
    elif bg_loss_mode == "no_loss":
        gap_loss_func_for_train = GapLoss(mode=bg_loss_mode)
        gap_loss_func_for_eval = GapLoss(mode=bg_loss_mode)
        gap_min_eval, gap_max_eval = (
            0.0,
            0.0,
        )
    else:
        raise ValueError(
            "bg_loss_mode must be 'band', 'minimization' or 'maximization'"
        )

    return gap_loss_func_for_train, gap_loss_func_for_eval, gap_min_eval, gap_max_eval


def main_experiment(
    exp_name=None,
    num_steps: int = 200,
    model_name="Crystalformer",  #'Crystalformer' or 'ALIGNN' or 'Both'
    copy_mutation="copy_mutation",
    initial_dataset="megnet",  # 'megnet' or 'jarvis_supercon'
    ox_solution_mode: str = "strict_unique",
    wyckoff_expander: Optional[WyckoffExpander] = None  ,
    input_dataframe: Optional[pd.DataFrame] = None  ,
    bg_margin_for_train: float = 0.04,
    bg_margin_for_eval: float = 0.2,
    bg_loss_mode: str = "band",  # 'band', 'minimize', 'maximize'
    num_candidate: int = 256,  # max 4096
    num_batch_crystal: int = 256,  # max 4096
    target_bandgap: Optional[float] = None,
    perovskite_size: Optional[str] = None,
    crystal_system: Optional[str] = None,
    atom_lr: float = 2.0,
    lattice_lr: float = 0.003,
    coords_lr: float = 0.005,
    atom_cycle: int = 1,
    lattice_cycle: int = 1,
    coords_cycle: int = 1,
    e_form_criteria: float = -0.5,
    e_form_coef: float = 1.0,
    limit_coords_displacement=0.15,
    angle_range=(30, 150),
    abc_range=(2.0, 10.0),
    max_ox_states: int = 10,
    test_mode: bool = False,
    ALIGNN_update_step: Optional[List[int]] = None,
    tolerance_range=(None, None),  # tolerance range for perovskite
    device="cuda",
    mask_method="super_common",  # atomic mask type
    num_workers: int = 4,
    task="bg_and_eform",
    config: Optional[Dict[str, Any]] = None,
    energy_coef: float = 0.0,
    energy_every: int = 1,
    mattersim_model: Optional[str] = None,
    initial_structures_path: Optional[str] = None,
    mask_dict_path: Optional[str] = None,
    alignn_weights_dir: Optional[str] = None,
    use_element_count_l1_loss: bool = False,
    element_count_l1_weight: float = 1.0,
    superhard_cfg: Optional[Dict[str, Any]] = None,
    optimizer: str = "gradient",
    seed: Optional[int] = None,
    sa_config: Optional[Dict[str, Any]] = None,
    grad_sa_hybrid_config: Optional[Dict[str, Any]] = None,
    poscar_magmom_min: Optional[float] = None,

):
    if optimizer == "grad_sa_hybrid" and model_name != "ALIGNN":
        raise NotImplementedError(
            "optimizer=grad_sa_hybrid is currently supported only for model_name='ALIGNN'"
        )

    # ----------------- Hyperparameters ----------------- #
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    if test_mode:
        num_candidate = 8  # len(os.listdir(dir_path))
        num_batch_crystal = 8  # 1024
        num_steps = 200

    # ---------------- dataset directory for templete structures and oxidation-number mask ----------------- #
    if initial_structures_path is None:
        dir_path, dataset_name = define_initial_dataset_dir_path(
            crystal_system=crystal_system,
            initial_dataset=initial_dataset,
            neutral="super_common",
            max_atoms=10,
            perovskite_mode="random",
            perovskite_size=perovskite_size,
        )
    else:
        dir_path = initial_structures_path
        dataset_name = os.path.basename(os.path.normpath(dir_path))
    intermid_dir_path = os.path.join(get_master_dir(), "data/intermidiate/")
    if mask_dict_path is not None and crystal_system == "perovskite":
        raise ValueError(
            "mask_dict_path is not supported for perovskite mode. "
            "Use the default ionic masks/radii instead."
        )

    # ----------------- Perovskite settings ----------------- #
    if crystal_system == "perovskite":
        tolerance_range = (0.8, 1.0)
        angle_optimization = False
        tolerance_loss_func = ToleranceLoss(
            low=tolerance_range[0], high=tolerance_range[1]
        )
        max_ox_states = 2
    else:
        tolerance_loss_func = None
        angle_optimization = True
        max_ox_states = 10
        #if (limit_coords_displacement is not None):
        #    raise ValueError(
        #        "limit_coords_displacement must be None for non-perovskite system"
        #    )
    # for super cell
    if perovskite_size in ["2x2x2", "3x3x3", "4x4x4"]:
        print(
            f"<info> modify coords_lr for larger perovskite {perovskite_size}. coords_lr:",
            coords_lr,
            end=" -> ",
        )
        perovskite_size_dict = {"2x2x2": 2, "3x3x3": 3, "4x4x4": 4}
        print(
            f"{limit_coords_displacement} \n<info> modify abc_range for larger perovskite {abc_range}. abc_range:",
            abc_range,
            end=" -> ",
        )
        abc_range = np.array(abc_range) * perovskite_size_dict[perovskite_size]
        abc_range = tuple(abc_range)
        print(f"{abc_range}")

    predictors = None
    loss_composer = None
    primary_metrics = None
    if config and config.get("predictors"):
        predictor_types = {p["type"].lower() for p in config["predictors"]}
        if model_name == "ALIGNN" and predictor_types - {"alignn"}:
            raise ValueError("ALIGNN mode supports only alignn predictors")
        if model_name == "Crystalformer" and predictor_types - {"crystalformer"}:
            raise ValueError("Crystalformer mode supports only crystalformer predictors")
        predictors = build_predictors(config["predictors"], device=device)
        if not config.get("loss", {}).get("terms"):
            raise ValueError("loss.terms is required when predictors are configured")
        evaluator = SafeExpressionEvaluator()
        loss_composer = LossComposer(config["loss"]["terms"], evaluator)
        primary_metrics = _parse_primary_metrics(config)
        if not primary_metrics:
            raise ValueError("primary_metrics is required when predictors are configured")

    energy_module = None
    if energy_coef > 0.0:
        if energy_every < 1:
            raise ValueError("energy_every must be >= 1")
        energy_module = MatterSimEnergyModule(
            device=device, model_path=mattersim_model
        )

    # -----------------  load pretrained models ------------------ #
    if model_name == "Crystalformer":
        if predictors is None:
            bandgap_model, e_form_model = Load_Pretrained_Crystalformers()
        else:
            bandgap_model, e_form_model = None, None
        optimize_solution = optimize_solution_Crystalformer
        evaluation_for_each_batch = evaluation_for_each_batch_Crystalformer
    elif model_name == "ALIGNN":
        if predictors is None:
            bandgap_model, e_form_model = Load_Pretrained_ALIGNN(
                task,
                device=device,
                weights_dir=alignn_weights_dir,
            )
        else:
            bandgap_model, e_form_model = None, None
        optimize_solution = optimize_solution_ALIGNN
        evaluation_for_each_batch = evaluation_for_each_batch_ALIGNN
    else:
        raise NotImplementedError
    clear_output()

    # -----------------  Create setting dict ----------------- - #
    settings_dict = {}
    settings_dict["intermid_dir_path"] = intermid_dir_path
    settings_dict["crystal_system"] = crystal_system
    settings_dict["max_ox_states"] = max_ox_states
    settings_dict["num_candidate"] = num_candidate
    settings_dict["num_batch_crystal"] = num_batch_crystal
    settings_dict["use_atomic_mask"] = True
    settings_dict["mask_method"] = mask_method
    settings_dict["test_mode"] = test_mode
    settings_dict["learning_rates"] = [lattice_lr, atom_lr, coords_lr]
    settings_dict["learning_rate_cycle"] = [lattice_cycle, atom_cycle, coords_cycle]
    settings_dict["num_steps"] = num_steps
    settings_dict["angle_optimization"] = angle_optimization
    settings_dict["test_mode"] = test_mode
    settings_dict["angle_range"] = angle_range
    settings_dict["abc_range"] = abc_range
    settings_dict["copy_mutation"] = copy_mutation
    settings_dict["initial_dataset"] = initial_dataset
    settings_dict["ox_solution_mode"] = ox_solution_mode
    settings_dict["num_workers"] = num_workers
    settings_dict["optimizer"] = optimizer
    if sa_config is not None:
        settings_dict["sa_config"] = sa_config
    if grad_sa_hybrid_config is not None:
        settings_dict["grad_sa_hybrid_config"] = grad_sa_hybrid_config
    if superhard_cfg is not None:
        settings_dict["superhard_cfg"] = superhard_cfg
    if ALIGNN_update_step is not None:
        settings_dict["ALIGNN_update_step"] = (
            ALIGNN_update_step  # list(np.arange(200)+1)
        )
    else:
        settings_dict["ALIGNN_update_step"] = []

    # ----------------- Prepare band gap loss functions and evaluation thresholds ----------------- #
    gap_loss_func_for_train, gap_loss_func_for_eval, gap_min_eval, gap_max_eval = (
        prepare_gap_loss_funcs(
            bg_loss_mode=bg_loss_mode,
            target_bandgap=target_bandgap,
            bg_margin_for_train=bg_margin_for_train,
            bg_margin_for_eval=bg_margin_for_eval,
        )
    )

    # ----------------- Experiment and File Names ----------------- #
    paths = prepare_result_directories(exp_name)
    result_dir = paths["result_dir"]
    npz_path = paths["npz_path"]
    poscar_saved_dir = paths["poscar_saved_dir"]
    history_img_path = paths["history_img_path"]
    timing_log_path = os.path.join(result_dir, "opt_timing.jsonl")

    # -----------------  Load Data ------------------ #
    all_inputs_dict = load_initial_data(
        settings_dict=settings_dict,
        model_name = model_name,
        dir_path = dir_path,
        intermid_dir_path = settings_dict['intermid_dir_path'],
        mask_dict_path = mask_dict_path,
        crystal_system = settings_dict['crystal_system'],
        initial_dataset = settings_dict["initial_dataset"],
        ox_solution_mode = settings_dict["ox_solution_mode"],
        max_ox_states = settings_dict['max_ox_states'],
        num_candidate = settings_dict['num_candidate'],
        batch_size = settings_dict['num_batch_crystal'],
        use_atomic_mask = settings_dict['use_atomic_mask'],
        num_workers = settings_dict['num_workers'],
        mask_method = settings_dict['mask_method'],
        test_mode = settings_dict['test_mode'],
        angle_range = settings_dict['angle_range'],
        abc_range = settings_dict['abc_range'],
        input_dataframe=input_dataframe,
        wyckoff_expander=wyckoff_expander,
        device=device,
    )            

    total_atoms = 0
    total_structures = 0
    for mini_batch_inputs_dict, _ in all_inputs_dict:
        size = mini_batch_inputs_dict["size"]
        if torch.is_tensor(size):
            size = size.detach().cpu().numpy()
        total_atoms += int(np.sum(size))
        total_structures += int(len(size))
    avg_atoms = float(total_atoms) / float(total_structures) if total_structures else 0.0
    with open(timing_log_path, "a", encoding="ascii") as f:
        f.write(
            json.dumps(
                {
                    "type": "dataset_stats",
                    "avg_atoms": avg_atoms,
                    "total_structures": int(total_structures),
                },
                ensure_ascii=True,
            )
            + "\n"
        )

    # -----------------  Optimization and evaluation for each mini batch ------------------ #
    dist_temp_sche = set_schedule(num_steps=num_steps)
    optimized_dict_list, time_series_list = [], []
    for mini_batch_data in tqdm(all_inputs_dict, desc="mini_batch", leave=False):
        # select batch candidate crystals
        mini_batch_inputs_dict, scalers = mini_batch_data

        # Optimization
        optimize_kwargs = {
            "settings_dict": settings_dict,
            "mini_batch_inputs_dict": mini_batch_inputs_dict,
            "learning_rates": [lattice_lr, atom_lr, coords_lr],
            "learning_rate_cycle": [lattice_cycle, atom_cycle, coords_cycle],
            "num_steps": num_steps,
            "dist_temp_sche": dist_temp_sche,
            "e_form_coef": e_form_coef,
            "gap_loss_func": gap_loss_func_for_train,
            "tolerance_loss_func": tolerance_loss_func,
            "angle_optimization": angle_optimization,
            "bandgap_model": bandgap_model,
            "e_form_model": e_form_model,
            "predictors": predictors,
            "loss_composer": loss_composer,
            "primary_metrics": primary_metrics,
            "energy_module": energy_module,
            "energy_coef": energy_coef,
            "energy_every": energy_every,
            "use_element_count_l1_loss": use_element_count_l1_loss,
            "element_count_l1_weight": element_count_l1_weight,
            "adding_noise_scale": None,
            "scalers": scalers,
            "crystal_system": crystal_system,
            "limit_coords_displacement": limit_coords_displacement,
            "device": device,
            "time_series_save": False,
            "e_form_min": None,
            "wyckoff_expander": wyckoff_expander,
            "optimizer": optimizer,
        }
        if model_name == "ALIGNN":
            optimize_kwargs["timing_log_path"] = timing_log_path
        optimized_mini_batch_inputs_dict, time_series, scalers = optimize_solution(
            **optimize_kwargs
        )
        # Evaluation
        optimized_mini_batch_inputs_dict = evaluation_for_each_batch(
            settings_dict = settings_dict,
            optimized_mini_batch_inputs_dict=optimized_mini_batch_inputs_dict,
            scalers=scalers,
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            predictors=predictors,
            loss_composer=loss_composer,
            primary_metrics=primary_metrics,
            e_form_coef=e_form_coef,
            gap_loss_func=gap_loss_func_for_eval,
            crystal_system=crystal_system,
            limit_coords_displacement=limit_coords_displacement,
            tolerance_loss_func=tolerance_loss_func,
            adding_noise_scale=None,
            e_form_min=None,
            wyckoff_expander=wyckoff_expander,
            energy_module=energy_module,
            energy_coef=energy_coef,
        )

        stored_mini_batch_inputs_dict = _move_result_payload_to_cpu(
            optimized_mini_batch_inputs_dict
        )
        optimized_dict_list.append(stored_mini_batch_inputs_dict)

        time_series_list.append(time_series)

    # -----------------  Save and Visualization ------------------ #
    save_optimization_results(
        npz_path=npz_path,
        optimized_dict_list=optimized_dict_list,
        model_name=model_name,
        crystal_system=crystal_system,
    )

    # -----------------  evaluation and analysis -----------------  #
    if task == "B_and_eform":
        main_target_name = "B"
    elif task == "superhard":
        main_target_name = "G"
    elif task == "tc_and_efrom":
        main_target_name = "Tc"
    else:
        main_target_name = "bandgap"
    evaluation_and_analysis(
        npz_path,
        bg_loss_mode,
        poscar_saved_dir,
        history_img_path,
        crystal_system,
        perovskite_size,
        gap_min_eval,
        gap_max_eval,
        model_name,
        num_candidate,
        tolerance_range,
        e_form_criteria,
        limit_coords_displacement,
    main_target_name,
        wyckoff_expander,
        task,
        superhard_cfg=superhard_cfg,
        poscar_magmom_min=poscar_magmom_min,
    )

    clear_output()
    img = Image.open(history_img_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()
    display(history_img_path)

    return result_dir
