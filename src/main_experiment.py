import itertools
import os
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

# model loading and optimization (assumed to be project-specific modules)
from src.smoacs_opt_eval.alignn_eval import evaluation_for_each_batch_ALIGNN
from src.smoacs_opt_eval.alignn_graph_update import calculate_update_steps
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
    task="bg_and_eform",
):
    # ----------------- Hyperparameters ----------------- #

    if test_mode:
        num_candidate = 8  # len(os.listdir(dir_path))
        num_batch_crystal = 8  # 1024
        num_steps = 200

    # ---------------- dataset directory for templete structures and oxidation-number mask ----------------- #
    dir_path, dataset_name = define_initial_dataset_dir_path(
        crystal_system=crystal_system,
        initial_dataset=initial_dataset,
        neutral="super_common",
        max_atoms=10,
        perovskite_mode="random",
        perovskite_size=perovskite_size,
    )
    intermid_dir_path = os.path.join(get_master_dir(), "data/intermidiate/")

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
        if limit_coords_displacement is not None:
            raise ValueError(
                "limit_coords_displacement must be None for non-perovskite system"
            )
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

    # -----------------  load pretrained models ------------------ #
    if model_name == "Crystalformer":
        bandgap_model, e_form_model = Load_Pretrained_Crystalformers()
        optimize_solution = optimize_solution_Crystalformer
        evaluation_for_each_batch = evaluation_for_each_batch_Crystalformer
    elif model_name == "ALIGNN":
        bandgap_model, e_form_model = Load_Pretrained_ALIGNN(task)
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

    # -----------------  Load Data ------------------ #
    all_inputs_dict = load_initial_data(
        settings_dict=settings_dict,
        model_name=model_name,
        dir_path=dir_path,
        intermid_dir_path=settings_dict["intermid_dir_path"],
        crystal_system=settings_dict["crystal_system"],
        initial_dataset=settings_dict["initial_dataset"],
        max_ox_states=settings_dict["max_ox_states"],
        num_candidate=settings_dict["num_candidate"],
        batch_size=settings_dict["num_batch_crystal"],
        use_atomic_mask=settings_dict["use_atomic_mask"],
        mask_method=settings_dict["mask_method"],
        test_mode=settings_dict["test_mode"],
        angle_range=settings_dict["angle_range"],
        abc_range=settings_dict["abc_range"],
    )

    # -----------------  Optimization and evaluation for each mini batch ------------------ #
    dist_temp_sche = set_schedule(num_steps=num_steps)
    optimized_dict_list, time_series_list = [], []
    for mini_batch_data in tqdm(all_inputs_dict, desc="mini_batch", leave=False):
        # select batch candidate crystals
        mini_batch_inputs_dict, scalers = mini_batch_data

        # Optimization for each mini batch
        optimized_mini_batch_inputs_dict, time_series, scalers = optimize_solution(
            settings_dict=settings_dict,
            mini_batch_inputs_dict=mini_batch_inputs_dict,
            learning_rates=[lattice_lr, atom_lr, coords_lr],
            learning_rate_cycle=[lattice_cycle, atom_cycle, coords_cycle],
            num_steps=num_steps,
            dist_temp_sche=dist_temp_sche,
            e_form_coef=e_form_coef,
            gap_loss_func=gap_loss_func_for_train,
            tolerance_loss_func=tolerance_loss_func,
            angle_optimization=angle_optimization,
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            adding_noise_scale=None,
            scalers=scalers,
            crystal_system=crystal_system,
            limit_coords_displacement=limit_coords_displacement,
            device=device,
            time_series_save=False,
            e_form_min=None,
        )
        # Evaluation for each mini batch
        optimized_mini_batch_inputs_dict = evaluation_for_each_batch(
            settings_dict=settings_dict,
            optimized_mini_batch_inputs_dict=optimized_mini_batch_inputs_dict,
            scalers=scalers,
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            ef_coef4eval=e_form_coef,
            gap_loss_func=gap_loss_func_for_eval,
            crystal_system=crystal_system,
            limit_coords_displacement=limit_coords_displacement,
            tolerance_loss_func=tolerance_loss_func,
            adding_noise_scale=None,
            e_form_min=None,
        )

        optimized_dict_list.append(optimized_mini_batch_inputs_dict)

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
    )

    clear_output()
    img = Image.open(history_img_path)
    plt.figure(figsize=(10, 10))
    plt.imshow(img)
    plt.show()
    display(history_img_path)

    return result_dir
