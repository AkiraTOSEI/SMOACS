from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.losses.loss_utils import calculate_loss_from_output
from src.optimization.optimizer_utils import define_optimizer_and_scheduler
from src.smoacs_opt_eval.alignn_forward import forward_propagation_ALIGNN
from src.smoacs_opt_eval.alignn_graph_update import (
    update_graphs_and_trainable_parameters,
)
from src.smoacs_opt_eval.copy_and_mutate import parse_evolution_parameters
from src.smoacs_opt_eval.step_update_utils import update_parameters_step
from src.utils.common import create_element_list
from src.utils.feature_utils import calculate_onehot
from src.utils.nn_utils import temperature_softmax
from src.utils.poscar_utils import save_poscar
from src.utils.scalers import ABC_Scaler, Angle_Scaler


def optimize_solution_ALIGNN(
    settings_dict: Dict[str, Any],
    mini_batch_inputs_dict: Dict[str, Any],
    learning_rates: List[float],
    learning_rate_cycle: List[int],
    num_steps: int,
    dist_temp_sche: List[float],
    e_form_coef: float,
    gap_loss_func: torch.nn.Module,
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    scalers: dict,
    adding_noise_scale: Optional[float],
    e_form_min: Optional[float],
    limit_coords_displacement: Optional[float],
    crystal_system: Optional[str] = None,
    angle_optimization: bool = True,
    tolerance_loss_func=Optional[torch.nn.Module],
    device: str = "cuda",
    time_series_save: bool = True,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Performs optimization of a batch of candidate crystal structures using ALIGNN.

    Args:
        settings_dict: Dictionary containing configuration settings.
        mini_batch_inputs_dict: Dictionary containing batch-specific model inputs.
        learning_rates: List of learning rates for lattice, atom, and coordinate optimization.
        learning_rate_cycle: List of cycle lengths for each optimizer.
        num_steps: Number of optimization steps.
        dist_temp_sche: Temperature schedule for softmax during optimization.
        e_form_coef: weight for  formation energy loss.
        gap_loss_func: Loss function module for bandgap.
        bandgap_model: ALIGNN model predicting bandgap.
        e_form_model: ALIGNN model predicting formation energy.
        scalers: Dictionary containing ABC and angle scalers.
        adding_noise_scale: Amount of noise to add to coordinates.
        e_form_min: Minimum formation energy threshold.
        limit_coords_displacement: Maximum displacement from initial coordinates.
        crystal_system: Optional crystal system type (e.g. "perovskite").
        angle_optimization: Whether to optimize lattice angles.
        tolerance_loss_func: Loss function module for tolerance.
        device: Computation device ('cuda' or 'cpu').
        time_series_save: Whether to store optimization history.

    Returns:
        Tuple containing:
            - optimized_mini_batch_inputs_dict: Dictionary of optimized inputs.
            - time_series: List of history arrays for structural variables and predictions.
            - scalers: Updated scalers.
    """
    abc_scaler = scalers["abc_scaler"]
    angle_scaler = scalers["angle_scaler"]
    atomic_distribution = nn.Parameter(
        mini_batch_inputs_dict["atomic_distribution"].clone().to("cuda"),
        requires_grad=True,
    )
    scaled_batch_abc = nn.Parameter(
        abc_scaler.scale(mini_batch_inputs_dict["batch_abc"].clone().to("cuda")),
        requires_grad=True,
    )

    scaled_batch_angle = nn.Parameter(
        angle_scaler.scale(mini_batch_inputs_dict["batch_angle"].clone().to("cuda")),
        requires_grad=True,
    )
    ox_mask_learnable_tensor_per_crystal = nn.Parameter(
        mini_batch_inputs_dict["ox_mask_learnable_tensor_per_crystal"]
        .to("cuda")
        .clone(),
        requires_grad=True,
    )
    batch_dir_coords = nn.Parameter(
        mini_batch_inputs_dict["batch_dir_coords"].to("cuda").clone(),
        requires_grad=True,
    )
    radii_tensor = mini_batch_inputs_dict["radii_tensor"].to("cuda")
    site_ids = mini_batch_inputs_dict["site_ids"].to("cuda")

    abc_range = settings_dict["abc_range"]
    angle_range = settings_dict["angle_range"]

    # optimizer and scheduler
    (
        optimizer_lattice,
        optimizer_atom,
        optimizer_coords,
        scheduler_lattice,
        scheduler_atom,
        scheduler_coords,
    ) = define_optimizer_and_scheduler(
        lattice_lr=learning_rates[0],
        atom_lr=learning_rates[1],
        coords_lr=learning_rates[2],
        ox_mask_learnable_tensor_per_crystal=ox_mask_learnable_tensor_per_crystal,
        atomic_distribution=atomic_distribution,
        scaled_batch_abc=scaled_batch_abc,
        scaled_batch_angle=scaled_batch_angle,
        batch_dir_coords=batch_dir_coords,
        angle_optimization=angle_optimization,
        num_steps=num_steps,
        lattice_cycle=learning_rate_cycle[0],
        atom_cycle=learning_rate_cycle[1],
        coords_cycle=learning_rate_cycle[2],
    )

    # initialzation of history
    lattice_mat_history, coords_history, normed_coords_history, elememts_history = (
        [],
        [],
        [],
        [],
    )
    abc_history, angle_history = [], []
    gap_history, e_form_history, t_loss_history = [], [], []
    VALID_ELEMENTS98 = create_element_list()
    lattice_lr_history, atom_lr_history, coords_lr_history = [], [], []

    # (English) After copy_and_mutate, it is necessary to perform the ALIGNN graph update_step.
    copy_mutation = settings_dict["copy_mutation"]
    parse_evolution_parameters_dict = parse_evolution_parameters(copy_mutation)
    if parse_evolution_parameters_dict["steps_cm"][0] != "None":
        additional_alignn_update_steps = [
            int(step) + 1 for step in parse_evolution_parameters_dict["steps_cm"]
        ]
        settings_dict["ALIGNN_update_step"].extend(additional_alignn_update_steps)

    optimization_targets = [
        scaled_batch_abc,
        scaled_batch_angle,
        batch_dir_coords,
        atomic_distribution,
        ox_mask_learnable_tensor_per_crystal,
    ]
    optimization_target_names = [
        "scaled_batch_abc",
        "scaled_batch_angle",
        "batch_dir_coords",
        "atomic_distribution",
        "ox_mask_learnable_tensor_per_crystal",
    ]
    for step_i in tqdm(range(num_steps), desc="Optimizing", leave=False):
        ### Update graph and inputs
        if step_i in settings_dict["ALIGNN_update_step"]:
            print(f"<info> graph update at step {step_i}")
            # 学習パラメータとOptimizerを再度初期化する。
            #  cosine annealingのlearning rate の最大値を、現在の値へと更新する
            lattice_lr = optimizer_lattice.param_groups[0]["lr"]
            atom_lr = optimizer_atom.param_groups[0]["lr"]
            coords_lr = optimizer_coords.param_groups[0]["lr"]
            settings_dict["learning_rates"] = [lattice_lr, atom_lr, coords_lr]

            (
                upatate_mini_batch_inputs_dict,
                updated_scalers,
                optimiers_dict,
                learnable_parameters_dict,
            ) = update_graphs_and_trainable_parameters(
                optimized_mini_batch_inputs_dict={
                    "atomic_distribution": atomic_distribution,
                    "batch_abc": abc_scaler.rescale(scaled_batch_abc),
                    "batch_angle": angle_scaler.rescale(scaled_batch_angle),
                    "batch_dir_coords": batch_dir_coords,
                    "ox_mask_learnable_tensor_per_crystal": ox_mask_learnable_tensor_per_crystal,
                    "size": mini_batch_inputs_dict["size"],
                    "atomic_mask": mini_batch_inputs_dict["atomic_mask"],
                    "ox_states_used_mask": mini_batch_inputs_dict[
                        "ox_states_used_mask"
                    ],
                    "atom_feat_matrix": mini_batch_inputs_dict["atom_feat_matrix"],
                    "site_ids": mini_batch_inputs_dict["site_ids"],
                    "radii_tensor": mini_batch_inputs_dict["radii_tensor"],
                    "init_coords": mini_batch_inputs_dict["init_coords"],
                    "scaled_batch_abc": scaled_batch_abc,
                    "scaled_batch_angle": scaled_batch_angle,
                    "fnames": mini_batch_inputs_dict["fnames"],
                    "g": mini_batch_inputs_dict["g"],
                },
                settings_dict=settings_dict,
                tmp_poscar_dir="./tmp_poscars",
            )
            del mini_batch_inputs_dict, scalers
            mini_batch_inputs_dict = upatate_mini_batch_inputs_dict
            scalers = updated_scalers
            abc_scaler, angle_scaler = scalers["abc_scaler"], scalers["angle_scaler"]
            ##### update variables
            scaled_batch_abc, scaled_batch_angle, batch_dir_coords = (
                learnable_parameters_dict["scaled_batch_abc"],
                learnable_parameters_dict["scaled_batch_angle"],
                learnable_parameters_dict["batch_dir_coords"],
            )
            atomic_distribution, ox_mask_learnable_tensor_per_crystal = (
                learnable_parameters_dict["atomic_distribution"],
                learnable_parameters_dict["ox_mask_learnable_tensor_per_crystal"],
            )
            optimizer_lattice, optimizer_atom, optimizer_coords = (
                optimiers_dict["optimizer_lattice"],
                optimiers_dict["optimizer_atom"],
                optimiers_dict["optimizer_coords"],
            )
            scheduler_lattice, scheduler_atom, scheduler_coords = (
                optimiers_dict["scheduler_lattice"],
                optimiers_dict["scheduler_atom"],
                optimiers_dict["scheduler_coords"],
            )
            optimization_targets = [
                scaled_batch_abc,
                scaled_batch_angle,
                batch_dir_coords,
                atomic_distribution,
                ox_mask_learnable_tensor_per_crystal,
            ]

        ### backpropのたびに計算する
        # re-calculate atomic features, crystal structures, bondlength, bond angles
        bandgap_pred, ef_pred, output_dict = forward_propagation_ALIGNN(
            optimization_targets=optimization_targets,
            fixed_inputs=mini_batch_inputs_dict,
            scalers=scalers,
            temperature=dist_temp_sche[step_i],
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            adding_noise_scale=adding_noise_scale,
            limit_coords_displacement=limit_coords_displacement,
            abc_range=abc_range,
            angle_range=angle_range,
        )
        total_loss, each_loss, gap_loss, ef_loss, tolerance_loss, tolerance = (
            calculate_loss_from_output(
                bandgap_pred=bandgap_pred,
                ef_pred=ef_pred,
                sharpened_ox_mask=output_dict["sharpened_ox_mask"],
                normalized_dist=output_dict["normalized_dist"],
                site_ids=site_ids,
                gap_loss_func=gap_loss_func,
                tolerance_loss_func=tolerance_loss_func,
                ef_coef=e_form_coef,
                crystal_system=crystal_system,
                e_form_min=e_form_min,
                radii_tensor=radii_tensor,
            )
        )

        if step_i == 0:
            # save initial poscar
            save_poscar(mini_batch_inputs_dict | output_dict, "init_poscars")

        # Update parameters
        size = mini_batch_inputs_dict["size"].to("cuda")
        cry_atom_data = torch.repeat_interleave(
            torch.arange(len(size), device=size.device), size
        )
        total_loss, optimization_targets, additional_info = update_parameters_step(
            optimization_targets=optimization_targets,
            optimization_target_names=optimization_target_names,
            step_i=step_i,
            total_loss=total_loss,
            gap_loss=gap_loss,
            ef_loss=ef_loss,
            tolerance_loss=tolerance_loss,
            each_loss=each_loss,
            total_steps=num_steps,
            learning_rates=learning_rates,
            cry_atom_data=cry_atom_data,
            size=size,
            atomic_mask=mini_batch_inputs_dict["atomic_mask"],
            ox_states_used_mask=mini_batch_inputs_dict["ox_states_used_mask"],
            site_ids=mini_batch_inputs_dict["site_ids"].to("cuda"),
            radii_tensor=mini_batch_inputs_dict["radii_tensor"].to("cuda"),
            init_coords=mini_batch_inputs_dict["init_coords"].to("cuda"),
            copy_mutation=copy_mutation,
            angle_optimization=angle_optimization,
            optimizer_lattice=optimizer_lattice,
            optimizer_atom=optimizer_atom,
            optimizer_coords=optimizer_coords,
            scheduler_lattice=scheduler_lattice,
            scheduler_atom=scheduler_atom,
            scheduler_coords=scheduler_coords,
            fixed_inputs=mini_batch_inputs_dict,
            scalers=scalers,
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            limit_coords_displacement=limit_coords_displacement,
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            Ef_coef_value=e_form_coef,
            crystal_system=crystal_system,
            e_form_min=e_form_min,
        )
        # update optimizer
        optimizer_atom = additional_info["optimizer_atom"]
        optimizer_lattice = additional_info["optimizer_lattice"]
        optimizer_coords = additional_info["optimizer_coords"]
        scheduler_lattice = additional_info["scheduler_lattice"]
        scheduler_atom = additional_info["scheduler_atom"]
        scheduler_coords = additional_info["scheduler_coords"]

        mini_batch_inputs_dict["batch"] = additional_info["new_cry_atom_data"].to(
            "cuda"
        )
        mini_batch_inputs_dict["size"] = additional_info["new_size"].to("cuda")
        mini_batch_inputs_dict["atomic_mask"] = additional_info["atomic_mask"].to(
            "cuda"
        )
        mini_batch_inputs_dict["ox_states_used_mask"] = additional_info[
            "ox_states_used_mask"
        ].to("cuda")
        mini_batch_inputs_dict["site_ids"] = additional_info["site_ids"].to("cuda")
        mini_batch_inputs_dict["radii_tensor"] = additional_info["radii_tensor"].to(
            "cuda"
        )
        mini_batch_inputs_dict["init_coords"] = additional_info["init_coords"].to(
            "cuda"
        )

        scalers["abc_scaler"] = additional_info["abc_scaler"]
        abc_scaler = additional_info["abc_scaler"]
        # 損失の記録
        gap_history.append(bandgap_pred.detach().cpu().numpy())
        e_form_history.append(ef_pred.detach().cpu().numpy())
        t_loss_history.append(tolerance.detach().cpu().numpy())

        # learning rateの記録
        lattice_lr_history.append(optimizer_lattice.param_groups[0]["lr"])
        atom_lr_history.append(optimizer_atom.param_groups[0]["lr"])
        coords_lr_history.append(optimizer_coords.param_groups[0]["lr"])
        # print(f'step{step_i}. coords_lr:{optimizer_coords.param_groups[0]["lr"]}, lattice_lr:{optimizer_lattice.param_groups[0]["lr"]}, atom_lr:{optimizer_atom.param_groups[0]["lr"]}')

        if time_series_save:
            lattice_mat_history.append(
                output_dict["lattice_vectors"].detach().cpu().numpy()
            )
            normed_coords_history.append(
                output_dict["normed_batch_dir_coords"].detach().cpu().numpy()
            )
            coords_history.append(batch_dir_coords.detach().cpu().numpy())
            abc_history.append(output_dict["batch_abc"].detach().cpu().numpy())
            angle_history.append(output_dict["batch_angle"].detach().cpu().numpy())
            elements = [
                VALID_ELEMENTS98[atom_idx]
                for atom_idx in torch.argmax(output_dict["normalized_dist"], dim=1)
            ]
            elememts_history.append(elements)

    # End of optimization
    [
        scaled_batch_abc,
        scaled_batch_angle,
        batch_dir_coords,
        atomic_distribution,
        ox_mask_learnable_tensor_per_crystal,
    ] = optimization_targets

    optimized_mini_batch_inputs_dict = mini_batch_inputs_dict | output_dict
    optimized_mini_batch_inputs_dict["atomic_distribution"] = atomic_distribution
    optimized_mini_batch_inputs_dict["scaled_batch_abc"] = scaled_batch_abc
    optimized_mini_batch_inputs_dict["scaled_batch_angle"] = scaled_batch_angle
    optimized_mini_batch_inputs_dict["ox_mask_learnable_tensor_per_crystal"] = (
        ox_mask_learnable_tensor_per_crystal
    )
    optimized_mini_batch_inputs_dict["batch_dir_coords"] = batch_dir_coords
    optimized_mini_batch_inputs_dict["gap_history"] = np.transpose(
        np.array(gap_history).squeeze(), (1, 0)
    )
    optimized_mini_batch_inputs_dict["ef_history"] = np.transpose(
        np.array(e_form_history).squeeze(), (1, 0)
    )
    optimized_mini_batch_inputs_dict["t_history"] = np.transpose(
        np.array(t_loss_history).squeeze(), (1, 0)
    )
    optimized_mini_batch_inputs_dict["lattice_lr_history"] = lattice_lr_history
    optimized_mini_batch_inputs_dict["atom_lr_history"] = atom_lr_history
    optimized_mini_batch_inputs_dict["coords_lr_history"] = coords_lr_history
    time_series = [
        lattice_mat_history,
        coords_history,
        normed_coords_history,
        abc_history,
        angle_history,
        elememts_history,
    ]

    return (optimized_mini_batch_inputs_dict, time_series, scalers)
