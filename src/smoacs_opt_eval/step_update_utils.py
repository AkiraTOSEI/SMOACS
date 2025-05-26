from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.smoacs_opt_eval.copy_and_mutate import (
    copy_and_mutate_update,
    parse_evolution_parameters,
)
from src.utils.nn_utils import temperature_softmax


def update_parameters_step(
    optimization_targets: List[torch.Tensor],
    optimization_target_names: List[str],
    step_i: int,
    total_steps: int,
    total_loss: torch.Tensor,
    each_loss: torch.Tensor,
    gap_loss: torch.Tensor,
    ef_loss: torch.Tensor,
    tolerance_loss: torch.Tensor,
    learning_rates: List[float],  # [lr_lattice, lr_atom, lr_coords]
    copy_mutation: str,
    cry_atom_data: torch.Tensor,
    size: torch.Tensor,
    atomic_mask: torch.Tensor,
    ox_states_used_mask: torch.Tensor,
    site_ids: torch.Tensor,
    radii_tensor: torch.Tensor,
    init_coords: torch.Tensor,
    angle_optimization: bool,
    optimizer_lattice: torch.optim.Optimizer,
    optimizer_atom: torch.optim.Optimizer,
    optimizer_coords: torch.optim.Optimizer,
    scheduler_lattice: torch.optim.lr_scheduler._LRScheduler,
    scheduler_atom: torch.optim.lr_scheduler._LRScheduler,
    scheduler_coords: torch.optim.lr_scheduler._LRScheduler,
    fixed_inputs: dict,
    scalers: dict,
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    limit_coords_displacement: float,
    gap_loss_func: torch.nn.Module,
    tolerance_loss_func: Any,
    Ef_coef_value: float,
    crystal_system: Optional[str],
    e_form_min: float,
) -> Tuple[float, List[torch.Tensor], Dict[str, Any]]:
    """
    Perform one step of parameter update during structure optimization.

    This function handles gradient-based (Adam) and simulated annealing (SA) optimization modes.
    It also supports evolutionary operations like copy-and-mutate updates at specified steps.

    Args:
        optimization_targets (List[torch.Tensor]): Trainable tensors for optimization (e.g., coords, abc, atomic distribution).
        optimization_target_names (List[str]): Names of each optimization target.
        step_i (int): Current optimization step.
        total_steps (int): Total number of optimization steps.
        total_loss (torch.Tensor): Total loss tensor (backward target).
        each_loss (torch.Tensor): Per-structure total losses.
        gap_loss (torch.Tensor): Per-structure bandgap losses.
        ef_loss (torch.Tensor): Per-structure formation energy losses.
        tolerance_loss (torch.Tensor): Per-structure tolerance factor losses.
        learning_rates (List[float]): Learning rates for [lattice, atom, coords].
        copy_mutation (str): Optimization mode ("copy_mutation", "sa", etc.).
        cry_atom_data (torch.Tensor): Tensor mapping atoms to their respective crystal.
        size (torch.Tensor): Number of atoms per structure.
        atomic_mask (torch.Tensor): Mask for valid atoms.
        ox_states_used_mask (torch.Tensor): Mask for valid oxidation states.
        site_ids (torch.Tensor): Per-atom site classification (A/B/X).
        radii_tensor (torch.Tensor): Ionic radii for atoms at each site.
        init_coords (torch.Tensor): Initial direct coordinates.
        angle_optimization (bool): Whether to optimize lattice angles.
        optimizer_lattice (Optimizer): Optimizer for lattice vectors.
        optimizer_atom (Optimizer): Optimizer for atomic distribution.
        optimizer_coords (Optimizer): Optimizer for atomic coordinates.
        scheduler_lattice (_LRScheduler): Scheduler for lattice optimizer.
        scheduler_atom (_LRScheduler): Scheduler for atomic optimizer.
        scheduler_coords (_LRScheduler): Scheduler for coordinate optimizer.
        fixed_inputs (dict): Fixed inputs for re-forward propagation.
        scalers (dict): Dictionary containing `abc_scaler` and `angle_scaler`.
        bandgap_model (nn.Module): Pretrained bandgap prediction model.
        e_form_model (nn.Module): Pretrained formation energy prediction model.
        limit_coords_displacement (float): Max allowed displacement from initial coordinates.
        gap_loss_func (nn.Module): Loss function for bandgap.
        tolerance_loss_func (Any): Loss function for tolerance factor.
        Ef_coef_value (float): Scaling coefficient for formation energy loss.
        crystal_system (Optional[str]): Crystal system name (e.g., "perovskite").
        e_form_min (float): Minimum formation energy threshold.

    Returns:
        Tuple containing:
            updated_loss (float): Updated scalar loss value.
            updated_targets (List[torch.Tensor]): Updated list of optimization targets.
            additional_info (Dict[str, Any]): Metadata including updated optimizers, masks, and scaler.
    """
    additional_info = {}
    if copy_mutation.startswith("copy_mutation"):
        optimizer_lattice.zero_grad()
        optimizer_atom.zero_grad()
        optimizer_coords.zero_grad()
        total_loss.backward()

        # Apply gradient steps
        optimizer_atom.step()
        optimizer_lattice.step()
        optimizer_coords.step()
        # Update learning rates
        scheduler_lattice.step()
        scheduler_atom.step()
        scheduler_coords.step()

        updated_loss = total_loss.item()
        additional_info["mode"] = "adam"
        additional_info["optimizer_lattice"] = optimizer_lattice
        additional_info["optimizer_atom"] = optimizer_atom
        additional_info["optimizer_coords"] = optimizer_coords
        additional_info["scheduler_lattice"] = scheduler_lattice
        additional_info["scheduler_atom"] = scheduler_atom
        additional_info["scheduler_coords"] = scheduler_coords
        additional_info["new_cry_atom_data"] = cry_atom_data
        additional_info["new_size"] = size
        additional_info["atomic_mask"] = atomic_mask
        additional_info["ox_states_used_mask"] = ox_states_used_mask
        additional_info["abc_scaler"] = scalers["abc_scaler"]
        additional_info["site_ids"] = site_ids
        additional_info["radii_tensor"] = radii_tensor
        additional_info["init_coords"] = fixed_inputs["init_coords"]

        # get current learning rates from scheduler_coords, optimizer_lattice, optimizer_atom
        current_learning_rates = [
            scheduler_lattice.get_last_lr()[0],
            optimizer_atom.param_groups[0]["lr"],
            optimizer_coords.param_groups[0]["lr"],
        ]

        # Perform evolutionary update after the optimizer step:
        # Copy parameters from selected samples (Group B) to others (Group C) and add mutation noise.
        parse_evolution_parameters_dict = parse_evolution_parameters(copy_mutation)
        if step_i in parse_evolution_parameters_dict["steps_cm"]:
            print(
                "---------------------------- \n    <info> Performing copy and mutate update... \n ----------------------------"
            )
            (
                optimization_targets,
                optimizers_and_schedulers,
                new_cry_atom_data,
                new_size,
                atomic_mask,
                ox_states_used_mask,
                site_ids,
                radii_tensor,
                init_coords,
                abc_scaler,
            ) = copy_and_mutate_update(
                optimization_targets=optimization_targets,
                optimization_target_names=optimization_target_names,
                crystal_system=crystal_system,
                cry_atom_data=cry_atom_data,
                size=size,
                atomic_mask=atomic_mask,
                ox_states_used_mask=ox_states_used_mask,
                site_ids=site_ids,
                radii_tensor=radii_tensor,
                init_coords=init_coords,
                current_learning_rates=current_learning_rates,
                current_step=step_i,
                total_steps=total_steps,
                angle_optimization=angle_optimization,
                each_loss=each_loss,
                gap_loss=gap_loss,
                ef_loss=ef_loss,
                tolerance_loss=tolerance_loss,
                abc_scaler=scalers["abc_scaler"],
                group_C_use_rate=parse_evolution_parameters_dict["group_C_use_rate"],
                atom_dist_init=parse_evolution_parameters_dict["atom_dist_init"],
                top_ratio=parse_evolution_parameters_dict["top_ratio"],
                mutation_noise=parse_evolution_parameters_dict["mutation_noise"],
            )
            # optimizers_and_schedulers
            additional_info["optimizer_lattice"] = optimizers_and_schedulers[0]
            additional_info["optimizer_atom"] = optimizers_and_schedulers[1]
            additional_info["optimizer_coords"] = optimizers_and_schedulers[2]
            additional_info["scheduler_lattice"] = optimizers_and_schedulers[3]
            additional_info["scheduler_atom"] = optimizers_and_schedulers[4]
            additional_info["scheduler_coords"] = optimizers_and_schedulers[5]
            # Update cry_atom_data and size
            additional_info["new_cry_atom_data"] = new_cry_atom_data
            additional_info["new_size"] = new_size
            additional_info["atomic_mask"] = atomic_mask
            additional_info["ox_states_used_mask"] = ox_states_used_mask
            additional_info["site_ids"] = site_ids
            additional_info["radii_tensor"] = radii_tensor
            additional_info["init_coords"] = init_coords

            # Update abc_scaler
            additional_info["abc_scaler"] = abc_scaler

    else:
        raise ValueError(f"Unknown copy_mutation: {copy_mutation}")

    return updated_loss, optimization_targets, additional_info
