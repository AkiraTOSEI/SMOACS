import random
from typing import List, Optional, Tuple

import torch

from src.optimization.optimizer_utils import define_optimizer_and_scheduler
from src.utils.common import create_element_list
from src.utils.mask_utils import (
    masking_atomic_distribution,
    stacking_learnable_oxsides_mask,
)
from src.utils.nn_utils import temperature_softmax
from src.utils.scalers import ABC_Scaler


def parse_evolution_parameters(copy_mutation_str: str) -> dict:
    """Parses the copy_mutation string to extract parameters for the evolutionary update.

    The copy_mutation string is expected to contain the evolution parameters after a
    'copy_mutation___' delimiter. Individual settings are separated by '__', and each
    setting consists of a key and one or more values separated by '-' (the first '-' is used
    as the separator). If a parameter not defined in the default list is encountered, a
    ValueError is raised.

    Default parameters:
        top_ratio: float = 0.25,
        mutation_noise: float = 0.05,
        steps_cm: List[int] = [50, 100, 150]

    For example, the input string:
        "copy_mutation___steps_cm-50-100-150__mutation_noise-0.05"
    will yield a dictionary:
        {
            "top_ratio": 0.25,            # default remains because not specified
            "mutation_noise": 0.05,
            "steps_cm": [50, 100, 150]
            "group_C_use_rate": 0.1,
            "atom_dist_init": True,
        }

    Also, if the input is:
        "copy_mutation___steps_cm-10"
    then the resulting dictionary will have:
        "steps_cm": [10]

    Args:
        copy_mutation_str (str): The optimization mode string.

    Returns:
        dict: A dictionary containing the evolution parameters.

    Raises:
        ValueError: If an unknown parameter is found in the input string.
    """
    # Default parameters for evolutionary update
    defaults = {
        "top_ratio": 0.25,
        "mutation_noise": 0.05,
        "steps_cm": [50, 100, 150],
        "group_C_use_rate": 0.1,
        "atom_dist_init": True,
    }

    # Check if the string starts with "copy_mutation"
    if copy_mutation_str.startswith("copy_mutation"):
        parts = copy_mutation_str.split("___", 1)
        if len(parts) == 2:
            param_str = parts[1]
        else:
            return defaults
    else:
        # If not starting with "copy_mutation", assume no evolution parameters provided.
        return defaults

    # Split the parameters using '__' as delimiter.
    settings = param_str.split("__")
    for setting in settings:
        # Each setting is expected to be in the format "key-value" or "key-value1-value2-..."
        tokens = setting.split("-")
        if len(tokens) < 2:
            continue  # skip empty or malformed setting
        key = tokens[0].strip()
        # Check if the key is valid.
        if key not in defaults:
            raise ValueError(f"Unknown evolution parameter: '{key}'")
        # Determine whether the default value is a list.
        if isinstance(defaults[key], list):
            # Always treat the value as a list, even if a single token is provided.
            values = []
            for token in tokens[1:]:
                token = token.strip()
                # Check if the token represents a number.
                if token.replace(".", "", 1).isdigit():
                    # Convert to float if it contains a dot, else int.
                    if "." in token:
                        values.append(float(token))
                    else:
                        values.append(int(token))
                else:
                    values.append(token)
            defaults[key] = values
        else:
            # For non-list defaults, use the first token after the key.
            value = tokens[1].strip()
            if value.replace(".", "", 1).isdigit():
                value_converted = float(value) if "." in value else int(value)
            else:
                value_converted = value
            defaults[key] = value_converted

    return defaults


def copy_and_mutate_update(
    optimization_targets: List[torch.Tensor],
    optimization_target_names: List[str],
    current_learning_rates: List[float],
    crystal_system: Optional[str],
    cry_atom_data: torch.Tensor,
    size: torch.Tensor,
    atomic_mask: torch.Tensor,
    ox_states_used_mask: torch.Tensor,
    site_ids: torch.Tensor,
    radii_tensor: torch.Tensor,
    init_coords: torch.Tensor,
    each_loss: torch.Tensor,
    gap_loss: torch.Tensor,
    ef_loss: torch.Tensor,
    tolerance_loss: torch.Tensor,
    top_ratio: float,
    abc_scaler,  # 型は ABC_Scaler
    mutation_noise: float,
    group_C_use_rate: float,
    atom_dist_init: bool,
    angle_optimization: bool,
    current_step: int,
    total_steps: int,
    Ef_criteria: float = -0.5,
    tolerance_criteria: float = 0.1,
    test_mode: bool = False,
):
    """Performs an evolutionary update (copy and mutate) on mini-batch samples,
    and returns the updated parameters along with new cry_atom_data and new_size.

    For each sample in the mini-batch (assumed along dimension 0), the update is performed as follows:
      - Group A: Samples with gap_loss==0 and ef_loss<=Ef_criteria (and for 'perovskite', tolerance_loss<=tolerance_criteria)
                 are kept unchanged.
      - Group B: Among the remaining samples, the top fraction (specified by top_ratio) with the lowest each_loss
                 are kept unchanged.
      - Group C: For the remaining samples, each sample is replaced by a copy of a randomly chosen sample from
                 Group A (or Group A∪Group B in non-test mode) with added noise (scaled by mutation_noise).

    For parameters whose shape can vary (i.e. "batch_dir_coords" and "atomic_distribution"),
    the global parameter tensor (shape [total_atoms, D]) is updated as follows:
      - Extract indices corresponding to the target sample (selected_indices) from cry_atom_data.
      - Split the global tensor into three parts: front, target block, and back.
      - For the chosen good sample, extract its block (good_indices), adjust the number of rows to match the target's
        atom count, add mutation noise to create mutated_block.
      - Reconstruct the global tensor by concatenating: front, mutated_block, back.
      - Also update cry_atom_data for the target block: assign the chosen sample's crystal ID.

    Finally, new_size is computed as a tensor of shape (original_batch_size,) where each element is the count of
    atoms assigned to that crystal ID in new_cry_atom_data.

    Args:
        optimization_targets (List[torch.Tensor]): List of parameter tensors. [scaled_batch_abc, scaled_batch_angle, batch_dir_coords, atomic_distribution, ox_mask_learnable_tensor_per_crystal]
        optimization_target_names (List[str]): List of names corresponding to the parameters. ['scaled_batch_abc', 'scaled_batch_angle', 'batch_dir_coords', 'atomic_distribution', 'ox_mask_learnable_tensor_per_crystal']
        current_learning_rates (List[float]): Learning rates for each group.
        crystal_system (Optional[str]): e.g. 'perovskite'; affects Group A criteria.
        cry_atom_data (torch.Tensor): 1D tensor mapping each row (atom) to a sample (crystal) ID.
        each_loss (torch.Tensor): 1D tensor of per-sample loss (shape: [batch_size]).
        gap_loss (torch.Tensor): 1D tensor of per-sample gap loss.
        ef_loss (torch.Tensor): 1D tensor of per-sample ef loss.
        tolerance_loss (torch.Tensor): 1D tensor of per-sample tolerance loss.
        top_ratio (float): Fraction of non-Group-A samples to designate as Group B.
        abc_scaler: An instance of ABC_Scaler (for updating batch_abc if needed).
        mutation_noise (float): Noise scale for mutation.
        group_C_use_rate (float): Probability threshold for skipping update in Group C.
        atom_dist_init (bool): Flag related to atomic distribution initialization.
        Ef_criteria (float, optional): Threshold for ef_loss for Group A. Defaults to -0.5.
        tolerance_criteria (float, optional): Threshold for tolerance_loss (for perovskite). Defaults to 0.1.
        test_mode (bool, optional): If True, use only Group A for mutation and set mutation_noise to 0. Defaults to True.

    Returns:
        Tuple[List[torch.Tensor], torch.Tensor, torch.Tensor]:
            - Updated list of parameter tensors.
            - new_cry_atom_data: Updated cry_atom_data tensor.
            - new_size: 1D tensor (length = original batch_size) where each element is the count of atoms for that crystal.
    """
    batch_size = each_loss.size(0)
    assert batch_size <= cry_atom_data.size(0), (
        "batch_size must be <= cry_atom_data size"
    )

    # Identify Group A:
    if crystal_system == "perovskite":
        group_A = (
            (gap_loss == 0)
            & (ef_loss <= Ef_criteria)
            & (tolerance_loss <= tolerance_criteria)
        ).nonzero(as_tuple=True)[0]
    else:
        group_A = ((gap_loss == 0) & (ef_loss <= Ef_criteria)).nonzero(as_tuple=True)[0]

    # Identify non-A indices (Group B and C)
    if crystal_system == "perovskite":
        non_A_mask = ~(
            (gap_loss == 0)
            & (ef_loss <= Ef_criteria)
            & (tolerance_loss <= tolerance_criteria)
        )
    else:
        non_A_mask = ~((gap_loss == 0) & (ef_loss <= Ef_criteria))

    non_A_indices = non_A_mask.nonzero(as_tuple=True)[0]

    # Among non-A, select top_ratio fraction with lowest each_loss as Group B.
    if non_A_indices.numel() > 0:
        non_A_losses = each_loss[non_A_indices]
        num_B = max(1, int(top_ratio * non_A_indices.numel()))
        _, sorted_indices = torch.sort(non_A_losses)
        group_B = non_A_indices[sorted_indices[:num_B]]
    else:
        group_B = torch.tensor([], dtype=torch.long, device=each_loss.device)

    # Group C: Non-A samples that are not in Group B.
    group_B_set = set(group_B.tolist())
    group_C = [i for i in non_A_indices.tolist() if i not in group_B_set]

    if test_mode:
        print(f"<info> test mode in copy_and_mutate_update")

        print(type(group_A), group_A)
        print(type(group_B), group_B)
        print(type(group_C), group_C)

        group_C = set(group_B.tolist() + group_C)
        group_B = torch.tensor([])
        print(
            f"Group A: {group_A.tolist()}, Group B: {group_B.tolist()}, Group C: {group_C}"
        )
        print(f"Test choice group (A and B) : {(group_A.tolist() + group_B.tolist())}")
        mutation_noise = 0.0
        group_C_use_rate = -1.0
        atom_dist_init = False

    # new_cry_atom_data: Initialize with the original cry_atom_data.
    new_cry_atom_data = cry_atom_data.clone()
    new_size = size.clone()

    # For each sample i in Group C, perform update.
    with torch.no_grad():
        for i in group_C:
            # Randomly decide whether to skip the update for this sample.
            if random.random() < group_C_use_rate:
                continue
            # Choose a good crystal from Group A or Group B.
            chosen_good_crystal = int(
                random.choice((group_A.tolist() + group_B.tolist()))
            )

            # サンプル i の更新開始前に、インデックスを事前計算して保存する
            orig_selected_indices = (new_cry_atom_data == i).nonzero(as_tuple=True)[0]
            orig_good_indices = (new_cry_atom_data == chosen_good_crystal).nonzero(
                as_tuple=True
            )[0]

            # Update each parameter in optimization_targets.
            for param, param_name in zip(
                optimization_targets, optimization_target_names
            ):
                if param_name in [
                    "scaled_batch_abc",
                    "scaled_batch_angle",
                    "ox_mask_learnable_tensor_per_crystal",
                ]:
                    # Update fixed parameters (assumed shape [batch_size, ...])
                    if (not angle_optimization) and param_name == "scaled_batch_angle":
                        pass
                    elif param_name == "scaled_batch_abc":
                        noise = (
                            torch.randn_like(param[chosen_good_crystal])
                            * mutation_noise
                        )
                        param[i].copy_(param[chosen_good_crystal].clone() + noise)
                        if test_mode:
                            print(f"i: {i}, chosen_good_crystal: {chosen_good_crystal}")
                            print(
                                f"(Before) abc_scaler.init_batch_abc[i]: {abc_scaler.init_batch_abc[i]}"
                            )
                        abc_scaler.init_batch_abc[i] = abc_scaler.init_batch_abc[
                            chosen_good_crystal
                        ]
                        # あとで消す
                        if test_mode:
                            print(
                                f"(After) abc_scaler.init_batch_abc[i]: {abc_scaler.init_batch_abc[i]}"
                            )
                            print(
                                f"abc_scaler.init_batch_abc[chosen_good_crystal]: {abc_scaler.init_batch_abc[chosen_good_crystal]}"
                            )

                    else:
                        noise = (
                            torch.randn_like(param[chosen_good_crystal])
                            * mutation_noise
                        )
                        param[i].copy_(param[chosen_good_crystal].clone() + noise)
                else:  # param_name in ['batch_dir_coords', 'atomic_distribution']:
                    # For variable shape parameters:
                    # param shape: [total_atoms, D]
                    # Get indices for sample i and for chosen_good_crystal.
                    start_i = orig_selected_indices[0].item()
                    end_i = orig_selected_indices[-1].item() + 1
                    front_tensor = param[:start_i]
                    back_tensor = param[end_i:]

                    # If the number of atoms differs, adjust the size of good_block.
                    if param_name == "atomic_distribution" and atom_dist_init:
                        # re-initialize atomic_distribution using uniform distribution
                        good_block = (
                            torch.ones_like(param[orig_good_indices].clone()) * i
                        )
                    else:
                        # Determine the target count for sample i.
                        good_block = param[orig_good_indices].clone()

                    mutated_block = (
                        good_block + torch.randn_like(good_block) * mutation_noise
                    )
                    new_param = torch.cat(
                        [front_tensor, mutated_block, back_tensor], dim=0
                    )
                    param.data = new_param.clone()

                    if param_name == "atomic_distribution":
                        # update atomic_mask and ox_states_used_mask
                        ## atomic_mask update:
                        front_tensor = atomic_mask[:start_i]
                        back_tensor = atomic_mask[end_i:]
                        good_block = atomic_mask[orig_good_indices].clone()
                        atomic_mask = torch.cat(
                            [front_tensor, good_block, back_tensor], dim=0
                        )
                        ## ox_states_used_mask update:
                        front_tensor = ox_states_used_mask[:start_i]
                        back_tensor = ox_states_used_mask[end_i:]
                        good_block = ox_states_used_mask[orig_good_indices].clone()
                        ox_states_used_mask = torch.cat(
                            [front_tensor, good_block, back_tensor], dim=0
                        )

                        # update init_coords
                        front_tensor = init_coords[:start_i]
                        back_tensor = init_coords[end_i:]
                        good_block = init_coords[orig_good_indices].clone()
                        init_coords = torch.cat(
                            [front_tensor, good_block, back_tensor], dim=0
                        )

                        # Update site_ids for the target block.
                        front_tensor = site_ids[:start_i]
                        back_tensor = site_ids[end_i:]
                        good_block = site_ids[orig_good_indices].clone()
                        site_ids = torch.cat(
                            [front_tensor, good_block, back_tensor], dim=0
                        )

                        # Update radii_tensor fofr the target block.
                        front_tensor = radii_tensor[:start_i]
                        back_tensor = radii_tensor[end_i:]
                        good_block = radii_tensor[orig_good_indices].clone()
                        radii_tensor = torch.cat(
                            [front_tensor, good_block, back_tensor], dim=0
                        )

                        # Update cry_atom_data for the target block.
                        orig_selected_indices = (new_cry_atom_data == i).nonzero(
                            as_tuple=True
                        )[0]
                        orig_good_indices = (
                            new_cry_atom_data == chosen_good_crystal
                        ).nonzero(as_tuple=True)[0]
                        start_i = orig_selected_indices[0].item()
                        end_i = orig_selected_indices[-1].item() + 1
                        # Update cry_atom_data for the target block.
                        # cry_atom_data update:
                        front_tensor = new_cry_atom_data[:start_i]
                        back_tensor = new_cry_atom_data[end_i:]
                        mutated_block = torch.ones_like(good_block) * i
                        mutated_block = torch.full(
                            (good_block.shape[0],),
                            i,
                            dtype=cry_atom_data.dtype,
                            device=cry_atom_data.device,
                        )
                        new_cry_atom_data = torch.cat(
                            [front_tensor, mutated_block, back_tensor], dim=0
                        )
                        # size update:
                        new_size[i] = mutated_block.numel()

    # Update the parameters in optimization_targets with the new values.
    new_optimization_targets = []
    with torch.no_grad():
        for param in optimization_targets:
            # params to learnable parameters
            new_optimization_targets.append(
                torch.nn.Parameter(param.data.clone().detach(), requires_grad=True)
            )

    # Update abc_scaler base length.
    abc_scaler.init_batch_abc
    abc_scaler.update_base_length()

    # optimizer and scheduler
    (
        optimizer_lattice,
        optimizer_atom,
        optimizer_coords,
        scheduler_lattice,
        scheduler_atom,
        scheduler_coords,
    ) = define_optimizer_and_scheduler(
        lattice_lr=current_learning_rates[0],
        atom_lr=current_learning_rates[1],
        coords_lr=current_learning_rates[2],
        ox_mask_learnable_tensor_per_crystal=new_optimization_targets[
            4
        ],  # ox_mask_learnable_tensor_per_crystal,
        atomic_distribution=new_optimization_targets[3],  # atomic_distribution,
        scaled_batch_abc=new_optimization_targets[0],  # scaled_batch_abc,
        scaled_batch_angle=new_optimization_targets[1],  # scaled_batch_angle,
        batch_dir_coords=new_optimization_targets[2],  # batch_dir_coords,
        angle_optimization=angle_optimization,
        num_steps=total_steps - current_step,
        lattice_cycle=1,
        atom_cycle=1,
        coords_cycle=1,
    )
    optimizers_and_schedulers = (
        optimizer_lattice,
        optimizer_atom,
        optimizer_coords,
        scheduler_lattice,
        scheduler_atom,
        scheduler_coords,
    )

    return (
        new_optimization_targets,
        optimizers_and_schedulers,
        new_cry_atom_data,
        new_size,
        atomic_mask,
        ox_states_used_mask,
        site_ids,
        radii_tensor,
        init_coords,
        abc_scaler,
    )
