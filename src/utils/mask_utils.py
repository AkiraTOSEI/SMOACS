import torch

from src.utils.nn_utils import temperature_softmax


def create_learnable_oxides_mask(mini_batch_inputs_dict):
    """
    a function to create learnable mask tensor for oxides used in the model
    """
    _ox_mask_learnable_tensor_per_crystal = []
    total_idx = 0

    for idx in mini_batch_inputs_dict["size"]:
        total_idx += idx
        _ox_mask_learnable_tensor_per_crystal.append(
            mini_batch_inputs_dict["ox_states_used_mask"][total_idx - 1]
        )
    # ox_mask_learnable_tensor_per_crystal = torch.nn.Parameter(torch.stack(_ox_mask_learnable_tensor_per_crystal))
    ox_mask_learnable_tensor_per_crystal = torch.stack(
        _ox_mask_learnable_tensor_per_crystal
    )

    # test
    stacked_learnable_ox_weight = stacking_learnable_oxsides_mask(
        ox_mask_learnable_tensor_per_crystal, mini_batch_inputs_dict["size"]
    )
    # print("stacked_learnable_ox_weight", stacked_learnable_ox_weight, stacked_learnable_ox_weight.shape)
    # print("ox_states_used_mask", mini_batch_inputs_dict['ox_states_used_mask'], mini_batch_inputs_dict['ox_states_used_mask'].shape)
    assert (
        (
            stacked_learnable_ox_weight.detach()
            == mini_batch_inputs_dict["ox_states_used_mask"]
        )
        .all()
        .detach()
        .cpu()
        .numpy()
    )

    return ox_mask_learnable_tensor_per_crystal


def stacking_learnable_oxsides_mask(
    ox_mask_learnable_tensor_per_crystal: torch.Tensor, size: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        ox_mask_learnable_tensor_per_crystal: torch.Tensor, shape=(num_crystals, num_oxides)
    """
    return torch.concat(
        [
            torch.stack([ox_mask_learnable_tensor_per_crystal[idx]] * size)
            for idx, size in enumerate(size)
        ],
        dim=0,
    )


import torch


def create_learnable_oxides_mask(mini_batch_inputs_dict):
    """
    a function to create learnable mask tensor for oxides used in the model
    """
    _ox_mask_learnable_tensor_per_crystal = []
    total_idx = 0

    for idx in mini_batch_inputs_dict["size"]:
        total_idx += idx
        _ox_mask_learnable_tensor_per_crystal.append(
            mini_batch_inputs_dict["ox_states_used_mask"][total_idx - 1]
        )
    # ox_mask_learnable_tensor_per_crystal = torch.nn.Parameter(torch.stack(_ox_mask_learnable_tensor_per_crystal))
    ox_mask_learnable_tensor_per_crystal = torch.stack(
        _ox_mask_learnable_tensor_per_crystal
    )

    # test
    stacked_learnable_ox_weight = stacking_learnable_oxsides_mask(
        ox_mask_learnable_tensor_per_crystal, mini_batch_inputs_dict["size"]
    )
    # print("stacked_learnable_ox_weight", stacked_learnable_ox_weight, stacked_learnable_ox_weight.shape)
    # print("ox_states_used_mask", mini_batch_inputs_dict['ox_states_used_mask'], mini_batch_inputs_dict['ox_states_used_mask'].shape)
    assert (
        (
            stacked_learnable_ox_weight.detach()
            == mini_batch_inputs_dict["ox_states_used_mask"]
        )
        .all()
        .detach()
        .cpu()
        .numpy()
    )

    return ox_mask_learnable_tensor_per_crystal


def stacking_learnable_oxsides_mask(
    ox_mask_learnable_tensor_per_crystal: torch.Tensor, size: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        ox_mask_learnable_tensor_per_crystal: torch.Tensor, shape=(num_crystals, num_oxides)
    """
    return torch.concat(
        [
            torch.stack([ox_mask_learnable_tensor_per_crystal[idx]] * size)
            for idx, size in enumerate(size)
        ],
        dim=0,
    )


def masking_atomic_distribution(
    atomic_distribution: torch.Tensor,
    atomic_mask: torch.Tensor,
    ox_states_used_mask: torch.Tensor,
    stacked_learnable_ox_weight: torch.Tensor,
    softmax_temp: float,
    eps: float = 1e-6,
):
    """
    Apply atomic and oxidation-state masks to the atomic distribution with temperature-scaled softmax.

    This function takes an initial atomic distribution and applies:
    1. Element-wise masking using a 3D atomic mask tensor.
    2. Softmax sharpening across atomic types.
    3. Oxidation-state-specific masking using a learnable softmax-weighted oxidation mask.
    The result is a filtered and normalized atomic distribution suitable for structure optimization.

    Args:
        atomic_distribution (torch.Tensor):
            Initial atomic type distribution tensor of shape (N, num_atom_types).
        atomic_mask (torch.Tensor):
            Element-wise atomic mask tensor of shape (N, num_atom_types, max_ox_states), used to gate allowed atom-oxidation combinations.
        ox_states_used_mask (torch.Tensor):
            Binary mask (learnable) indicating which oxidation states are valid. Shape: (N, max_ox_states).
        stacked_learnable_ox_weight (torch.Tensor):
            Learnable weight for oxidation states. Shape: (N, max_ox_states).
        softmax_temp (float):
            Temperature parameter for softmax. Controls sharpness of output distribution.
        eps (float, optional):
            Small value added for numerical stability to avoid negative or zero values. Default is 1e-6.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - masked_atomic_distribution (torch.Tensor):
                Final masked and normalized atomic distribution. Shape: (N, num_atom_types).
            - sharpened_ox_mask (torch.Tensor):
                Normalized oxidation mask used during weighting. Shape: (N, max_ox_states).
    """
    assert (
        atomic_distribution.shape[0]
        == atomic_mask.shape[0]
        == ox_states_used_mask.shape[0]
        == stacked_learnable_ox_weight.shape[0]
    )
    assert atomic_distribution.shape[1] == atomic_mask.shape[1]
    assert (
        ox_states_used_mask.shape[1]
        == atomic_mask.shape[2]
        == stacked_learnable_ox_weight.shape[1]
    )
    # multiple masks for each atomic feature
    # atomic_distribution: (N, num_type_of_atoms)
    # atomic_mask: (N, num_type_of_atoms, max_ox_state)

    atomic_distribution = torch.clip(
        atomic_distribution, min=eps
    )  # (N, num_type_of_atoms), clip to avoid minus value for selected atoms
    stacked_learnable_ox_weight = torch.clip(
        stacked_learnable_ox_weight, min=eps
    )  # (N, max_ox_state), clip to avoid minus value for selected oxidation states

    masked_atomic_distribution = (
        atomic_distribution.unsqueeze(-1) * atomic_mask
    )  # (N, num_type_of_atoms, max_ox_state)
    masked_atomic_distribution = temperature_softmax(
        masked_atomic_distribution, temperature=softmax_temp, dim=1
    )  # (N, num_type_of_atoms, max_ox_state)
    sharpened_ox_mask = temperature_softmax(
        ox_states_used_mask * stacked_learnable_ox_weight,
        temperature=softmax_temp,
        dim=1,
    ).unsqueeze(1)  # (N, 1, max_ox_state)
    masked_atomic_distribution = torch.sum(
        masked_atomic_distribution * sharpened_ox_mask, dim=-1
    )  # (N, num_type_of_atoms)
    masked_atomic_distribution = temperature_softmax(
        masked_atomic_distribution, temperature=softmax_temp, dim=1
    )  # (N, num_type_of_atoms)

    return masked_atomic_distribution, sharpened_ox_mask.squeeze(1)
