from typing import Optional, Tuple
import torch

from src.utils.mask_utils import (
    masking_atomic_distribution,
    stacking_learnable_oxsides_mask,
)
from src.utils.nn_utils import temperature_softmax

def compute_normalized_dist_and_ox_mask(
    atomic_distribution: torch.Tensor,
    atomic_mask: torch.Tensor,
    ox_states_used_mask: torch.Tensor,
    temperature: float,
    size: torch.Tensor,
    ox_mask_learnable_tensor_per_crystal: Optional[list] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized atomic distribution and sharpened oxidation state mask.

    If a learnable oxidation mask is provided, it is stacked and used for masking
    the atomic distribution. Otherwise, default behavior uses temperature-softmax 
    without masking.

    Args:
        atomic_distribution (torch.Tensor): Raw atomic distribution logits.
        atomic_mask (torch.Tensor): Binary mask indicating valid atomic positions.
        ox_states_used_mask (torch.Tensor): Mask indicating allowed oxidation states.
        temperature (float): Temperature parameter for softmax.
        size (torch.Tensor): Size tensor indicating number of atoms per crystal.
        ox_mask_learnable_tensor_per_crystal (Optional[list], optional): 
            List of learnable oxidation state tensors per crystal. If None,
            masking is skipped. Defaults to None.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: 
            normalized_dist: Normalized atomic distribution (after softmax or masking).
            sharpened_ox_mask: Masked oxidation state tensor or NaN-filled placeholder.
    """
    if ox_mask_learnable_tensor_per_crystal is not None:
        stacked_learnable_ox_weight = stacking_learnable_oxsides_mask(
            ox_mask_learnable_tensor_per_crystal, size
        )
        normalized_dist, sharpened_ox_mask = masking_atomic_distribution(
            atomic_distribution,
            atomic_mask,
            ox_states_used_mask,
            stacked_learnable_ox_weight,
            temperature,
        )
    else:
        sharpened_ox_mask = torch.zeros_like(ox_states_used_mask) * torch.nan
        normalized_dist = temperature_softmax(
            atomic_distribution, temperature=temperature
        )

    return normalized_dist, sharpened_ox_mask