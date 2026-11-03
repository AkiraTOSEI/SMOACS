from typing import Optional, Tuple
import torch

from src.utils.mask_utils import (
    masking_atomic_distribution,
    stacking_learnable_oxsides_mask,
)
from src.utils.nn_utils import temperature_softmax


def create_element_mask(
    num_elements: int,
    max_atomic_number: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Create a SMOACS-column mask for atomic numbers up to ``max_atomic_number``."""
    if not 0 < max_atomic_number <= num_elements:
        raise ValueError(
            "max_atomic_number must be in [1, num_elements], "
            f"got max_atomic_number={max_atomic_number}, num_elements={num_elements}"
        )
    mask = torch.zeros(num_elements, device=device, dtype=dtype)
    mask[:max_atomic_number] = 1
    return mask


def apply_element_mask(
    atomic_distribution: torch.Tensor,
    element_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Hard-mask an atomic distribution and renormalize each row."""
    if element_mask is None:
        return atomic_distribution
    if atomic_distribution.ndim != 2:
        raise ValueError(
            "atomic_distribution must be two-dimensional, "
            f"got shape {tuple(atomic_distribution.shape)}"
        )
    if element_mask.ndim != 1 or element_mask.shape[0] != atomic_distribution.shape[1]:
        raise ValueError(
            "element_mask must have shape (num_elements,), "
            f"got {tuple(element_mask.shape)} for {tuple(atomic_distribution.shape)}"
        )

    mask = element_mask.to(
        device=atomic_distribution.device, dtype=atomic_distribution.dtype
    )
    masked_distribution = atomic_distribution * mask.unsqueeze(0)
    row_sum = masked_distribution.sum(dim=1, keepdim=True)
    if not torch.isfinite(row_sum).all():
        raise ValueError("Masked atomic distribution contains non-finite row sums")
    if torch.any(row_sum <= 0):
        raise ValueError(
            "No MatterSim-supported element remains in an atomic distribution row"
        )
    return masked_distribution / row_sum


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
