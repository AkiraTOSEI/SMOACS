from typing import Dict, Optional, Tuple

import torch


def atom_dist_to_features(atomic_distribution, atom_feat_matrix):
    """
    Convert atomic distribution to atomic features.
    """
    atomic_features = torch.matmul(atomic_distribution, atom_feat_matrix)
    return atomic_features


def calculate_onehot(
    output_dict: Dict[str, torch.Tensor],
    atom_feat_matrix: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert probabilistic atomic distributions to one-hot vectors.

    This function converts soft atomic distributions and oxidation masks into
    one-hot encoded tensors and optionally computes atomic features.

    Args:
        output_dict (Dict[str, torch.Tensor]): Dictionary with 'normalized_dist' and 'sharpened_ox_mask'.
        atom_feat_matrix (Optional[torch.Tensor]): Atomic feature matrix. Shape: (num_types, feature_dim)

    Returns:
        Tuple[torch.Tensor, ...]:
            - max_val: Maximum value of the atomic distribution for each atom.
            - onehot_x: One-hot encoded atomic species. Shape: same as normalized_dist.
            - onehot_atom_feat: Atomic feature vectors from one-hot encoding. Shape: (N_atoms, feature_dim)
            - onehot_ox_mask: One-hot encoded oxidation state mask. Shape: (N_atoms, max_ox)
    """
    max_val, max_index = torch.max(output_dict["normalized_dist"], dim=1)
    onehot_x = torch.zeros(
        output_dict["normalized_dist"].shape,
        device=output_dict["normalized_dist"].device,
    )
    onehot_x[torch.arange(output_dict["normalized_dist"].shape[0]), max_index] = 1
    assert (torch.max(onehot_x, dim=1)[1] == max_index).all()

    _, max_ox_index = torch.max(output_dict["sharpened_ox_mask"], dim=1)
    onehot_ox_mask = torch.zeros_like(output_dict["sharpened_ox_mask"])
    onehot_ox_mask[
        torch.arange(output_dict["sharpened_ox_mask"].shape[0]), max_ox_index
    ] = 1
    assert (torch.max(onehot_ox_mask, dim=1)[1] == max_ox_index).all()

    if atom_feat_matrix is not None:
        onehot_atom_feat = torch.matmul(onehot_x, atom_feat_matrix)
    else:
        onehot_atom_feat = torch.zeros_like(onehot_x) * torch.nan
    return max_val, onehot_x, onehot_atom_feat, onehot_ox_mask
