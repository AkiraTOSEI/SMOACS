from typing import Tuple

import numpy as np
import torch
from jarvis.core.atoms import Atoms

from .utils import temperature_softmax


def compute_lattice_vectors(batch_abc: torch.Tensor, batch_angle: torch.Tensor) -> torch.Tensor:
    """
    Computes the lattice vectors for a batch of crystal lattices based on their side lengths and angles.

    This function calculates the lattice vectors for each set of crystal lattice parameters in the batch. The lattice
    vectors are computed using the side lengths (a, b, c) and angles (alpha, beta, gamma) provided for each crystal lattice.

    Args:
        batch_abc (torch.Tensor): A tensor containing the side lengths of the crystal lattices in the batch.
                                  The shape is (B, 3), where B is the batch size.
        batch_angle (torch.Tensor): A tensor containing the angles (in degrees) of the crystal lattices in the batch.
                                    The shape is (B, 3), where B is the batch size. Angles are in the order (alpha, beta, gamma).

    Returns:
        torch.Tensor: A tensor of shape (B, 3, 3) containing the lattice vectors for each crystal lattice in the batch.
    """
    # convert angles to radians
    angle_rad = batch_angle * np.pi / 180.0
    
    # cosines and sines of the angles
    cos_alpha = torch.cos(angle_rad[:, 0])
    cos_beta = torch.cos(angle_rad[:, 1])
    cos_gamma = torch.cos(angle_rad[:, 2])
    sin_gamma = torch.sin(angle_rad[:, 2])
    
    # get the batch size B
    B = batch_abc.shape[0]
    
    # initialize the lattice vector tensor
    lattice_vec = torch.zeros((B, 3, 3), dtype=batch_abc.dtype, device=batch_abc.device)
    
    # calculate lattice vectors
    lattice_vec[:, 0, 0] = batch_abc[:, 0]  # a
    lattice_vec[:, 1, 0] = batch_abc[:, 1] * cos_gamma  # b*cos(gamma)
    lattice_vec[:, 1, 1] = batch_abc[:, 1] * sin_gamma  # b*sin(gamma)
    lattice_vec[:, 2, 0] = batch_abc[:, 2] * cos_beta  # c*cos(beta)
    lattice_vec[:, 2, 1] = batch_abc[:, 2] * (cos_alpha - cos_beta * cos_gamma) / sin_gamma
    lattice_vec[:, 2, 2] = batch_abc[:, 2] * torch.sqrt(1 - cos_beta**2 - ((cos_alpha - cos_beta * cos_gamma) / sin_gamma)**2)
    
    return lattice_vec


def direct_to_cartesian(dir_coords: torch.Tensor, lattice_vectors: torch.Tensor) -> torch.Tensor:
    """
    Convert direct coordinates to cartesian coordinates using lattice vectors.

    The conversion is done by matrix multiplying the direct coordinates with the lattice vectors.

    Args:
        dir_coords (torch.Tensor): The direct coordinates, with shape (N, 3), where N is the number
                                   of atoms in the structure.
        lattice_vectors (torch.Tensor): The lattice vectors, with shape (3, 3).

    Returns:
        torch.Tensor: The cartesian coordinates, with the same shape as dir_coords.
    """
    return torch.matmul(dir_coords, lattice_vectors)



def direct_to_cartesian_batch(dir_coords: torch.Tensor, lattice_vectors: torch.Tensor) -> torch.Tensor:
    """
    Convert direct coordinates to cartesian coordinates using lattice vectors.

    The conversion is done by matrix multiplying the direct coordinates with the lattice vectors.

    Args:
        dir_coords (torch.Tensor): The direct coordinates, with shape (N, 3), where N is the number
                                   of atoms in the structure.
        lattice_vectors (torch.Tensor): The lattice vectors, with shape (N, 3, 3).

    Returns:
        torch.Tensor: The cartesian coordinates, with the same shape as dir_coords.
    """
    _bool1 = len(dir_coords.shape) == 2
    _bool2 = len(lattice_vectors.shape) == 3
    _bool3 = dir_coords.shape[1] == 3
    if not (_bool1 and _bool2 and _bool3):
        print("dir_coords.shape:",dir_coords.shape)
        print("lattice_vectors.shape:",lattice_vectors.shape)
        raise Exception("The shapes of dir_coords and lattice_vectors are not compatible.")

    
    cartesian_coords = torch.bmm(dir_coords.unsqueeze(1), lattice_vectors)
    return cartesian_coords.squeeze(1)

def masking_atomic_distribution(
    atomic_distribution: torch.Tensor, 
    atomic_mask: torch.Tensor,
    ox_states_used_mask: torch.Tensor,
    stacked_learnable_ox_weight : torch.Tensor,
    softmax_temp: float,
    eps: float = 1e-6
    ):
    """
    Mask atomic distribution with atomic mask.
    Args:
        atomic_distribution: atomic distribution. shape = (N, num_type_of_atoms)
        atomic_mask: atomic mask. shape = (N, num_type_of_atoms) or (N, num_type_of_atoms, max_ox_state)
        ox_states_used_mask: mask for used oxidation states. a learnable parameter. shape = (N, max_ox_state)
        stacked_learnable_ox_weight: learnable weight for oxidation states. shape = (N, max_ox_state)
        softmax_temp: temperature for softmax
    
    """
    assert atomic_distribution.shape[0] == atomic_mask.shape[0] == ox_states_used_mask.shape[0] == stacked_learnable_ox_weight.shape[0]
    assert atomic_distribution.shape[1] == atomic_mask.shape[1]
    assert ox_states_used_mask.shape[1] == atomic_mask.shape[2] == stacked_learnable_ox_weight.shape[1]
    # multiple masks for each atomic feature
    # atomic_distribution: (N, num_type_of_atoms)
    # atomic_mask: (N, num_type_of_atoms, max_ox_state)
    
    atomic_distribution = torch.clip(atomic_distribution, min=eps) # (N, num_type_of_atoms), clip to avoid minus value for selected atoms
    stacked_learnable_ox_weight = torch.clip(stacked_learnable_ox_weight, min=eps) # (N, max_ox_state), clip to avoid minus value for selected oxidation states
    
    masked_atomic_distribution = atomic_distribution.unsqueeze(-1) * atomic_mask # (N, num_type_of_atoms, max_ox_state)
    masked_atomic_distribution = temperature_softmax(masked_atomic_distribution, temperature=softmax_temp, dim=1) # (N, num_type_of_atoms, max_ox_state)
    sharpened_ox_mask = temperature_softmax(ox_states_used_mask*stacked_learnable_ox_weight, temperature=softmax_temp, dim=1).unsqueeze(1) # (N, 1, max_ox_state)
    masked_atomic_distribution = torch.sum(masked_atomic_distribution * sharpened_ox_mask, dim=-1) # (N, num_type_of_atoms)
    masked_atomic_distribution = temperature_softmax(masked_atomic_distribution, temperature=softmax_temp, dim=1) # (N, num_type_of_atoms)

    return masked_atomic_distribution, sharpened_ox_mask.squeeze(1)



def compute_abc_angle(batch_lattice_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the side lengths and angles of crystal lattices from their lattice vectors for a batch.

    This function calculates the side lengths (a, b, c) and angles (alpha, beta, gamma) of the crystal lattices
    based on the provided lattice vectors for each item in the batch. The angles are computed in degrees.

    Args:
        batch_lattice_vec (torch.Tensor): A tensor containing the lattice vectors for the crystal lattices in the batch.
                                          The shape is (B, 3, 3), where B is the batch size.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                                           - The first tensor contains the side lengths (a, b, c) of the crystal lattices in the batch, with shape (B, 3).
                                           - The second tensor contains the angles (alpha, beta, gamma) in degrees of the crystal lattices in the batch, with shape (B, 3).
    """
    B = batch_lattice_vec.shape[0]
    batch_abc = torch.zeros((B, 3), dtype=batch_lattice_vec.dtype, device=batch_lattice_vec.device)
    batch_angle = torch.zeros((B, 3), dtype=batch_lattice_vec.dtype, device=batch_lattice_vec.device)
    
    # 辺の長さを計算
    batch_abc[:, 0] = torch.norm(batch_lattice_vec[:, 0, :], dim=1)
    batch_abc[:, 1] = torch.norm(batch_lattice_vec[:, 1, :], dim=1)
    batch_abc[:, 2] = torch.norm(batch_lattice_vec[:, 2, :], dim=1)
    
    # 成す角を計算
    # alpha = angle between b and c
    batch_angle[:, 0] = torch.acos(torch.sum(batch_lattice_vec[:, 1, :] * batch_lattice_vec[:, 2, :], dim=1) / (batch_abc[:, 1] * batch_abc[:, 2]))
    # beta = angle between a and c
    batch_angle[:, 1] = torch.acos(torch.sum(batch_lattice_vec[:, 0, :] * batch_lattice_vec[:, 2, :], dim=1) / (batch_abc[:, 0] * batch_abc[:, 2]))
    # gamma = angle between a and b
    batch_angle[:, 2] = torch.acos(torch.sum(batch_lattice_vec[:, 0, :] * batch_lattice_vec[:, 1, :], dim=1) / (batch_abc[:, 0] * batch_abc[:, 1]))
    
    # ラジアンから度へ変換
    batch_angle = torch.rad2deg(batch_angle)
    
    return batch_abc, batch_angle


def check_nearest_neighbor(atoms_data: Atoms, r: float = 0.5, verbose: int = 0) -> bool:
    """
    Checks for any atom pairs within a specified distance in a supercell.

    Args:
        atoms_data (Atoms): An instance of Atoms containing atomic data.
        r (float): Radius to check for neighbor proximity. Default is 0.5.
        verbose (int): Verbosity level. A non-zero value prints detailed bond information.

    Returns:
        bool: True if no atom pairs are found within the specified distance, False otherwise.
    """
    try:
        # Generate supercell
        sc_atoms_data = atoms_data.make_supercell_matrix([2, 2, 2])
        coords = sc_atoms_data.coords  # Assuming this now correctly handles supercell coordinates

        # Compute distance matrix efficiently
        from scipy.spatial import distance_matrix
        dist_matrix = distance_matrix(coords, coords)
        np.fill_diagonal(dist_matrix, np.inf)

        # get original atom ids
        atom_ids = np.array([[atoms_data.elements.index(e)]*(2*2*2) for e in atoms_data.elements]).reshape(-1)

        # Identify pairs with distance less than r
        close_pairs = np.where(dist_matrix < r)

        if verbose > 0:
            for _row, _col in zip(*close_pairs):
                coord0 = coords[_row]
                coord1 = coords[_col]
                print(f"bond between: Atom-{atom_ids[_row]} {sc_atoms_data.elements[_row]}({coord0[0]:.3f},{coord0[1]:.3f},{coord0[2]:.3f}) ----- Atom-{atom_ids[_col]} {sc_atoms_data.elements[_col]}({coord1[0]:.3f},{coord1[1]:.3f},{coord1[2]:.3f}), distance:{dist_matrix[_row, _col]:.3f}")
                original_coord0 = atoms_data.coords[atom_ids[_row]]
                original_coord1 = atoms_data.coords[atom_ids[_col]]
                print(f"original coord:  Atom-{atom_ids[_row]} {atoms_data.elements[atom_ids[_row]]}({original_coord0[0]:.3f},{original_coord0[1]:.3f},{original_coord0[2]:.3f}) ----- Atom-{atom_ids[_col]} {atoms_data.elements[atom_ids[_col]]}({original_coord1[0]:.3f},{original_coord1[1]:.3f},{original_coord1[2]:.3f})")
                print("")

        return not any(dist_matrix.flatten() < r)

    except Exception as e:
        print(f"An error occurred: {e}")
        return False