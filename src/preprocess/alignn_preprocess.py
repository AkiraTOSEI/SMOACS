from typing import Optional, Tuple

import torch

from src.graph.graph_utils import (
    compute_bondlength,
    create_batch_lattice_vectors,
)
from src.utils.coord import calculate_bond_cosine, compute_lattice_vectors
from src.utils.feature_utils import atom_dist_to_features

from src.utils.scalers import ABC_Scaler, Angle_Scaler
from src.preprocess.prepro_utils import compute_normalized_dist_and_ox_mask
from src.spg_wp.lattice_params_control import fix_special_lattice_angles, enforce_equal_lattice_lengths
from src.spg_wp.wyckoff_expander import WyckoffExpander


def ALIGNN_preprocess(
        scaled_batch_abc:torch.Tensor,
        scaled_batch_angle:torch.Tensor,
        batch_dir_coords:torch.Tensor,
        atomic_distribution:torch.Tensor,
        batch_dst_ids:torch.Tensor, 
        batch_src_ids:torch.Tensor, 
        batch_displace:torch.Tensor,
        z_src_ids:torch.Tensor,
        z_dst_ids:torch.Tensor,
        size:torch.Tensor, # actual number of atoms in each crystal
        wp_size:torch.Tensor, # number of Wyckoff positions in each crystal
        site_ids:torch.Tensor,
        init_coords:torch.Tensor,
        temperature:float,
        abc_scaler:ABC_Scaler,
        angle_scaler:Angle_Scaler,
        num_edges:int,
        n_nodes:int,
        atom_feat_matrix:torch.Tensor,
        limit_coords_displacement: Optional[float],
        adding_noise_scale: Optional[float],
        abc_range: Tuple[float, float],
        angle_range: Tuple[float, float],
        wyckoff_expander: Optional[WyckoffExpander] = None,
        atomic_mask: Optional[torch.Tensor] = None,
        ox_states_used_mask: Optional[torch.Tensor] = None,
        ox_mask_learnable_tensor_per_crystal: Optional[torch.Tensor] = None,
        value_round: Optional[int] = None,
        atomic_dist_Z_more_than_86: bool = True,
        abc_equal_eps: float = 1e-5,
        angle_fix_eps: float = 1e-3,

   ) -> Tuple[torch.Tensor, ...]:
    """
    Preprocess input tensors before feeding them into the ALIGNN model.

    This function performs coordinate normalization, lattice vector reconstruction,
    atomic feature computation, and bond geometry calculations (lengths and angles),
    along with optional masking and temperature-scaled softmax.

    Args:
        scaled_batch_abc (torch.Tensor): Normalized lattice lengths. Shape: (N, 3)
        scaled_batch_angle (torch.Tensor): Normalized lattice angles. Shape: (N, 3)
        batch_dir_coords (torch.Tensor): Direct coordinates of atoms. Shape: (total_atoms, 3)
        atomic_distribution (torch.Tensor): Raw atomic distributions. Shape: (total_atoms, 98)
        batch_dst_ids (torch.Tensor): Destination indices of edges. Shape: (num_edges,)
        batch_src_ids (torch.Tensor): Source indices of edges. Shape: (num_edges,)
        batch_displace (torch.Tensor): Displacement vectors for periodicity. Shape: (num_edges, 3)
        z_src_ids (torch.Tensor): Source indices for angle computation. Shape: (num_angles,)
        z_dst_ids (torch.Tensor): Destination indices for angle computation. Shape: (num_angles,)
        size (torch.Tensor): Number of atoms in each structure. Shape: (N,)
        init_coords (torch.Tensor): Initial direct coordinates for clipping. Shape: (total_atoms, 3)
        temperature (float): Temperature parameter for softmax.
        abc_scaler (ABC_Scaler): Scaler object to denormalize abc.
        angle_scaler (Angle_Scaler): Scaler object to denormalize angles.
        num_edges (int): Total number of edges (bonds).
        atom_feat_matrix (torch.Tensor): Feature matrix for elements. Shape: (98, feat_dim)
        limit_coords_displacement (Optional[float]): Max allowed shift from init_coords.
        adding_noise_scale (Optional[float]): Magnitude of Gaussian noise to add.
        abc_range (Tuple[float, float]): Min and max clamp values for abc.
        angle_range (Tuple[float, float]): Min and max clamp values for angles.
        atomic_mask (Optional[torch.Tensor]): Atomic-level mask for valid elements.
        ox_states_used_mask (Optional[torch.Tensor]): Binary mask for oxidation states. Shape: (N, max_ox_state)
        ox_mask_learnable_tensor_per_crystal (Optional[torch.Tensor]): Per-crystal learnable mask. Shape: (N_crystals, max_ox_state)
        value_round (Optional[int]): If specified, rounds output values to this many decimals.
        atomic_dist_Z_more_than_86 (bool): Whether to mask out elements Z > 86. Default is True.

    Returns:
        Tuple containing:
            - atomic_features (torch.Tensor): Aggregated atomic features. Shape: (total_atoms, feat_dim)
            - bondlength (torch.Tensor): Bond lengths. Shape: (num_edges,)
            - bond_angles (torch.Tensor): Bond angles. Shape: (num_angles,)
            - batch_abc (torch.Tensor): Denormalized lattice lengths. Shape: (N, 3)
            - batch_angle (torch.Tensor): Denormalized lattice angles. Shape: (N, 3)
            - lattice_vectors (torch.Tensor): Lattice vectors. Shape: (N, 3, 3)
            - normalized_dist (torch.Tensor): Final normalized atomic distribution. Shape: (total_atoms, 98)
            - normed_batch_dir_coords (torch.Tensor): Final direct coordinates. Shape: (total_atoms, 3)
            - sharpened_ox_mask (torch.Tensor): Final oxidation mask. Shape: (total_atoms, max_ox_state)
    """
    if atomic_dist_Z_more_than_86:
        # atomic_distribution.shapeにおいて、原子番号86以上の原子を除外するために、ゼロをいれる
        # atomic_distribution.shape = (N, 98)
        mask_tensor = torch.ones_like(atomic_distribution, dtype=atomic_distribution.dtype, device=atomic_distribution.device)
        mask_tensor[:, 86:] = 0
        atomic_distribution = atomic_distribution * mask_tensor



    if adding_noise_scale is not None:
        noise_dir_coords = (torch.rand_like(batch_dir_coords, device=batch_dir_coords.device)*2-1.0)*adding_noise_scale
        noise_scaled_abc = (torch.rand_like(scaled_batch_abc, device=scaled_batch_abc.device)*2-1.0)*adding_noise_scale
        noise_scaled_angle = (torch.rand_like(scaled_batch_angle, device=scaled_batch_angle.device)*2-1.0)*adding_noise_scale
    else:
        noise_dir_coords = 0
        noise_scaled_abc = 0
        noise_scaled_angle = 0

    batch_abc = abc_scaler.rescale(scaled_batch_abc+noise_scaled_abc) 
    batch_angle = angle_scaler.rescale(scaled_batch_angle+noise_scaled_angle)

    # 初期値(init_coords)との差分をlimit_coords_displacementに留め, 内部座標の値を周期的境界条件で0~1に制限する
    if limit_coords_displacement is not None:
        normed_batch_dir_coords = torch.clamp(batch_dir_coords+noise_dir_coords, min = init_coords-limit_coords_displacement, max = init_coords+limit_coords_displacement)
    else:
        normed_batch_dir_coords = batch_dir_coords
    normed_batch_dir_coords = torch.remainder(normed_batch_dir_coords+noise_dir_coords, 1.) # 内部座標の値を周期的境界条件で0~1に制限する

    # Compute normalized atomic distribution and sharpened oxidation state mask.
    normalized_dist, sharpened_ox_mask = compute_normalized_dist_and_ox_mask(
        atomic_distribution=atomic_distribution,
        atomic_mask=atomic_mask,
        ox_states_used_mask=ox_states_used_mask,
        temperature=temperature,
        size=wp_size,
        ox_mask_learnable_tensor_per_crystal=ox_mask_learnable_tensor_per_crystal,  # can be None
    )
    # Expand Wyckoff positions if a WyckoffExpander is provided.
    if wyckoff_expander is not None:
        normed_batch_dir_coords , normalized_dist = wyckoff_expander.wyckoff_expand(
            wp_coords = normed_batch_dir_coords,
            wp_atom_dist = normalized_dist,
            site_ids = site_ids,
        )

        batch_abc = enforce_equal_lattice_lengths(
            batch_abc=batch_abc,
            copy_mask=wyckoff_expander.abc_copy_mask,
        )

        batch_angle = fix_special_lattice_angles(
            batch_angle=batch_angle,
            mask=wyckoff_expander.angle_mask,
            fixed_angle_tensor=wyckoff_expander.fixed_angle_tensor,
        )

    assert n_nodes == normalized_dist.shape[0] == normed_batch_dir_coords.shape[0], f"n_nodes: {n_nodes}, normalized_dist.shape[0]: {normalized_dist.shape[0]}, normed_batch_dir_coords.shape[0]: {normed_batch_dir_coords.shape[0]}"

    atomic_features = atom_dist_to_features(normalized_dist, atom_feat_matrix)
    lattice_vectors = compute_lattice_vectors(batch_abc, batch_angle)
    batch_lattice_vectors = create_batch_lattice_vectors(lattice_vectors, num_edges)
    if value_round is not None:
        batch_lattice_vectors = torch.round(batch_lattice_vectors,decimals=value_round)
        normed_batch_dir_coords = torch.round(normed_batch_dir_coords,decimals=value_round)

    edge_vectors, bondlength = compute_bondlength(normed_batch_dir_coords, batch_dst_ids, batch_src_ids, batch_displace, batch_lattice_vectors)
    bond_angles = calculate_bond_cosine(-edge_vectors[z_src_ids], edge_vectors[z_dst_ids])

    if value_round is not None:
        bond_angles = torch.round(bond_angles,decimals=value_round)
        bondlength = torch.round(bondlength,decimals=value_round)

    return atomic_features, bondlength, bond_angles, batch_abc, batch_angle, lattice_vectors, normalized_dist, normed_batch_dir_coords, sharpened_ox_mask