from typing import Optional, Tuple

import torch

from src.utils.coord import compute_lattice_vectors, direct_to_cartesian_batch
from src.utils.mask_utils import (
    masking_atomic_distribution,
    stacking_learnable_oxsides_mask,
)
from src.utils.nn_utils import temperature_softmax
from src.utils.scalers import ABC_Scaler, Angle_Scaler


def calculate_cartesian_vec(
    num_candidate: int,
    size: torch.Tensor,
    coords4input: torch.Tensor,
    abc: torch.Tensor,
    angle: torch.Tensor,
):
    """
    Convert fractional coordinates to Cartesian coordinates using computed lattice vectors.

    This function calculates the Cartesian coordinates (`pos`) of atoms based on their
    fractional coordinates and the corresponding lattice vectors derived from `abc` and `angle`.
    It also returns the computed lattice vectors for each candidate structure.

    Args:
        num_candidate (int): Number of candidate crystals (N).
        size (torch.Tensor): Number of atoms in each crystal. Shape: (N,)
        coords4input (torch.Tensor): Fractional coordinates of all atoms. Shape: (sum(size), 3)
        abc (torch.Tensor): Lattice lengths (a, b, c) for each structure. Shape: (N, 3)
        angle (torch.Tensor): Lattice angles (alpha, beta, gamma) for each structure. Shape: (N, 3)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - pos (torch.Tensor): Cartesian coordinates of all atoms. Shape: (sum(size), 3)
            - trans_vec (torch.Tensor): Lattice vectors per crystal. Shape: (N, 3, 3)
    """
    # (stepごとの処理) 更新された結晶格子から格子ベクトルの計算をし、原子座標を直交座標に変換
    trans_vec = compute_lattice_vectors(abc, angle)

    # 内部座標を0~1の値にclipする
    coords4input = torch.clip(coords4input, 0, 1)

    # trans_vecを結晶数だけ展開し、各原子座標を直交座標に変換
    trans_vec4input = torch.cat(
        [
            trans_vec[i, ...].unsqueeze(0).repeat(size[i], 1, 1)
            for i in range(num_candidate)
        ],
        dim=0,
    )
    pos = direct_to_cartesian_batch(
        dir_coords=coords4input, lattice_vectors=trans_vec4input
    )
    return pos, trans_vec


def Crystalformer_preprocess(
    scaled_batch_abc: torch.Tensor,
    scaled_batch_angle: torch.Tensor,
    batch_dir_coords: torch.Tensor,
    atomic_distribution: torch.Tensor,
    init_coords: torch.Tensor,
    size: torch.Tensor,
    temperature: float,
    abc_scaler: ABC_Scaler,
    angle_scaler: Angle_Scaler,
    atomic_mask: Optional[torch.Tensor],
    ox_states_used_mask: Optional[torch.Tensor],
    limit_coords_displacement: Optional[float],
    ox_mask_learnable_tensor_per_crystal: Optional[torch.Tensor],
    adding_noise_scale: Optional[float],
    abc_range: Tuple[float, float],
    angle_range: Tuple[float, float],
    atomic_dist_Z_more_than_86: bool = True,
) -> Tuple[
    torch.Tensor,  # batch_abc
    torch.Tensor,  # batch_angle
    torch.Tensor,  # normed_batch_dir_coords
    torch.Tensor,  # normalized_dist
    torch.Tensor,  # pos
    torch.Tensor,  # trans_vec
    torch.Tensor,  # sharpened_ox_mask
]:
    """
    Preprocesses input tensors for Crystalformer optimization.

    This function performs preprocessing for batched crystal structures prior to
    Crystalformer optimization, including:
    - Rescaling and clamping of lattice parameters (abc, angles)
    - Adding Gaussian-like noise (optional)
    - Clipping atomic coordinates to unit cell (0–1)
    - Limiting displacement from initial coordinates
    - Temperature-controlled sharpening of atomic distributions
    - Computing Cartesian coordinates and lattice vectors

    Args:
        scaled_batch_abc (torch.Tensor): Scaled lattice lengths (shape: [N, 3]).
        scaled_batch_angle (torch.Tensor): Scaled lattice angles in degrees (shape: [N, 3]).
        batch_dir_coords (torch.Tensor): Fractional atomic coordinates (shape: [total_atoms, 3]).
        atomic_distribution (torch.Tensor): Atomic type distribution (shape: [total_atoms, 98]).
        init_coords (torch.Tensor): Initial atomic coordinates (shape: [total_atoms, 3]).
        size (torch.Tensor): Number of atoms per structure (shape: [N]).
        temperature (float): Softmax temperature for atomic distribution sharpening.
        abc_scaler (ABC_Scaler): Scaler object for lattice lengths.
        angle_scaler (Angle_Scaler): Scaler object for lattice angles.
        atomic_mask (Optional[torch.Tensor]): Optional atomic mask for element filtering.
        ox_states_used_mask (Optional[torch.Tensor]): Mask indicating valid oxidation states.
        limit_coords_displacement (Optional[float]): Limit for atomic coordinate displacement.
        ox_mask_learnable_tensor_per_crystal (Optional[torch.Tensor]): Crystal-wise learnable oxide masks.
        adding_noise_scale (Optional[float]): Noise amplitude to add to coordinates and lattice params.
        abc_range (Tuple[float, float]): Min/max lattice lengths after rescaling.
        angle_range (Tuple[float, float]): Min/max lattice angles after rescaling.
        atomic_dist_Z_more_than_86 (bool): If True, masks atomic numbers > 86.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - batch_abc: Rescaled and clamped lattice lengths [N, 3]
            - batch_angle: Rescaled and clamped lattice angles [N, 3]
            - normed_batch_dir_coords: Updated atomic coordinates [total_atoms, 3]
            - normalized_dist: Softmax-sharpened atomic distribution [total_atoms, 98]
            - pos: Cartesian atomic coordinates [total_atoms, 3]
            - trans_vec: Lattice vectors per structure [N, 3, 3]
            - sharpened_ox_mask: Oxidation state mask after sharpening [total_atoms, max_ox_states]
    """
    if atomic_dist_Z_more_than_86:
        # atomic_distribution.shape = (N, 98)
        mask_tensor = torch.ones_like(
            atomic_distribution,
            dtype=atomic_distribution.dtype,
            device=atomic_distribution.device,
        )
        mask_tensor[:, 86:] = 0
        atomic_distribution = atomic_distribution * mask_tensor

    # add noise
    if adding_noise_scale is not None:
        noise_dir_coords = (
            torch.rand_like(batch_dir_coords, device=batch_dir_coords.device) * 2 - 1.0
        ) * adding_noise_scale
        noise_scaled_abc = (
            torch.rand_like(scaled_batch_abc, device=scaled_batch_abc.device) * 2 - 1.0
        ) * adding_noise_scale
        noise_scaled_angle = (
            torch.rand_like(scaled_batch_angle, device=scaled_batch_angle.device) * 2
            - 1.0
        ) * adding_noise_scale
    else:
        noise_dir_coords = 0
        noise_scaled_abc = 0
        noise_scaled_angle = 0

    # stepごとの前処理
    batch_abc = abc_scaler.rescale(scaled_batch_abc + noise_scaled_abc)
    batch_angle = angle_scaler.rescale(scaled_batch_angle + noise_scaled_angle)
    ## 制限をかける
    batch_abc = torch.clamp(batch_abc, min=abc_range[0], max=abc_range[1])
    batch_angle = torch.clamp(batch_angle, min=angle_range[0], max=angle_range[1])

    # 初期値(init_coords)との差分をlimit_coords_displacementに留め, 内部座標の値を周期的境界条件で0~1に制限する
    if limit_coords_displacement is not None:
        normed_batch_dir_coords = torch.clamp(
            batch_dir_coords + noise_dir_coords,
            min=init_coords - limit_coords_displacement,
            max=init_coords + limit_coords_displacement,
        )
    else:
        normed_batch_dir_coords = batch_dir_coords
    normed_batch_dir_coords = torch.remainder(
        normed_batch_dir_coords + noise_dir_coords, 1.0
    )  # 内部座標の値を周期的境界条件で0~1に制限する

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

    pos, trans_vec = calculate_cartesian_vec(
        num_candidate=size.shape[0],
        size=size,
        coords4input=normed_batch_dir_coords,
        abc=batch_abc,
        angle=batch_angle,
    )
    return (
        batch_abc,
        batch_angle,
        normed_batch_dir_coords,
        normalized_dist,
        pos,
        trans_vec,
        sharpened_ox_mask,
    )
