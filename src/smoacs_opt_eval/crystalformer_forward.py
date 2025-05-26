from typing import Dict, List, Optional, Tuple

import torch

from src.preprocess.crystalformer_preprocess import Crystalformer_preprocess
from src.utils.coord import compute_lattice_vectors
from src.utils.feature_utils import atom_dist_to_features
from src.utils.mask_utils import (
    masking_atomic_distribution,
    stacking_learnable_oxsides_mask,
)
from src.utils.nn_utils import temperature_softmax
from src.utils.scalers import ABC_Scaler, Angle_Scaler


def Crystalformer_prediction(
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    normalized_dist: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    trans_vec: torch.Tensor,
    size: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Predicts bandgap and formation energy using pretrained Crystalformer models.

    Args:
        bandgap_model (torch.nn.Module): Pretrained model for bandgap prediction.
        e_form_model (torch.nn.Module): Pretrained model for formation energy prediction.
        normalized_dist (torch.Tensor): Normalized atomic distribution. Shape: (num_atoms, num_elements)
        pos (torch.Tensor): Atomic positions (direct coordinates). Shape: (num_atoms, 3)
        batch (torch.Tensor): Batch indices for each atom. Shape: (num_atoms,)
        trans_vec (torch.Tensor): Lattice vectors per sample. Shape: (num_samples, 3, 3)
        size (torch.Tensor): Number of atoms in each structure. Shape: (num_samples,)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Predicted bandgap and formation energy per structure. Shape: (num_samples,)
    """
    if bandgap_model.training:
        bandgap_model.eval()
    if e_form_model.training:
        e_form_model.eval()
    bandgap_pred = torch.clip(
        bandgap_model((normalized_dist, pos, batch, trans_vec, size)), min=0
    )
    ef_pred = e_form_model((normalized_dist, pos, batch, trans_vec, size))

    return bandgap_pred, ef_pred


def forward_propagation_Crystalformer(
    optimization_targets: List[torch.Tensor],
    fixed_inputs: Dict[str, torch.Tensor],
    scalers: dict,
    temperature: float,
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    adding_noise_scale: Optional[float],
    limit_coords_displacement: Optional[float],
    abc_range: Tuple[float, float],
    angle_range: Tuple[float, float],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Executes forward propagation for Crystalformer optimization.

    Args:
        optimization_targets (List[torch.Tensor]): List containing [scaled_batch_abc, scaled_batch_angle, batch_dir_coords, atomic_distribution, ox_mask_learnable_tensor_per_crystal].
        fixed_inputs (Dict[str, torch.Tensor]): Dictionary with static inputs including batch, atomic masks, size, and initialization coordinates.
        scalers (dict): Dictionary containing abc_scaler and angle_scaler.
        temperature (float): Temperature parameter for softmax.
        bandgap_model (torch.nn.Module): Model for bandgap prediction.
        e_form_model (torch.nn.Module): Model for formation energy prediction.
        adding_noise_scale (Optional[float]): If specified, noise added to coordinates and lattice parameters.
        limit_coords_displacement (Optional[float]): Maximum allowed deviation from init_coords.
        abc_range (Tuple[float, float]): Clamping range for abc lattice lengths.
        angle_range (Tuple[float, float]): Clamping range for lattice angles.

    Returns:
        Tuple:
            - bandgap_pred (torch.Tensor): Predicted bandgaps. Shape: (num_samples,)
            - ef_pred (torch.Tensor): Predicted formation energies. Shape: (num_samples,)
            - output_dict (Dict[str, torch.Tensor]): Dictionary of intermediate tensors for further use or analysis.
    """
    (
        scaled_batch_abc,
        scaled_batch_angle,
        batch_dir_coords,
        atomic_distribution,
        ox_mask_learnable_tensor_per_crystal,
    ) = optimization_targets
    size = fixed_inputs["size"].to("cuda")
    batch = fixed_inputs["batch"].to("cuda")
    abc_scaler = scalers["abc_scaler"]
    angle_scaler = scalers["angle_scaler"]
    batch_ori = fixed_inputs["batch"].to("cuda")
    atomic_mask = fixed_inputs["atomic_mask"].to("cuda")
    ox_states_used_mask = fixed_inputs["ox_states_used_mask"].to("cuda")
    init_coords = fixed_inputs["init_coords"].to("cuda")
    batch = batch_ori - batch_ori.min()  # batchを0から始まるように調整

    (
        batch_abc,
        batch_angle,
        normed_batch_dir_coords,
        normalized_dist,
        pos,
        lattice_vectors,
        sharpened_ox_mask,
    ) = Crystalformer_preprocess(
        scaled_batch_abc=scaled_batch_abc,
        scaled_batch_angle=scaled_batch_angle,
        batch_dir_coords=batch_dir_coords,
        atomic_distribution=atomic_distribution,
        size=size,
        temperature=temperature,
        abc_scaler=abc_scaler,
        angle_scaler=angle_scaler,
        atomic_mask=atomic_mask,
        ox_states_used_mask=ox_states_used_mask,
        ox_mask_learnable_tensor_per_crystal=ox_mask_learnable_tensor_per_crystal,
        init_coords=init_coords,
        limit_coords_displacement=limit_coords_displacement,
        adding_noise_scale=adding_noise_scale,
        abc_range=abc_range,
        angle_range=angle_range,
    )

    bandgap_pred, ef_pred = Crystalformer_prediction(
        bandgap_model=bandgap_model,
        e_form_model=e_form_model,
        normalized_dist=normalized_dist,
        pos=pos,
        batch=batch,
        trans_vec=lattice_vectors,
        size=size,
    )

    output_dict = {
        "bandgap_pred": bandgap_pred,
        "ef_pred": ef_pred,
        "batch_abc": batch_abc,
        "batch_angle": batch_angle,
        "lattice_vectors": lattice_vectors,
        "pos": pos,
        "normalized_dist": normalized_dist,
        "normed_batch_dir_coords": normed_batch_dir_coords,
        "batch_dir_coords": batch_dir_coords,
        "sharpened_ox_mask": sharpened_ox_mask,
        "batch": batch,
    }
    return bandgap_pred, ef_pred, output_dict
