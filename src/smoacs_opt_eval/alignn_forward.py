from typing import Dict, List, Optional, Tuple

import dgl
import torch
from torch import nn

from src.preprocess.alignn_preprocess import ALIGNN_preprocess


def ALIGNN_prediction(
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    g: dgl.DGLGraph,
    lg: dgl.DGLGraph,
    atomic_features: torch.Tensor,
    bondlength: torch.Tensor,
    bond_angles: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Run ALIGNN bandgap and formation energy prediction for a given crystal graph.

    Args:
        bandgap_model (torch.nn.Module): Pretrained ALIGNN model for bandgap prediction.
        e_form_model (torch.nn.Module): Pretrained ALIGNN model for formation energy prediction.
        g (dgl.DGLGraph): Crystal graph.
        lg (dgl.DGLGraph): Line graph (angle-level graph).
        atomic_features (torch.Tensor): Tensor of atomic input features. Shape: (N_atoms, feat_dim).
        bondlength (torch.Tensor): Bond length features. Shape: (N_edges,).
        bond_angles (torch.Tensor): Bond angle features. Shape: (N_triplets,).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Bandgap prediction tensor.
            - Formation energy prediction tensor.
    """
    bandgap_pred = torch.clip(
        bandgap_model((g, lg), atomic_features, bondlength, bond_angles), min=0
    )
    ef_pred = e_form_model((g, lg), atomic_features, bondlength, bond_angles)
    return bandgap_pred, ef_pred


def forward_propagation_ALIGNN(
    optimization_targets: List[torch.Tensor],
    fixed_inputs: Dict[str, torch.Tensor],
    scalers: dict,
    temperature: float,
    limit_coords_displacement: Optional[float],
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    abc_range: Tuple[float, float],
    angle_range: Tuple[float, float],
    adding_noise_scale: Optional[float],
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Performs a full forward pass through ALIGNN using optimized input variables.

    Args:
        optimization_targets (List[torch.Tensor]):
            List of learnable input tensors [scaled_batch_abc, scaled_batch_angle, batch_dir_coords, atomic_distribution, ox_mask_learnable_tensor_per_crystal].
        fixed_inputs (Dict[str, torch.Tensor]):
            Fixed input tensors and metadata such as graph edges and masks.
        scalers (Dict[str, any]): Dictionary containing abc_scaler and angle_scaler.
        temperature (float): Softmax temperature used for masking atomic distribution.
        limit_coords_displacement (Optional[float]): Maximum allowed shift in coordinates.
        bandgap_model (torch.nn.Module): ALIGNN model for bandgap prediction.
        e_form_model (torch.nn.Module): ALIGNN model for formation energy prediction.
        abc_range (Tuple[float, float]): Clamping range for abc lattice values.
        angle_range (Tuple[float, float]): Clamping range for angle values.
        adding_noise_scale (Optional[float]): Optional Gaussian noise scale.

    Returns:
        Tuple containing:
            - bandgap_pred (torch.Tensor): Predicted bandgap values.
            - ef_pred (torch.Tensor): Predicted formation energy values.
            - output_dict (Dict[str, torch.Tensor]): Dictionary of intermediate outputs including
              atomic features, coordinates, structure parameters, and bond/angle features.
    """
    (
        scaled_batch_abc,
        scaled_batch_angle,
        batch_dir_coords,
        atomic_distribution,
        ox_mask_learnable_tensor_per_crystal,
    ) = optimization_targets
    batch_dst_ids = fixed_inputs["batch_dst_ids"]
    batch_src_ids = fixed_inputs["batch_src_ids"]
    batch_displace = fixed_inputs["batch_displace"]
    z_src_ids = fixed_inputs["z_src_ids"]
    z_dst_ids = fixed_inputs["z_dst_ids"]
    size = fixed_inputs["size"]
    num_edges = fixed_inputs["num_edges"]
    atom_feat_matrix = fixed_inputs["atom_feat_matrix"]
    atomic_mask = fixed_inputs["atomic_mask"]
    ox_states_used_mask = fixed_inputs["ox_states_used_mask"]
    g = fixed_inputs["g"]
    lg = fixed_inputs["lg"]

    abc_scaler = scalers["abc_scaler"]
    angle_scaler = scalers["angle_scaler"]

    (
        atomic_features,
        bondlength,
        bond_angles,
        batch_abc,
        batch_angle,
        lattice_vectors,
        normalized_dist,
        normed_batch_dir_coords,
        sharpened_ox_mask,
    ) = ALIGNN_preprocess(
        scaled_batch_abc=scaled_batch_abc.to("cuda"),
        scaled_batch_angle=scaled_batch_angle.to("cuda"),
        batch_dir_coords=batch_dir_coords.to("cuda"),
        atomic_distribution=atomic_distribution.to("cuda"),
        batch_dst_ids=batch_dst_ids.to("cuda"),
        batch_src_ids=batch_src_ids.to("cuda"),
        batch_displace=batch_displace.to("cuda"),
        z_src_ids=z_src_ids.to("cuda"),
        z_dst_ids=z_dst_ids.to("cuda"),
        size=size.to("cuda"),
        init_coords=fixed_inputs["init_coords"].to("cuda"),
        temperature=temperature,
        abc_scaler=abc_scaler,
        angle_scaler=angle_scaler,
        limit_coords_displacement=limit_coords_displacement,
        num_edges=num_edges.to("cuda"),
        adding_noise_scale=adding_noise_scale,
        abc_range=abc_range,
        angle_range=angle_range,
        atom_feat_matrix=atom_feat_matrix.to("cuda"),
        atomic_mask=atomic_mask.to("cuda"),
        ox_states_used_mask=ox_states_used_mask.to("cuda"),
        ox_mask_learnable_tensor_per_crystal=ox_mask_learnable_tensor_per_crystal.to(
            "cuda"
        ),
    )

    bandgap_pred, ef_pred = ALIGNN_prediction(
        bandgap_model=bandgap_model,
        e_form_model=e_form_model,
        g=g.to("cuda"),
        lg=lg.to("cuda"),
        atomic_features=atomic_features.to("cuda"),
        bondlength=bondlength.to("cuda"),
        bond_angles=bond_angles.to("cuda"),
    )

    output_dict = {
        "bandgap_pred": bandgap_pred,
        "ef_pred": ef_pred,
        "atomic_features": atomic_features,
        "bondlength": bondlength,
        "bond_angles": bond_angles,
        "batch_abc": batch_abc,
        "batch_angle": batch_angle,
        "lattice_vectors": lattice_vectors,
        "normalized_dist": normalized_dist,
        "normed_batch_dir_coords": normed_batch_dir_coords,
        "sharpened_ox_mask": sharpened_ox_mask,
    }
    return bandgap_pred, ef_pred, output_dict
