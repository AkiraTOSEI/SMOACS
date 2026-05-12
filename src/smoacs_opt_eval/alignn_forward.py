from typing import Any, Dict, List, Optional, Tuple

import dgl
import torch
from torch import nn

from src.preprocess.alignn_preprocess import ALIGNN_preprocess
from src.spg_wp.wyckoff_expander import WyckoffExpander


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


def predict_all_alignn(
    predictors: Dict[str, Any],
    g: dgl.DGLGraph,
    lg: dgl.DGLGraph,
    atomic_features: torch.Tensor,
    bondlength: torch.Tensor,
    bond_angles: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    model_inputs = {
        "g": g,
        "lg": lg,
        "atomic_features": atomic_features,
        "bondlength": bondlength,
        "bond_angles": bond_angles,
    }
    return {name: predictor.predict(model_inputs) for name, predictor in predictors.items()}


def forward_propagation_ALIGNN(
    optimization_targets: List[torch.Tensor],
    fixed_inputs: Dict[str, torch.Tensor],
    scalers: dict,
    temperature:float,
    limit_coords_displacement: Optional[float],
    bandgap_model: Optional[torch.nn.Module],
    e_form_model: Optional[torch.nn.Module],
    abc_range: Tuple[float, float],
    angle_range: Tuple[float, float],
    adding_noise_scale: Optional[float],
    wyckoff_expander: Optional[WyckoffExpander],
    predictors: Optional[Dict[str, Any]] = None,
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
    scaled_batch_abc, scaled_batch_angle, batch_dir_coords, atomic_distribution, ox_mask_learnable_tensor_per_crystal = optimization_targets
    batch_dst_ids = fixed_inputs['batch_dst_ids']
    batch_src_ids = fixed_inputs['batch_src_ids']
    batch_displace = fixed_inputs['batch_displace']
    z_src_ids = fixed_inputs['z_src_ids']
    z_dst_ids = fixed_inputs['z_dst_ids']
    size = fixed_inputs['size']
    wp_size = fixed_inputs['wp_size']
    num_edges = fixed_inputs['num_edges']
    atom_feat_matrix = fixed_inputs['atom_feat_matrix']
    atomic_mask = fixed_inputs['atomic_mask']
    ox_states_used_mask = fixed_inputs['ox_states_used_mask']
    g = fixed_inputs['g']
    lg = fixed_inputs['lg']

    abc_scaler = scalers['abc_scaler']
    angle_scaler = scalers['angle_scaler']


    device = scaled_batch_abc.device
    atomic_features, bondlength, bond_angles, batch_abc, batch_angle, lattice_vectors, normalized_dist, normed_batch_dir_coords, sharpened_ox_mask = ALIGNN_preprocess(
        scaled_batch_abc = scaled_batch_abc.to(device), 
        scaled_batch_angle = scaled_batch_angle.to(device), 
        batch_dir_coords = batch_dir_coords.to(device), 
        atomic_distribution = atomic_distribution.to(device),
        batch_dst_ids = batch_dst_ids.to(device),
        batch_src_ids = batch_src_ids.to(device),
        batch_displace = batch_displace.to(device),
        z_src_ids = z_src_ids.to(device),
        z_dst_ids = z_dst_ids.to(device),
        size=size.to(device),
        wp_size=wp_size.to(device),
        site_ids = fixed_inputs['site_ids'].to(device),
        init_coords = fixed_inputs['init_coords'].to(device),
        temperature=temperature,
        abc_scaler = abc_scaler,
        angle_scaler = angle_scaler,
        limit_coords_displacement = limit_coords_displacement,
        num_edges = num_edges.to(device),
        adding_noise_scale = adding_noise_scale,
        abc_range = abc_range,
        angle_range = angle_range,
        atom_feat_matrix = atom_feat_matrix.to(device),
        atomic_mask = atomic_mask.to(device),
        ox_states_used_mask = ox_states_used_mask.to(device),
        ox_mask_learnable_tensor_per_crystal = ox_mask_learnable_tensor_per_crystal.to(device),
        wyckoff_expander = wyckoff_expander,
        n_nodes = g.number_of_nodes(),
    )

    g = g.to(device)
    lg = lg.to(device)
    atomic_features = atomic_features.to(device)
    bondlength = bondlength.to(device)
    bond_angles = bond_angles.to(device)

    bandgap_pred, ef_pred = None, None
    if bandgap_model is not None and e_form_model is not None:
        bandgap_pred, ef_pred = ALIGNN_prediction(
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            g=g,
            lg=lg,
            atomic_features=atomic_features,
            bondlength=bondlength,
            bond_angles=bond_angles,
        )
    pred_dict = None
    if predictors is not None:
        pred_dict = predict_all_alignn(
            predictors=predictors,
            g=g,
            lg=lg,
            atomic_features=atomic_features,
            bondlength=bondlength,
            bond_angles=bond_angles,
        )

    output_dict = {
        'bandgap_pred': bandgap_pred,
        'ef_pred': ef_pred,
        'atomic_features': atomic_features,
        'bondlength': bondlength,
        'bond_angles': bond_angles,
        'batch_abc': batch_abc,
        'batch_angle': batch_angle,
        'lattice_vectors': lattice_vectors,
        'normalized_dist': normalized_dist,
        'normed_batch_dir_coords': normed_batch_dir_coords,
        'sharpened_ox_mask': sharpened_ox_mask,
        'pred_dict': pred_dict,
    }
    return bandgap_pred, ef_pred, output_dict
