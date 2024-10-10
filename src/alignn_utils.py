from typing import Dict, List, Optional

import dgl
import torch

from .alignn_preprocess import (
    calculate_bond_cosine,
    compute_bondlength,
    create_batch_lattice_vectors,
)
from .create_oxidation_mask import stacking_learnable_oxsides_mask
from .lattice_utils import compute_lattice_vectors, masking_atomic_distribution
from .utils import ABC_Scaler, Angle_Scaler, temperature_softmax


def atom_dist_to_features(atomic_distribution, atom_feat_matrix):
    """
    Convert atomic distribution to atomic features.
    """
    atomic_features = torch.matmul(atomic_distribution, atom_feat_matrix)
    return atomic_features

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
        size:torch.Tensor,
        init_coords:torch.Tensor,
        temperature:float,
        abc_scaler:ABC_Scaler,
        angle_scaler:Angle_Scaler,
        num_edges:int,
        atom_feat_matrix:torch.Tensor,
        limit_coords_displacement: Optional[float],
        adding_noise_scale: Optional[float],
        atomic_mask: Optional[torch.Tensor] = None,
        ox_states_used_mask: Optional[torch.Tensor] = None,
        ox_mask_learnable_tensor_per_crystal: Optional[torch.Tensor] = None,

    ):
    """
    Function to preprocess data before feeding into ALIGNN
    """
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

    if ox_mask_learnable_tensor_per_crystal is not None :
        stacked_learnable_ox_weight = stacking_learnable_oxsides_mask(ox_mask_learnable_tensor_per_crystal, size)
        normalized_dist, sharpened_ox_mask = masking_atomic_distribution(atomic_distribution, atomic_mask, ox_states_used_mask, stacked_learnable_ox_weight, temperature)
    else:
        sharpened_ox_mask = torch.zeros_like(ox_states_used_mask) * torch.nan
        normalized_dist = temperature_softmax(atomic_distribution, temperature=temperature)

    atomic_features = atom_dist_to_features(normalized_dist, atom_feat_matrix)
    lattice_vectors = compute_lattice_vectors(batch_abc, batch_angle)
    batch_lattice_vectors = create_batch_lattice_vectors(lattice_vectors, num_edges)
    edge_vectors, bondlength = compute_bondlength(normed_batch_dir_coords, batch_dst_ids, batch_src_ids, batch_displace, batch_lattice_vectors)
    bond_angles = calculate_bond_cosine(-edge_vectors[z_src_ids], edge_vectors[z_dst_ids])

    return atomic_features, bondlength, bond_angles, batch_abc, batch_angle, lattice_vectors, normalized_dist, normed_batch_dir_coords, sharpened_ox_mask


def ALIGNN_prediction(
    prediction_loss_setting_dict: dict,
    g: dgl.DGLGraph,
    lg: dgl.DGLGraph,
    atomic_features: torch.Tensor,
    bondlength: torch.Tensor,
    bond_angles: torch.Tensor,
)-> dict:
    
    prediction_dict = {}
    for loss_key in prediction_loss_setting_dict.keys():
        pred = prediction_loss_setting_dict[loss_key]['prediction_model']((g, lg), atomic_features, bondlength, bond_angles)
        if loss_key == 'bandgap':
            pred = torch.clip(pred, min=0)
        prediction_dict[loss_key+'_pred'] = pred

    return prediction_dict

def forward_propagation_ALIGNN(
    optimization_targets: List[torch.Tensor],
    fixed_inputs: Dict[str, torch.Tensor],
    scalers: dict,
    temperature:float,
    limit_coords_displacement: Optional[float],
    #bandgap_model: torch.nn.Module,
    #e_form_model: torch.nn.Module,
    prediction_loss_setting_dict:dict,
    adding_noise_scale: Optional[float],
    device:str,
):
    scaled_batch_abc, scaled_batch_angle, batch_dir_coords, atomic_distribution, ox_mask_learnable_tensor_per_crystal = optimization_targets
    batch_dst_ids = fixed_inputs['batch_dst_ids']
    batch_src_ids = fixed_inputs['batch_src_ids']
    batch_displace = fixed_inputs['batch_displace']
    z_src_ids = fixed_inputs['z_src_ids']
    z_dst_ids = fixed_inputs['z_dst_ids']
    size = fixed_inputs['size']
    num_edges = fixed_inputs['num_edges']
    atom_feat_matrix = fixed_inputs['atom_feat_matrix']
    atomic_mask = fixed_inputs['atomic_mask']
    ox_states_used_mask = fixed_inputs['ox_states_used_mask']
    g = fixed_inputs['g']
    lg = fixed_inputs['lg']

    abc_scaler = scalers['abc_scaler']
    angle_scaler = scalers['angle_scaler']


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
        init_coords = fixed_inputs['init_coords'].to(device),
        temperature=temperature,
        abc_scaler = abc_scaler,
        angle_scaler = angle_scaler,
        limit_coords_displacement = limit_coords_displacement,
        num_edges = num_edges.to(device),
        adding_noise_scale = adding_noise_scale,
        atom_feat_matrix = atom_feat_matrix.to(device),
        atomic_mask = atomic_mask.to(device),
        ox_states_used_mask = ox_states_used_mask.to(device),
        ox_mask_learnable_tensor_per_crystal = ox_mask_learnable_tensor_per_crystal.to(device),
    )

    pred_dict = ALIGNN_prediction(
        #bandgap_model = bandgap_model,
        #e_form_model = e_form_model,
        prediction_loss_setting_dict = prediction_loss_setting_dict,
        g = g.to(device),
        lg = lg.to(device),
        atomic_features = atomic_features.to(device),
        bondlength = bondlength.to(device),
        bond_angles = bond_angles.to(device),
    )

    output_dict = {
        #'bandgap_pred': bandgap_pred,
        #'e_form_pred': e_form_pred,
        'atomic_features': atomic_features,
        'bondlength': bondlength,
        'bond_angles': bond_angles,
        'batch_abc': batch_abc,
        'batch_angle': batch_angle,
        'lattice_vectors': lattice_vectors,
        'normalized_dist': normalized_dist,
        'normed_batch_dir_coords': normed_batch_dir_coords,
        'sharpened_ox_mask': sharpened_ox_mask,
    }
    return pred_dict, output_dict|pred_dict
