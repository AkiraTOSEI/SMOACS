from typing import Dict, List, Optional

import torch

from .create_oxidation_mask import stacking_learnable_oxsides_mask
from .lattice_utils import (
    compute_lattice_vectors,
    direct_to_cartesian_batch,
    masking_atomic_distribution,
)
from .utils import ABC_Scaler, Angle_Scaler, temperature_softmax


def calculate_cartesian_vec(
    num_candidate: int,
    size: torch.Tensor,
    coords4input: torch.Tensor,
    abc: torch.Tensor,
    angle: torch.Tensor,
):
    '''
    Calculate Cartesian coordinates of lattice internal coordinates (pos) and lattice vectors (trans_vec).
    Args:
        num_candidate: The number of candidate crystals to optimize == N
        size: The number of atoms contained in each crystal. shape = (N,)
        coords4input:  The lattice internal coordinates of the atoms in each crystal. shape = (total number of atoms in the candidate==size, 3)
        abc: The length of the lattice vectors of each crystal. shape = (N, 3)
        angle: The angle of the lattice vectors of each crystal. shape = (N, 3)
    Returns:
        pos: Lattice internal coordinates of Cartesian coordinates. shape = (total number of atoms in the batch==size, 3)
        trans_vec: Lattice vectors of Cartesian coordinates. shape = (N, 3, 3)
    '''
    # (step-wise processing) Calculate lattice vectors from the updated crystal lattice and convert atomic coordinates to Cartesian coordinates
    trans_vec = compute_lattice_vectors(abc, angle)

    #  Clip internal coordinates to values between 0 and 1
    coords4input = torch.clip(coords4input, 0, 1)

    # Expand trans_vec by the number of crystals and convert each atomic coordinate to a Cartesian coordinate.
    trans_vec4input = torch.cat([trans_vec[i,...].unsqueeze(0).repeat(size[i],1,1) for i in range(num_candidate)],dim=0)
    pos = direct_to_cartesian_batch(dir_coords=coords4input, lattice_vectors=trans_vec4input)
    return pos, trans_vec


def Crystalformer_preprocess(
        scaled_batch_abc: torch.Tensor,
        scaled_batch_angle: torch.Tensor,
        batch_dir_coords: torch.Tensor,
        atomic_distribution: torch.Tensor,
        init_coords: torch.Tensor,
        size: torch.Tensor,
        temperature:float,
        abc_scaler: ABC_Scaler,
        angle_scaler: Angle_Scaler,
        atomic_mask: Optional[torch.Tensor],
        ox_states_used_mask: Optional[torch.Tensor],
        limit_coords_displacement: Optional[float],
        ox_mask_learnable_tensor_per_crystal: Optional[torch.Tensor] ,
        adding_noise_scale: Optional[float],

    ):

    # add noise
    if adding_noise_scale is not None:
        noise_dir_coords = (torch.rand_like(batch_dir_coords, device=batch_dir_coords.device)*2-1.0)*adding_noise_scale
        noise_scaled_abc = (torch.rand_like(scaled_batch_abc, device=scaled_batch_abc.device)*2-1.0)*adding_noise_scale
        noise_scaled_angle = (torch.rand_like(scaled_batch_angle, device=scaled_batch_angle.device)*2-1.0)*adding_noise_scale
    else:
        noise_dir_coords = 0
        noise_scaled_abc = 0
        noise_scaled_angle = 0

    # preprocess
    batch_abc = abc_scaler.rescale(scaled_batch_abc+noise_scaled_abc)
    batch_angle = angle_scaler.rescale(scaled_batch_angle+noise_scaled_angle)
    
    # Limit the difference from the initial value (init_coords) within `limit_coords_displacement`, and limit the internal coordinates to 0~1 with periodic boundary conditions.
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
        
    pos, trans_vec = calculate_cartesian_vec(
            num_candidate=size.shape[0],
            size=size,
            coords4input=normed_batch_dir_coords,
            abc=batch_abc,
            angle=batch_angle
    )
    return batch_abc, batch_angle, normed_batch_dir_coords, normalized_dist, pos, trans_vec, sharpened_ox_mask



def Crystalformer_prediction(
    prediction_loss_setting_dict:dict,
    normalized_dist: torch.Tensor,
    pos: torch.Tensor,
    batch: torch.Tensor,
    trans_vec: torch.Tensor,
    size: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
    """
    calculate the prediction: bandgap or Ef
    """    
    prediction_dict = {}
    for loss_key in prediction_loss_setting_dict.keys():
        pred = prediction_loss_setting_dict[loss_key]['prediction_model']((normalized_dist, pos, batch, trans_vec, size))
        if loss_key == 'bandgap':
            pred = torch.clip(pred, min=0)
        prediction_dict[loss_key+'_pred'] = pred

    return prediction_dict


def forward_propagation_Crystalformer(
    optimization_targets: List[torch.Tensor],
    fixed_inputs: Dict[str, torch.Tensor],
    scalers: dict,
    temperature:float,
    prediction_loss_setting_dict:dict,
    adding_noise_scale: Optional[float],
    limit_coords_displacement: Optional[float],
    device: str,
):
    scaled_batch_abc, scaled_batch_angle, batch_dir_coords, atomic_distribution, ox_mask_learnable_tensor_per_crystal = optimization_targets
    size = fixed_inputs['size'].to(device)
    batch = fixed_inputs['batch'].to(device)
    abc_scaler = scalers['abc_scaler']
    angle_scaler = scalers['angle_scaler']
    batch_ori = fixed_inputs['batch'].to(device)
    atomic_mask = fixed_inputs['atomic_mask'].to(device)
    ox_states_used_mask = fixed_inputs['ox_states_used_mask'].to(device)
    init_coords = fixed_inputs['init_coords'].to(device)
    batch = batch_ori - batch_ori.min() # Adjust batch (lattice ID) to start from 0

    
    batch_abc, batch_angle, normed_batch_dir_coords, normalized_dist, pos, lattice_vectors, sharpened_ox_mask = Crystalformer_preprocess(
        scaled_batch_abc = scaled_batch_abc, 
        scaled_batch_angle = scaled_batch_angle, 
        batch_dir_coords = batch_dir_coords, 
        atomic_distribution = atomic_distribution,
        size=size,
        temperature=temperature,
        abc_scaler = abc_scaler,
        angle_scaler = angle_scaler,
        atomic_mask = atomic_mask,
        ox_states_used_mask = ox_states_used_mask,
        ox_mask_learnable_tensor_per_crystal = ox_mask_learnable_tensor_per_crystal,
        init_coords = init_coords,
        limit_coords_displacement = limit_coords_displacement,
        adding_noise_scale = adding_noise_scale,
    )

    prediction_dict = Crystalformer_prediction(

        prediction_loss_setting_dict = prediction_loss_setting_dict,
        normalized_dist = normalized_dist,
        pos = pos,
        batch = batch,
        trans_vec = lattice_vectors,
        size = size,
    )

    output_dict = {
        'batch_abc': batch_abc,
        'batch_angle': batch_angle,
        'lattice_vectors': lattice_vectors,
        'pos': pos,
        'normalized_dist': normalized_dist,
        'normed_batch_dir_coords': normed_batch_dir_coords,
        'batch_dir_coords': batch_dir_coords,
        'sharpened_ox_mask': sharpened_ox_mask,
    }
    return prediction_dict, output_dict|prediction_dict