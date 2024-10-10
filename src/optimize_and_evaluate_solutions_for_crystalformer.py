from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from .crystalformer_utils import (
    Crystalformer_prediction,
    forward_propagation_Crystalformer,
)
from .loss import calculate_loss_from_output
from .optimizer import define_optimizer_and_scheduler
from .utils import calculate_onehot


def optimize_solution_Crystalformer(
    settings_dict: Dict[str, Any],
    mini_batch_inputs_dict: dict,
    learning_rates: List[float],
    learning_rate_cycle: List[int], 
    num_steps: int,
    temp_schedule: List[float],
    prediction_loss_setting_dict: dict,
    atomic_dictribution_loss_setting_dict:dict,
    scalers:dict,
    limit_coords_displacement: Optional[float],
    adding_noise_scale: Optional[float],
    angle_optimization: bool = True,
    length_optimization: bool = True,
    device: str = 'cuda',

    ):
    """
    Optimize the solution for a batch of crystal structures using Crystalformer.
    """

    abc_scaler = scalers['abc_scaler']
    angle_scaler = scalers['angle_scaler']
    atomic_distribution = torch.nn.Parameter(mini_batch_inputs_dict['atomic_distribution'].clone().to(device))
    scaled_batch_abc = torch.nn.Parameter(abc_scaler.scale(mini_batch_inputs_dict['batch_abc'].clone().to(device)))
    scaled_batch_angle = torch.nn.Parameter(angle_scaler.scale(mini_batch_inputs_dict['batch_angle'].clone().to(device)))
    batch_dir_coords = torch.nn.Parameter(mini_batch_inputs_dict['batch_dir_coords'].clone().to(device)) 
    ox_mask_learnable_tensor_per_crystal = torch.nn.Parameter(mini_batch_inputs_dict['ox_mask_learnable_tensor_per_crystal'].clone().to(device))

    # optimizer and scheduler
    optimizer_lattice, optimizer_atom, optimizer_coords, scheduler_lattice, scheduler_atom, scheduler_coords = define_optimizer_and_scheduler(
        lattice_lr=learning_rates[0],
        atom_lr=learning_rates[1],
        coords_lr=learning_rates[2],
        ox_mask_learnable_tensor_per_crystal=ox_mask_learnable_tensor_per_crystal,
        atomic_distribution=atomic_distribution,
        scaled_batch_abc=scaled_batch_abc,
        scaled_batch_angle=scaled_batch_angle,
        batch_dir_coords=batch_dir_coords,
        angle_optimization=angle_optimization,
        length_optimization=length_optimization,
        num_steps=num_steps,
        lattice_cycle=learning_rate_cycle[0],
        atom_cycle=learning_rate_cycle[1],
        coords_cycle=learning_rate_cycle[2],
    )

    # initialzation of history
    gap_history, e_form_history, t_loss_history = [], [], []
    
    for step_i in tqdm(range(num_steps),desc="step",leave=False):
        # ステップごとの処理
        optimization_targets = [scaled_batch_abc, scaled_batch_angle, batch_dir_coords, atomic_distribution, ox_mask_learnable_tensor_per_crystal]
        pred_dict, output_dict = forward_propagation_Crystalformer(
            optimization_targets=optimization_targets,
            fixed_inputs=mini_batch_inputs_dict,
            scalers=scalers,
            temperature=temp_schedule[step_i],
            prediction_loss_setting_dict=prediction_loss_setting_dict,
            adding_noise_scale=adding_noise_scale,
            limit_coords_displacement=limit_coords_displacement,
            device=device,
        )

        total_loss, loss_value_dict = calculate_loss_from_output(

            pred_dict=pred_dict,
            prediction_loss_setting_dict=prediction_loss_setting_dict,
            atomic_dictribution_loss_setting_dict=atomic_dictribution_loss_setting_dict,
            num_batch_crystal = settings_dict['num_batch_crystal'],
            sharpened_ox_mask=output_dict['sharpened_ox_mask'],
            normalized_dist=output_dict['normalized_dist'],
            site_ids=mini_batch_inputs_dict['site_ids'].to(device),
            radii_tensor=mini_batch_inputs_dict['radii_tensor'],
            device=device,
        )

        optimizer_lattice.zero_grad(), optimizer_atom.zero_grad(), optimizer_coords.zero_grad()
        total_loss.backward()
        optimizer_lattice.step(), optimizer_atom.step(), optimizer_coords.step()
        scheduler_lattice.step(), scheduler_atom.step(), scheduler_coords.step()

        # record the results
        if 'bandgap_pred' in pred_dict.keys():
            gap_history.append(pred_dict['bandgap_pred'].detach().cpu().numpy())
        if 'e_form_pred' in pred_dict.keys():
            e_form_history.append(pred_dict['e_form_pred'].detach().cpu().numpy())
        if 'tolerance_onehot' in loss_value_dict.keys():
            t_loss_history.append(loss_value_dict['tolerance_onehot'].detach().cpu().numpy())



    # End of optimization
    optimized_mini_batch_inputs_dict = mini_batch_inputs_dict | output_dict 
    optimized_mini_batch_inputs_dict['atomic_distribution'] = atomic_distribution
    optimized_mini_batch_inputs_dict['scaled_batch_abc'] = scaled_batch_abc
    optimized_mini_batch_inputs_dict['scaled_batch_angle'] = scaled_batch_angle
    optimized_mini_batch_inputs_dict['ox_mask_learnable_tensor_per_crystal'] = ox_mask_learnable_tensor_per_crystal
    optimized_mini_batch_inputs_dict['batch_dir_coords'] = batch_dir_coords
    if len(gap_history)>0:
        optimized_mini_batch_inputs_dict['gap_history'] = np.transpose(np.array(gap_history).squeeze(), (1, 0))
    if len(e_form_history)>0:
        optimized_mini_batch_inputs_dict['e_form_history'] = np.transpose(np.array(e_form_history).squeeze(), (1, 0))
    if len(t_loss_history)>0:
        optimized_mini_batch_inputs_dict['t_history'] = np.transpose(np.array(t_loss_history).squeeze(), (1, 0))
     
    return optimized_mini_batch_inputs_dict, scalers


def evaluation_for_each_batch_Crystalformer(
        optimized_mini_batch_inputs_dict: dict,
        scalers: dict,
        settings_dict: dict,
        prediction_loss_setting_dict: dict,
        atomic_dictribution_loss_setting_dict:dict,
        limit_coords_displacement: Optional[float],
        adding_noise_scale: Optional[float],
        onehot_temperature:float = 1E-8,
        device: str = 'cuda',
        
    ):
    """
    Evaluate the optimized solution for a batch of crystal structures using Crystalformer.
    """
    scaled_batch_abc = optimized_mini_batch_inputs_dict['scaled_batch_abc']
    scaled_batch_angle = optimized_mini_batch_inputs_dict['scaled_batch_angle']
    batch_dir_coords = optimized_mini_batch_inputs_dict['batch_dir_coords']
    atomic_distribution = optimized_mini_batch_inputs_dict['atomic_distribution']
    radii_tensor = optimized_mini_batch_inputs_dict['radii_tensor']
    ox_mask_learnable_tensor_per_crystal = optimized_mini_batch_inputs_dict['ox_mask_learnable_tensor_per_crystal']
    size = optimized_mini_batch_inputs_dict['size'].to(device)
    batch_ori = optimized_mini_batch_inputs_dict['batch'].to(device)
    batch = batch_ori - batch_ori.min() # batchを0から始まるように調整

    with torch.no_grad():
        # ステップごとの処理
        distribution_prediction_dict, output_dict = forward_propagation_Crystalformer(
            optimization_targets=[scaled_batch_abc, scaled_batch_angle, batch_dir_coords, atomic_distribution, ox_mask_learnable_tensor_per_crystal],
            fixed_inputs=optimized_mini_batch_inputs_dict,
            scalers=scalers,
            temperature=onehot_temperature,
            #bandgap_model=bandgap_model,
            #e_form_model=e_form_model,
            prediction_loss_setting_dict=prediction_loss_setting_dict,
            adding_noise_scale=adding_noise_scale,
            limit_coords_displacement=limit_coords_displacement,
            device=device,
        )

        _, distribution_loss_dict  = calculate_loss_from_output(
            #bandgap_pred=bandgap_dist,
            #e_form_pred=ef_dist,
            pred_dict=distribution_prediction_dict,
            prediction_loss_setting_dict=prediction_loss_setting_dict,
            atomic_dictribution_loss_setting_dict=atomic_dictribution_loss_setting_dict,
            num_batch_crystal = settings_dict['num_batch_crystal'],
            sharpened_ox_mask=optimized_mini_batch_inputs_dict['sharpened_ox_mask'],
            normalized_dist=optimized_mini_batch_inputs_dict['normalized_dist'],
            site_ids=optimized_mini_batch_inputs_dict['site_ids'].to(device),
            radii_tensor=radii_tensor,
            device=device,
        )

        # calculation with one-hot atomic distribution
        max_val, onehot_x, _, onehot_ox_mask = calculate_onehot(output_dict, atom_feat_matrix=None)
        onehot_prediction_dict = Crystalformer_prediction(
            #bandgap_model = bandgap_model,
            #e_form_model = e_form_model,
            prediction_loss_setting_dict = prediction_loss_setting_dict,
            normalized_dist = onehot_x,
            pos = output_dict['pos'],
            batch = batch,
            trans_vec = output_dict['lattice_vectors'],
            size = size,
        )

        _, onehot_loss_dict  = calculate_loss_from_output(
            pred_dict=onehot_prediction_dict,
            prediction_loss_setting_dict=prediction_loss_setting_dict,
            atomic_dictribution_loss_setting_dict=atomic_dictribution_loss_setting_dict,
            num_batch_crystal = settings_dict['num_batch_crystal'],
            sharpened_ox_mask=onehot_ox_mask,
            normalized_dist= onehot_x,
            site_ids=optimized_mini_batch_inputs_dict['site_ids'].to(device),
            radii_tensor=radii_tensor,
            device=device,
        )

    # record the results
    optimized_mini_batch_inputs_dict['max_val'] = max_val
    optimized_mini_batch_inputs_dict['onehot_x'] = onehot_x
    optimized_mini_batch_inputs_dict['onehot_ox_mask'] = onehot_ox_mask

    for loss_key in prediction_loss_setting_dict.keys():
        optimized_mini_batch_inputs_dict[f'{loss_key}_onehot'] = onehot_prediction_dict[loss_key+'_pred']
        optimized_mini_batch_inputs_dict[f'{loss_key}_dist'] = distribution_prediction_dict[loss_key+'_pred']
        optimized_mini_batch_inputs_dict[f'loss_{loss_key}_onehot'] = onehot_loss_dict[loss_key+'_loss']
        optimized_mini_batch_inputs_dict[f'loss_{loss_key}_dist'] = distribution_loss_dict[loss_key+'_loss']
    for loss_key in atomic_dictribution_loss_setting_dict.keys():
        optimized_mini_batch_inputs_dict[f'loss_{loss_key}_onehot'] = onehot_loss_dict[loss_key+'_loss']
        optimized_mini_batch_inputs_dict[f'loss_{loss_key}_dist'] = distribution_loss_dict[loss_key+'_loss']
        optimized_mini_batch_inputs_dict[f'{loss_key}_onehot'] = onehot_loss_dict[loss_key]
        optimized_mini_batch_inputs_dict[f'{loss_key}_dist'] = distribution_loss_dict[loss_key]


    return optimized_mini_batch_inputs_dict