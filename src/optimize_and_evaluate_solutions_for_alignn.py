from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .alignn_utils import ALIGNN_prediction, forward_propagation_ALIGNN
from .loss import calculate_loss_from_output
from .optimizer import define_optimizer_and_scheduler
from .save_data import save_poscar
from .update_graphs_alignn import update_graphs_and_trainable_parameters
from .utils import calculate_onehot


def optimize_solution_ALIGNN(
    settings_dict: Dict[str, Any],
    mini_batch_inputs_dict: Dict[str, Any],
    learning_rates: List[float],
    learning_rate_cycle: List[int],
    num_steps:int, 
    temp_schedule:List[float],
    prediction_loss_setting_dict:dict,
    atomic_dictribution_loss_setting_dict:dict,
    scalers:dict,
    adding_noise_scale: Optional[float],
    limit_coords_displacement: Optional[float],
    angle_optimization: bool = True,
    length_optimization: bool = True,
    device: str = 'cuda',
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    optimize solutions
    """


    abc_scaler = scalers['abc_scaler']
    angle_scaler = scalers['angle_scaler']
    atomic_distribution = nn.Parameter(mini_batch_inputs_dict['atomic_distribution'].clone().to(device), requires_grad=True)
    scaled_batch_abc = nn.Parameter(abc_scaler.scale(mini_batch_inputs_dict['batch_abc'].clone().to(device)), requires_grad=True)
    scaled_batch_angle = nn.Parameter(angle_scaler.scale(mini_batch_inputs_dict['batch_angle'].clone().to(device)), requires_grad=True)
    ox_mask_learnable_tensor_per_crystal = nn.Parameter(mini_batch_inputs_dict['ox_mask_learnable_tensor_per_crystal'].to(device).clone(), requires_grad=True)
    batch_dir_coords=nn.Parameter(mini_batch_inputs_dict['batch_dir_coords'].to(device).clone(), requires_grad=True)
    radii_tensor = mini_batch_inputs_dict['radii_tensor'].to(device)
    site_ids = mini_batch_inputs_dict['site_ids'].to(device)

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
    lattice_lr_history, atom_lr_history, coords_lr_history = [], [], []

    for step_i in tqdm(range(num_steps), desc='Optimizing', leave=False):
        
        ### Update graph and inputs
        if step_i in settings_dict['ALIGNN_update_step']:
            print(f'<info> graph update at {step_i} steps')
            # reinitialize the learning parameters and Optimizer. Update the maximum value of the cosine annealing learning rate to the current value
            lattice_lr=optimizer_lattice.param_groups[0]['lr']
            atom_lr=optimizer_atom.param_groups[0]['lr']
            coords_lr=optimizer_coords.param_groups[0]['lr']
            settings_dict['learning_rates'] = [lattice_lr, atom_lr, coords_lr]
            mini_batch_inputs_dict, scalers, optimiers_dict, learnable_parameters_dict = update_graphs_and_trainable_parameters(
                optimized_mini_batch_inputs_dict= {
                    'atomic_distribution': atomic_distribution,
                    'batch_abc': abc_scaler.rescale(scaled_batch_abc),
                    'batch_angle': angle_scaler.rescale(scaled_batch_angle),
                    'batch_dir_coords': batch_dir_coords,
                    'ox_mask_learnable_tensor_per_crystal': ox_mask_learnable_tensor_per_crystal,
                    'size': mini_batch_inputs_dict['size'],
                    'atomic_mask': mini_batch_inputs_dict['atomic_mask'],
                    'ox_states_used_mask': mini_batch_inputs_dict['ox_states_used_mask'],
                    'atom_feat_matrix': mini_batch_inputs_dict['atom_feat_matrix'],
                    'site_ids': mini_batch_inputs_dict['site_ids'],
                    'radii_tensor': mini_batch_inputs_dict['radii_tensor'],
                    'init_coords': mini_batch_inputs_dict['init_coords'],
                    #'lattice_vectors': output_dict['lattice_vectors'],
                    'fnames': mini_batch_inputs_dict['fnames'],
                },
                settings_dict=settings_dict,
                tmp_poscar_dir='./tmp_poscars',
                device=device,
            )


            ##### update variables
            abc_scaler = scalers['abc_scaler']
            angle_scaler = scalers['angle_scaler']
            scaled_batch_abc, scaled_batch_angle, batch_dir_coords = learnable_parameters_dict['scaled_batch_abc'], learnable_parameters_dict['scaled_batch_angle'], learnable_parameters_dict['batch_dir_coords']
            atomic_distribution, ox_mask_learnable_tensor_per_crystal = learnable_parameters_dict['atomic_distribution'], learnable_parameters_dict['ox_mask_learnable_tensor_per_crystal']
            optimizer_lattice, optimizer_atom, optimizer_coords = optimiers_dict['optimizer_lattice'], optimiers_dict['optimizer_atom'], optimiers_dict['optimizer_coords']
            scheduler_lattice, scheduler_atom, scheduler_coords = optimiers_dict['scheduler_lattice'], optimiers_dict['scheduler_atom'], optimiers_dict['scheduler_coords']


        # re-calculate atomic features, crystal structures, bondlength, bond angles
        optimization_targets = [scaled_batch_abc, scaled_batch_angle, batch_dir_coords, atomic_distribution, ox_mask_learnable_tensor_per_crystal]
        pred_dict, output_dict = forward_propagation_ALIGNN(
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
            site_ids=site_ids,
            radii_tensor=radii_tensor,
            device=device,
        )

        if step_i == 0:
            # save initial poscar
            save_poscar(mini_batch_inputs_dict | output_dict, 'init_poscars')

        optimizer_lattice.zero_grad(), optimizer_atom.zero_grad(), optimizer_coords.zero_grad()
        total_loss.backward()
        optimizer_lattice.step(), optimizer_atom.step(), optimizer_coords.step()
        scheduler_lattice.step(), scheduler_atom.step(), scheduler_coords.step()

        # record the history of losses and learning rates
        ### loss
        if 'bandgap_pred' in pred_dict.keys():
            gap_history.append(pred_dict['bandgap_pred'].detach().cpu().numpy())
        if 'e_form_pred' in pred_dict.keys():
            e_form_history.append(pred_dict['e_form_pred'].detach().cpu().numpy())
        if 'tolerance_onehot' in loss_value_dict.keys():
            t_loss_history.append(loss_value_dict['tolerance_onehot'].detach().cpu().numpy())
        ### learning rate
        lattice_lr_history.append(optimizer_lattice.param_groups[0]['lr'])
        atom_lr_history.append(optimizer_atom.param_groups[0]['lr'])
        coords_lr_history.append(optimizer_coords.param_groups[0]['lr'])

    # final graph update
    if settings_dict['ALIGNN_update_step'] is not None and len(settings_dict['ALIGNN_update_step'])>0:
        print('<info> graph update at final steps')
        # reinitialize the learning parameters and Optimizer. Update the maximum value of the cosine annealing learning rate to the current value
        lattice_lr=optimizer_lattice.param_groups[0]['lr']
        atom_lr=optimizer_atom.param_groups[0]['lr']
        coords_lr=optimizer_coords.param_groups[0]['lr']
        settings_dict['learning_rates'] = [lattice_lr, atom_lr, coords_lr]
        mini_batch_inputs_dict, scalers, optimiers_dict, learnable_parameters_dict = update_graphs_and_trainable_parameters(
            optimized_mini_batch_inputs_dict= {
                'atomic_distribution': atomic_distribution,
                'batch_abc': abc_scaler.rescale(scaled_batch_abc),
                'batch_angle': angle_scaler.rescale(scaled_batch_angle),
                'batch_dir_coords': batch_dir_coords,
                'ox_mask_learnable_tensor_per_crystal': ox_mask_learnable_tensor_per_crystal,
                'size': mini_batch_inputs_dict['size'],
                'atomic_mask': mini_batch_inputs_dict['atomic_mask'],
                'ox_states_used_mask': mini_batch_inputs_dict['ox_states_used_mask'],
                'atom_feat_matrix': mini_batch_inputs_dict['atom_feat_matrix'],
                'site_ids': mini_batch_inputs_dict['site_ids'],
                'radii_tensor': mini_batch_inputs_dict['radii_tensor'],
                'init_coords': mini_batch_inputs_dict['init_coords'],
                #'lattice_vectors': output_dict['lattice_vectors'],
                'fnames': mini_batch_inputs_dict['fnames'],
            },
            settings_dict=settings_dict,
            tmp_poscar_dir='./tmp_poscars',
            device=device,
        )
        ##### update variables
        abc_scaler = scalers['abc_scaler']
        angle_scaler = scalers['angle_scaler']
        scaled_batch_abc, scaled_batch_angle, batch_dir_coords = learnable_parameters_dict['scaled_batch_abc'], learnable_parameters_dict['scaled_batch_angle'], learnable_parameters_dict['batch_dir_coords']
        atomic_distribution, ox_mask_learnable_tensor_per_crystal = learnable_parameters_dict['atomic_distribution'], learnable_parameters_dict['ox_mask_learnable_tensor_per_crystal']
        optimizer_lattice, optimizer_atom, optimizer_coords = optimiers_dict['optimizer_lattice'], optimiers_dict['optimizer_atom'], optimiers_dict['optimizer_coords']
        scheduler_lattice, scheduler_atom, scheduler_coords = optimiers_dict['scheduler_lattice'], optimiers_dict['scheduler_atom'], optimiers_dict['scheduler_coords']


    # End of optimization
    optimized_mini_batch_inputs_dict = mini_batch_inputs_dict | output_dict 
    optimized_mini_batch_inputs_dict['atomic_distribution'] = atomic_distribution
    optimized_mini_batch_inputs_dict['scaled_batch_abc'] = scaled_batch_abc
    optimized_mini_batch_inputs_dict['scaled_batch_angle'] = scaled_batch_angle
    optimized_mini_batch_inputs_dict['ox_mask_learnable_tensor_per_crystal'] = ox_mask_learnable_tensor_per_crystal
    optimized_mini_batch_inputs_dict['batch_dir_coords'] = batch_dir_coords
    optimized_mini_batch_inputs_dict['lattice_lr_history'] = lattice_lr_history
    optimized_mini_batch_inputs_dict['atom_lr_history'] = atom_lr_history
    optimized_mini_batch_inputs_dict['coords_lr_history'] = coords_lr_history
    if len(gap_history)>0:
        optimized_mini_batch_inputs_dict['gap_history'] = np.transpose(np.array(gap_history).squeeze(), (1, 0))
    if len(e_form_history)>0:
        optimized_mini_batch_inputs_dict['e_form_history'] = np.transpose(np.array(e_form_history).squeeze(), (1, 0))
    if len(t_loss_history)>0:
        optimized_mini_batch_inputs_dict['t_history'] = np.transpose(np.array(t_loss_history).squeeze(), (1, 0))
     
    return optimized_mini_batch_inputs_dict, scalers
    

def evaluation_for_each_batch_ALIGNN(
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
    evaluate solustions
    """

    g = optimized_mini_batch_inputs_dict['g'].to(device)
    lg = optimized_mini_batch_inputs_dict['lg'].to(device)
    atom_feat_matrix = optimized_mini_batch_inputs_dict['atom_feat_matrix'].to(device)
    radii_tensor = optimized_mini_batch_inputs_dict['radii_tensor'].to(device)
    
    print("site_ids in eval:", optimized_mini_batch_inputs_dict['site_ids'])


    with torch.no_grad():
        atomic_distribution = optimized_mini_batch_inputs_dict['atomic_distribution']
        scaled_batch_abc = optimized_mini_batch_inputs_dict['scaled_batch_abc']
        scaled_batch_angle = optimized_mini_batch_inputs_dict['scaled_batch_angle']
        ox_mask_learnable_tensor_per_crystal = optimized_mini_batch_inputs_dict['ox_mask_learnable_tensor_per_crystal']
        batch_dir_coords = optimized_mini_batch_inputs_dict['batch_dir_coords']
        optimization_targets = [scaled_batch_abc, scaled_batch_angle, batch_dir_coords, atomic_distribution, ox_mask_learnable_tensor_per_crystal]

        # calculation with atomic distribution
        distribution_prediction_dict, output_dict = forward_propagation_ALIGNN(
            optimization_targets=optimization_targets,
            fixed_inputs=optimized_mini_batch_inputs_dict,
            scalers=scalers,
            temperature=onehot_temperature,
            prediction_loss_setting_dict=prediction_loss_setting_dict,
            adding_noise_scale=adding_noise_scale,
            limit_coords_displacement=limit_coords_displacement,
            device=device,
        )
        _, distribution_loss_dict = calculate_loss_from_output(
            #bandgap_pred=bandgap_dist,
            #e_form_pred=ef_dist,
            pred_dict=distribution_prediction_dict,
            prediction_loss_setting_dict=prediction_loss_setting_dict,
            atomic_dictribution_loss_setting_dict=atomic_dictribution_loss_setting_dict,
            num_batch_crystal = settings_dict['num_batch_crystal'],
            sharpened_ox_mask=output_dict['sharpened_ox_mask'],
            normalized_dist=output_dict['normalized_dist'],
            site_ids=optimized_mini_batch_inputs_dict['site_ids'].to(device),
            radii_tensor=radii_tensor,
            device=device,
        )

        # calculation with one-hot atomic distribution
        max_val, onehot_x, onehot_atom_feat, onehot_ox_mask = calculate_onehot(output_dict, atom_feat_matrix)
        onehot_prediction_dict = ALIGNN_prediction(
            prediction_loss_setting_dict = prediction_loss_setting_dict,
            g = g.to(device),
            lg = lg.to(device),
            atomic_features = onehot_atom_feat, # one-hot atom features
            bondlength = output_dict['bondlength'],
            bond_angles = output_dict['bond_angles'],
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

    # 結果の記録
    optimized_mini_batch_inputs_dict = optimized_mini_batch_inputs_dict | output_dict
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