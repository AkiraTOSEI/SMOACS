import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from .alignn_for_inverse_problem import Load_Pretrained_ALIGNN
from .calculate_scores import evaluate_result
from .crystalformer_for_inverse_problem import Load_Pretrained_Crystalformers
from .load_data import load_initial_data
from .loss import loss_function_initialization
from .optimize_and_evaluate_solutions_for_alignn import (
    evaluation_for_each_batch_ALIGNN,
    optimize_solution_ALIGNN,
)
from .optimize_and_evaluate_solutions_for_crystalformer import (
    evaluation_for_each_batch_Crystalformer,
    optimize_solution_Crystalformer,
)
from .update_graphs_alignn import calculate_update_steps
from .utils import remove_files_in_directory, save_optimization_results, set_schedule


#if __name__ == '__main__':
def main_experiment(
    dir_path:str, 
    mask_data_path:str, 
    radii_data_path:str, 
    exp_name:str, 
    prediction_loss_setting_dict:dict,
    atomic_dictribution_loss_setting_dict:dict,
    num_steps:int=200,
    model_name = 'Crystalformer', #'Crystalformer' or 'ALIGNN' or 'Both'
    site_atom_oxidation_dict:Optional[List[Dict[str, List[int]]]]=None,
    num_candidate:int = 256, # max 4096
    num_batch_crystal:int = 256, # max 4096
    atom_lr:float=2.0,
    lattice_lr:float=0.003,
    coords_lr:float=0.005,
    atom_cycle:int=1,
    lattice_cycle:int=1,
    coords_cycle:int=1,
    neutral_check:bool=True,
    limit_coords_displacement:Optional[float]=None,
    angle_range = (30, 150),
    abc_range = (2.0, 10.0),
    max_ox_states:int = 10,
    adding_noise_scale = None, # None or float
    test_mode:bool=False,
    perovskite_evaluation:bool=False,
    ALIGNN_num_update_graphs: Optional[int]  = None,
    angle_optimization:bool=True,
    length_optimization:bool=True,
):



    # ----------------- Hyperparameters ----------------- #
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # -----------------  load pretrained models ------------------ #
    if model_name == 'Crystalformer':
        optimize_solution = optimize_solution_Crystalformer
        evaluation_for_each_batch = evaluation_for_each_batch_Crystalformer
        load_pretrained = Load_Pretrained_Crystalformers
    elif model_name == 'ALIGNN':
        optimize_solution = optimize_solution_ALIGNN
        evaluation_for_each_batch = evaluation_for_each_batch_ALIGNN
        load_pretrained = Load_Pretrained_ALIGNN
    else:
        raise NotImplementedError
    prediction_loss_setting_dict = load_pretrained(prediction_loss_setting_dict)


    # -----------------  initialize models and loss functions ------------------ #
    loss_function_initialization(prediction_loss_setting_dict, atomic_dictribution_loss_setting_dict)

    # -----------------  Settings ------------------ #
    settings_dict = {}
    settings_dict['mask_data_path'] = mask_data_path
    settings_dict['radii_data_path'] = radii_data_path
    settings_dict['max_ox_states'] = max_ox_states
    settings_dict['num_candidate'] = num_candidate
    settings_dict['num_batch_crystal'] = num_batch_crystal
    settings_dict['use_atomic_mask'] = True
    settings_dict['test_mode'] = test_mode
    settings_dict['learning_rates'] = [lattice_lr, atom_lr, coords_lr]
    settings_dict['learning_rate_cycle'] = [lattice_cycle, atom_cycle, coords_cycle]
    settings_dict['num_steps'] = num_steps
    settings_dict['angle_optimization'] = angle_optimization
    settings_dict['length_optimization'] = length_optimization
    settings_dict['test_mode'] = test_mode
    settings_dict['angle_range'] = angle_range
    settings_dict['abc_range'] = abc_range
    settings_dict['device'] = device
    settings_dict['site_atom_oxidation_dict'] = site_atom_oxidation_dict
    #### for ALIGNN graph update
    if  ALIGNN_num_update_graphs is not None:
        settings_dict['ALIGNN_update_step'] = calculate_update_steps(num_steps=num_steps, num_update=ALIGNN_num_update_graphs)
    else:
        settings_dict['ALIGNN_update_step'] = []


    # -----------------  Define Saved Directories ------------------ #
    result_dir = f'./results/{exp_name}'
    os.makedirs(result_dir, exist_ok=True)
    npz_path = os.path.join(result_dir, 'result.npz')
    csv_path = os.path.join(result_dir, 'result.csv')
    poscar_saved_dir = os.path.join(result_dir ,'poscar')
    os.makedirs(poscar_saved_dir, exist_ok=True)
    history_img_path = os.path.join(result_dir, 'history_img.png')
    for saved_dir_path in [poscar_saved_dir]:
        if os.path.exists(saved_dir_path):
            remove_files_in_directory(saved_dir_path)


    # -----------------  Load Data ------------------ #
    all_inputs_dict = load_initial_data(
        settings_dict=settings_dict,
        model_name = model_name,
        dir_path = dir_path,
        mask_data_path = settings_dict['mask_data_path'],
        radii_data_path = settings_dict['radii_data_path'],
        max_ox_states = settings_dict['max_ox_states'],
        num_candidate = settings_dict['num_candidate'],
        batch_size = settings_dict['num_batch_crystal'],
        use_atomic_mask = settings_dict['use_atomic_mask'],
        test_mode = settings_dict['test_mode'],
        angle_range = settings_dict['angle_range'],
        abc_range = settings_dict['abc_range'],
        device=settings_dict['device'],
    )            
    # -----------------  Optimization ------------------ #
    optimized_dict_list = []
    for mini_batch_data in tqdm(all_inputs_dict,desc="mini_batch",leave=False):
        ### select batch candidate crystals
        mini_batch_inputs_dict, scalers = mini_batch_data

        ### softmax temparature schedule
        temp_schedule = set_schedule(num_steps=num_steps)

        ### Optimization
        optimized_mini_batch_inputs_dict, scalers = optimize_solution(
            settings_dict=settings_dict,
            mini_batch_inputs_dict=mini_batch_inputs_dict,
            learning_rates = [lattice_lr, atom_lr, coords_lr],
            learning_rate_cycle = [lattice_cycle, atom_cycle, coords_cycle],
            num_steps=num_steps,
            temp_schedule=temp_schedule,
            angle_optimization=angle_optimization,
            length_optimization=length_optimization,
            prediction_loss_setting_dict=prediction_loss_setting_dict,
            atomic_dictribution_loss_setting_dict=atomic_dictribution_loss_setting_dict,
            adding_noise_scale=adding_noise_scale,
            scalers=scalers,
            limit_coords_displacement=limit_coords_displacement,
            device=device,
        )
        ### Evaluation
        optimized_mini_batch_inputs_dict = evaluation_for_each_batch(
            optimized_mini_batch_inputs_dict=optimized_mini_batch_inputs_dict,
            atomic_dictribution_loss_setting_dict=atomic_dictribution_loss_setting_dict,
            scalers=scalers,
            settings_dict=settings_dict,
            prediction_loss_setting_dict=prediction_loss_setting_dict,
            limit_coords_displacement=limit_coords_displacement,
            adding_noise_scale=None,
            device=device,
        )

        
        optimized_dict_list.append(optimized_mini_batch_inputs_dict)


    # -----------------  Save and Visualization ------------------ #
    save_optimization_results(
        npz_path = npz_path,
        optimized_dict_list = optimized_dict_list,
        prediction_loss_setting_dict = prediction_loss_setting_dict,
        atomic_dictribution_loss_setting_dict = atomic_dictribution_loss_setting_dict,
    )

    
    # -----------------  Evaluation and Calulate success rate ------------------ #
    result_dict = evaluate_result(
        npz_path=npz_path,
        csv_path=csv_path,
        poscar_saved_dir=poscar_saved_dir,
        neutral_check=neutral_check,
        model_name=model_name,
        num_candidate=num_candidate,
        prediction_loss_setting_dict=prediction_loss_setting_dict,
        atomic_dictribution_loss_setting_dict=atomic_dictribution_loss_setting_dict,
        limit_coords_displacement=limit_coords_displacement,
        history_img_path=history_img_path,
        exp_name=exp_name,
        num_steps=num_steps,
        perovskite_evaluation = perovskite_evaluation,
    )

    pd.DataFrame(result_dict, index=[0]).to_csv(os.path.join(result_dir, 'result_summary.csv'))


