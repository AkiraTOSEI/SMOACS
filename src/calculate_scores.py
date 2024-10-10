from typing import Optional

import numpy as np
import pandas as pd

from .save_data import (
    check_neurality_bondlength_and_save_structure,
    optimization_history_display_for_promising_candidates,
)


def perovskite_coordinate_check(
    init_coords:np.ndarray,
    dir_coords:np.ndarray,
    size:np.ndarray,
    limit_of_displacement :float = 0.15,
    eps:float = 1E-6,
)->np.ndarray:
    """
    ペロブスカイト構造の最適化において、原子の座標が一定の範囲内に収まっているかを確認する関数。
    """
    displacement = np.minimum(
        np.abs(init_coords - dir_coords),
        np.abs(init_coords - dir_coords + 1),
        np.abs(init_coords - dir_coords - 1),
    )
    assert (size[0]==size).all()
    return (displacement <= limit_of_displacement+eps).all(axis=1).reshape(len(size),size[0]).all(axis=1)



def list2str(elements):
    if type(elements) == list:
        elements.sort()
    return str(elements)

def check_criteria_satisfactions(
        loss_setting_dict:dict,
        d: np.lib.npyio.NpzFile,
        dataframe_dict:dict,
):
    criteria_success_keys = []
    criteria_keys = []
    for key in loss_setting_dict.keys():
        criteria_max = loss_setting_dict[key]['criteria_max']
        criteria_min = loss_setting_dict[key]['criteria_min']
        if criteria_min is not None and criteria_max is not None:
            dataframe_dict[f'{key}_success'] = (criteria_min <= d[f'{key}_onehot']) & (d[f'{key}_onehot'] <= criteria_max)
        elif criteria_min is None and criteria_max is None:
            raise ValueError("criteria_min and criteria_max cannot be None at the same time.")
        elif criteria_min is None:
            dataframe_dict[f'{key}_success'] = d[f'{key}_onehot'] <= criteria_max
        elif criteria_max is None:
            dataframe_dict[f'{key}_success'] = criteria_min <= d[f'{key}_onehot']
        else:
            raise ValueError("criteria_min and criteria_max cannot be None at the same time.")
        criteria_success_keys.append(f'{key}_success')
        criteria_keys.append(f'{key}_onehot')
    return criteria_keys, criteria_success_keys

def evaluate_result(
        npz_path:str,
        csv_path:str,
        poscar_saved_dir:str,
        model_name:str,
        num_candidate:int,
        prediction_loss_setting_dict:dict,
        atomic_dictribution_loss_setting_dict:dict,
        limit_coords_displacement:Optional[float],
        history_img_path:str,
        exp_name:str,
        num_steps:int,
        neutral_check:bool,
        perovskite_evaluation:bool,
):
    optimized_key_list = list(prediction_loss_setting_dict.keys()) + list(atomic_dictribution_loss_setting_dict.keys())

    d = np.load(npz_path, allow_pickle=True)
    abc_strings, angle_strings = [], []
    for abc in d['batch_abc']:
        abc_strings.append(f"[{abc[0]:.3f}, {abc[1]:.3f}, {abc[2]:.3f}]")
    for angle in d['batch_angle']:
        angle_strings.append(f"[{angle[0]:.3f}, {angle[1]:.3f}, {angle[2]:.3f}]")
        
    dataframe_dict = {
            'lattice_index':np.arange(len(d['size'])),
            'num_atoms':d['num_atoms'],
            'original_fnames':d['original_fnames'],
            'batch_abc':abc_strings,
            'batch_angle':angle_strings,       
    }

    for key in optimized_key_list:
        dataframe_dict[f'{key}_onehot'] = d[f'{key}_onehot']
        dataframe_dict[f'loss_{key}_onehot'] = d[f'loss_{key}_onehot']


    # check the success of the criteria
    criteria_keys1, criteria_success_keys1 = check_criteria_satisfactions(prediction_loss_setting_dict, d, dataframe_dict)
    criteria_keys2, criteria_success_keys2 = check_criteria_satisfactions(atomic_dictribution_loss_setting_dict, d, dataframe_dict)
    criteria_keys = criteria_keys1 + criteria_keys2
    criteria_success_keys = criteria_success_keys1 + criteria_success_keys2
    
    df = pd.DataFrame(dataframe_dict)
    df.reset_index(inplace=True)
    df.set_index('lattice_index',inplace=True)
    df.to_csv(csv_path)
    print(len(d['lattice_vectors']), num_candidate)
    assert len(d['lattice_vectors']) == num_candidate
    #assert len(d['dir_coords']) == len(d['batch'])

    # Save the optimized structure
    df2 = check_neurality_bondlength_and_save_structure(npz_path, csv_path, poscar_saved_dir, neutral_check = neutral_check)


    # Visualize the optimization process.
    if 'bandgap' in prediction_loss_setting_dict.keys() and 'e_form' in prediction_loss_setting_dict.keys():
        print(exp_name)
        gap_min=prediction_loss_setting_dict['bandgap']['loss_function']['target_bandgap'] - prediction_loss_setting_dict['bandgap']['loss_function']['margin'],
        gap_max=prediction_loss_setting_dict['bandgap']['loss_function']['target_bandgap'] + prediction_loss_setting_dict['bandgap']['loss_function']['margin'],
        optimization_history_display_for_promising_candidates(gap_min, gap_max, npz_path, csv_path, model_name, num_display=20, history_img_path=history_img_path)

    ## check the formation energy success
    #df['Ef_success'] = df['e_form_onehot'] < E_form_criteria
    
    df = pd.concat(
        [df.reset_index().set_index('original_fnames'), df2.set_index('original_fnames')]
    ,axis=1)

    # valid structure check
    if neutral_check: 
        df['valid_structure'] = (df['is_neutral'] & df['minbond_less_than_0.5'])
    else:
        df['valid_structure'] = df['minbond_less_than_0.5']   

    if perovskite_evaluation:
        # for perovskite
        tolerance_range = atomic_dictribution_loss_setting_dict['tolerance']['loss_function']['tolerance_range']
        tolerence_bool = (min(tolerance_range) <= df['tolerance_onehot']) & (df['tolerance_onehot'] <= max(tolerance_range))
        df['tolerance_success'] = tolerence_bool
        df['perov_coords'] = perovskite_coordinate_check(d['init_coords'], d['dir_coords'], d['size'], limit_of_displacement=limit_coords_displacement)
        df['perov_success'] = ((tolerence_bool) & df['perov_coords'])
    else:
        df['tolerance_success'] = [np.nan]*df.shape[0]
        df['tolerance_onehot']  = [np.nan]*df.shape[0]
        df['perov_coords'] = [np.nan]*df.shape[0]
        df['perov_success'] = [np.nan]*df.shape[0]

    ## overall success judgement
    df['success'] = True
    for key in criteria_success_keys + ['valid_structure']:
        df[key] = df[key].astype(bool)
        df['success'] = df['success'] & df[key]
    if perovskite_evaluation:
        df['success'] = df['success'] & df['perov_success']

    
    df.to_csv(csv_path)
    print(f"success rate: {df.success.sum()/df.shape[0]:.3f} ({df.success.sum()}/{df.shape[0]})")
        
    unique_elem_list = df['elements'].apply(list2str).unique().tolist()
    if 'nan' in unique_elem_list:
        unique_elem_list.remove('nan')
    success_unique_elem_list = df.loc[df['success']==1,'elements'].apply(list2str).unique().tolist()
    additional_result_dict = {
        'num_steps':num_steps,
        'element_diversity':len(unique_elem_list)/df.shape[0],
        'success_element_diversity':len(success_unique_elem_list)/np.divide(len(success_unique_elem_list), df.loc[df['success']==1].shape[0])
    }

    result_dict = df[ ['success'] + criteria_success_keys + criteria_keys + ['valid_structure','minbond_less_than_0.5','is_neutral','perov_success','perov_coords']].mean().to_dict()
    result_dict = {'exp_name':exp_name} | result_dict | additional_result_dict

    return result_dict