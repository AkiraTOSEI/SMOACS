import os
import shutil

import numpy as np
import pandas as pd
import torch
from jarvis.core.atoms import Atoms
from matplotlib import colors as mcolors
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from .create_oxidation_mask import stacking_learnable_oxsides_mask
from .lattice_utils import (
    check_nearest_neighbor,
    compute_abc_angle,
    compute_lattice_vectors,
    masking_atomic_distribution,
)
from .utils import (
    calculate_onehot,
    count_elements,
    create_element_list,
    elec_neutral_check_SUPER_COMMON,
    temperature_softmax,
)


def save_poscar(
    optimized_mini_batch_inputs_dict:dict, 
    tmp_poscar_dir:str,
    onehot_temperature:float = 1E-8
):
    '''
    function to save the structure during optimization.
    '''
    ### crean up tmp_poscar_dir
    if os.path.exists(tmp_poscar_dir):
        shutil.rmtree(tmp_poscar_dir)
    os.makedirs(tmp_poscar_dir, exist_ok=True)

    ### load data from optimized_mini_batch_inputs_dict
    batch_abc = optimized_mini_batch_inputs_dict['batch_abc'].detach().cpu()
    batch_angle = optimized_mini_batch_inputs_dict['batch_angle'].detach().cpu()
    atomic_distribution = optimized_mini_batch_inputs_dict['atomic_distribution'].detach().cpu()
    batch_dir_coords = optimized_mini_batch_inputs_dict['batch_dir_coords'].detach().cpu().numpy()
    normed_batch_dir_coords = np.remainder(batch_dir_coords, 1.) # 内部座標の値を周期的境界条件で0~1に制限する
    size = optimized_mini_batch_inputs_dict['size'].detach().cpu().numpy()
    fnames = optimized_mini_batch_inputs_dict['fnames']
    atomic_mask = optimized_mini_batch_inputs_dict['atomic_mask'].detach().cpu()
    ox_states_used_mask = optimized_mini_batch_inputs_dict['ox_states_used_mask'].detach().cpu()
    atom_feat_matrix = optimized_mini_batch_inputs_dict['atom_feat_matrix'].detach().cpu()
    ox_mask_learnable_tensor_per_crystal = optimized_mini_batch_inputs_dict['ox_mask_learnable_tensor_per_crystal'].detach().cpu()

    # compute lattice vectors
    lattice_vectors = compute_lattice_vectors(batch_abc, batch_angle).numpy()
    batch_abc = batch_abc.numpy()
    batch_angle = batch_angle.numpy()
    
    # Get elements
    ### masking
    if ox_mask_learnable_tensor_per_crystal is not None :
        stacked_learnable_ox_weight = stacking_learnable_oxsides_mask(ox_mask_learnable_tensor_per_crystal, size)
        normalized_dist, sharpened_ox_mask = masking_atomic_distribution(atomic_distribution, atomic_mask, ox_states_used_mask, stacked_learnable_ox_weight, onehot_temperature)
    else:
        normalized_dist = temperature_softmax(atomic_distribution, temperature=onehot_temperature)
        
    ### get onehot atomic distribution
    _, onehot_x, _, _= calculate_onehot({
        'normalized_dist':normalized_dist,
        'sharpened_ox_mask':sharpened_ox_mask,
       }, atom_feat_matrix)

    ### create element symbol list
    VALID_ELEMENTS98 = create_element_list()
    element_ids = np.argmax(onehot_x.numpy(), axis=1)
    elements = [VALID_ELEMENTS98[e_id] for e_id in element_ids]
    ### get atom-wise lattice id
    coordinate_lattice_id = np.concatenate([np.array([lattice_id]*num) for lattice_id, num in enumerate(optimized_mini_batch_inputs_dict['size'])])

    # save current poscar
    for lattice_id in range(len(size)):
        atom_dic  = {
                    'lattice_mat': [list(vec) for vec in lattice_vectors[lattice_id]],
                    'coords': [list(vec) for vec in normed_batch_dir_coords[coordinate_lattice_id==lattice_id]],
                    'elements': np.array(elements)[coordinate_lattice_id==lattice_id].tolist(),
                    'abc':batch_abc[lattice_id].tolist(),
                    'angles': batch_angle[lattice_id].tolist(),
                    'cartesian': False,
                    'props':['']*len(batch_dir_coords[coordinate_lattice_id==lattice_id])

        }
        ### if nan exists, use initial poscar. Otherwise, use optimized poscar
        if np.isnan(lattice_vectors[lattice_id]).any() or np.isnan(batch_abc[lattice_id]).any() or np.isnan(batch_angle[lattice_id]).any() or np.isnan(atomic_distribution[coordinate_lattice_id==lattice_id]).any() or np.isnan(batch_dir_coords[coordinate_lattice_id==lattice_id]).any():
            print(f"lattice_id: {fnames[lattice_id]} has nan. use initial poscar")
            shutil.copy(f"./init_poscars/{fnames[lattice_id]}", os.path.join(tmp_poscar_dir,f"{fnames[lattice_id]}"))
        else:
            Atoms.from_dict(atom_dic).write_poscar(os.path.join(tmp_poscar_dir,f"{fnames[lattice_id]}")) 



def optimization_history_display_for_promising_candidates(
    gap_min:float,
    gap_max:float,
    npz_path:str,
    csv_path:str,
    model_name:str,
    num_display:int,
    history_img_path:str
    ):
    d = np.load(npz_path, allow_pickle=True)
    # formation energyが低く、指定のバンドギャップを満たすものから順
    df = pd.read_csv(csv_path)
    if 'loss_bandgap_onehot' in df.columns and 'e_form_onehot' in df.columns:
        sorted_index = df.sort_values(['loss_bandgap_onehot','e_form_onehot'])['lattice_index'].values
    else:
        return None
    num_display = min(num_display, len(sorted_index))

    title_font_size, label_font_size, legend_font_size, tick_font_size = 16, 14, 14, 14  # 目盛りのフォントサイズを追加
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))  # 1行2列、図のサイズは12x5インチ

    
    colors_list = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.BASE_COLORS.values())+list(mcolors.CSS4_COLORS.values())
    # Bandgap optimization historyのプロット（最初のサブプロット）
    if model_name != 'Both':
        axs[0].plot(d['gap_history'][:,sorted_index[:num_display]]) 
        axs[0].hlines(gap_min, 0, d['gap_history'].shape[0], linestyles='dashed', colors='black', label='target area')
        axs[0].hlines(gap_max, 0, d['gap_history'].shape[0], linestyles='dashed', colors='black',)
    else:
        for i in range(num_display):
            color = colors_list[i]
            axs[0].plot(d['gap_history_alignn'][:,sorted_index[i]], color=color, linestyle='-')#, label=f'alignn_{i}')
            axs[0].plot(d['gap_history_crystalformer'][:,sorted_index[i]], color=color, linestyle='-.')
        axs[0].hlines(gap_min, 0, d['gap_history_alignn'].shape[0], linestyles='dashed', colors='black', label='target area')
        axs[0].hlines(gap_max, 0, d['gap_history_alignn'].shape[0], linestyles='dashed', colors='black')
        axs[0].plot([], color='gray', linestyle='-',label='alignn')
        axs[0].plot([], color='gray', linestyle='-.',label='crystalformer')
    
    axs[0].set_title('Bandgap optimization history', fontsize=title_font_size)  # タイトルのフォントサイズ
    axs[0].set_xlabel('step', fontsize=label_font_size)  # x軸ラベルのフォントサイズ
    axs[0].set_ylabel('bandgap (eV)', fontsize=label_font_size)  # y軸ラベルのフォントサイズ
    axs[0].tick_params(axis='both', labelsize=tick_font_size)  # x軸とy軸の目盛りのフォントサイズ

    axs[0].legend()  # 凡例の表示

    # Formation energy optimization historyのプロット（2番目のサブプロット）
    if model_name != 'Both':
        axs[1].plot(d['e_form_history'][:,sorted_index[:num_display]])
    else:
        for i in range(num_display):
            color = colors_list[i]
            axs[1].plot(d['e_form_history_alignn'][:,sorted_index[i]], color=color, linestyle='-')
            axs[1].plot(d['e_form_history_crystalformer'][:,sorted_index[i]], color=color, linestyle='-.')
    axs[1].set_title('Formation energy optimization history', fontsize=title_font_size)  # タイトルのフォントサイズ
    axs[1].set_xlabel('step', fontsize=label_font_size)  # x軸ラベルのフォントサイズ
    axs[1].set_ylabel('formation energy (eV/at.)', fontsize=label_font_size)  # y軸ラベルのフォントサイズ
    axs[1].tick_params(axis='both', labelsize=tick_font_size)  # x軸とy軸の目盛りのフォントサイズ

    plt.tight_layout()  # レイアウトの調整
    if os.path.dirname(history_img_path) != '':
        os.makedirs(os.path.dirname(history_img_path), exist_ok=True)
    plt.savefig(history_img_path)
    plt.close()
    figure = plt.figure(figsize=(10, 10))
    img = Image.open(history_img_path)
    plt.imshow(img)
    plt.show()



def check_neurality_bondlength_and_save_structure(
    npz_path:str,
    csv_path:str,
    saved_dir:str,
    neutral_check:bool = True,
    ) -> pd.DataFrame:
    """
    最適化された構造を保存する関数
    """

    # clean the directory
    for file in os.listdir(saved_dir):
        os.remove(os.path.join(saved_dir, file))
    
    VALID_ELEMENTS98=create_element_list()
    os.makedirs(f'{saved_dir}', exist_ok=True)

    d = np.load(npz_path, allow_pickle=True)

    # formation energyが低く、指定のバンドギャップを満たすものから順に並べる
    if 'loss_bandgap_onehot' in pd.read_csv(csv_path).columns and 'e_form_onehot' in pd.read_csv(csv_path).columns:
        sorted_df = pd.read_csv(csv_path).sort_values(['loss_bandgap_onehot','e_form_onehot'])
    elif  'loss_bandgap_onehot' in pd.read_csv(csv_path).columns:
        sorted_df = pd.read_csv(csv_path).sort_values(['loss_bandgap_onehot'])
    elif 'e_form_onehot' in pd.read_csv(csv_path).columns:
        sorted_df = pd.read_csv(csv_path).sort_values(['e_form_onehot'])
    else:
        sorted_df = pd.read_csv(csv_path)
    sorted_index = sorted_df['lattice_index'].values
    
    opt_abc, opt_angle = compute_abc_angle(torch.tensor(d['lattice_vectors']))
    coordinate_lattice_id = np.concatenate([np.array([lattice_id]*num) for lattice_id, num in enumerate(d['num_atoms'])])

    os.makedirs(f'{saved_dir}', exist_ok=True)
    os.makedirs('./cif', exist_ok=True)

    neutralities = []
    for num_i, lattice_id in enumerate(tqdm(sorted_index, desc='save_structure', total=len(sorted_index))):

        # exclude nan data
        if np.isnan(d['dir_coords'][coordinate_lattice_id==lattice_id]).any() or np.isnan(d['lattice_vectors'][lattice_id]).any():
            continue

        # Neutral check
        elements_list = [VALID_ELEMENTS98[atom_idx] for atom_idx in np.argmax(d['onehot_x'][coordinate_lattice_id==lattice_id], axis=1)]
        atoms, stoichs = [], []
        for _atom, _stoich in count_elements(elements_list).items():
            atoms.append(_atom)
            stoichs.append([_stoich])

        if neutral_check:
            is_neutral, all_elements, ox_states = elec_neutral_check_SUPER_COMMON(num_i, len(sorted_index), atoms, stoichs, True)
            neutral_ok = is_neutral
        else:
            is_neutral, all_elements, ox_states = np.nan, [np.nan], [np.nan]
            neutral_ok = True

        # create Atoms object
        atom_dic  = {
            'lattice_mat': [list(vec) for vec in d['lattice_vectors'][lattice_id]],
            'coords': [list(vec) for vec in d['dir_coords'][coordinate_lattice_id==lattice_id]],
            'elements': elements_list,
            'abc':opt_abc[lattice_id].detach().cpu().numpy().tolist(),
            'angles': opt_angle[lattice_id].detach().cpu().numpy().tolist(),
            'cartesian': False,
            'props':['']*len(d['dir_coords'][coordinate_lattice_id==lattice_id])
        }
        assert len(atom_dic['elements']) == len(atom_dic['coords'])
        assert len(atom_dic['elements']) == len(atom_dic['props'])


        atoms_data = Atoms.from_dict(atom_dic)

        # bondlength check 
        bondlength_check = check_nearest_neighbor(atoms_data)

        # add to data dictionary
        neutralities.append(
            {   
                'original_fnames':d['original_fnames'][lattice_id],
                'elements':atoms,
                'is_neutral':is_neutral,
                'elements':all_elements,
                'ox_states':ox_states,
                'minbond_less_than_0.5':bondlength_check,
                'one_atom':len(atoms)==1,
            }
        )        

        # structure saved
        fname = f"Optimized_based_on_{d['original_fnames'][lattice_id]}"
        atoms_data.write_poscar(os.path.join(saved_dir, f'{fname}'))


    return pd.DataFrame(neutralities)