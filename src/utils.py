
import itertools
import os
import shutil
from typing import Dict, List, NoReturn, Optional, Tuple

import numpy as np
import torch
from pymatgen.core.periodic_table import Element as Element_pmg
from smact import Element
from tqdm import tqdm


def create_element_list():
    VALID_ELEMENTS98 = [
        'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 
        'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y', 
        'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 
        'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 
        'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf'][:98]
    return VALID_ELEMENTS98


def set_schedule(
    num_steps: int,
    T_min :float = 0.0001,
    T_max :float = 0.01,
    ):
    # 原子のソフトマックス温度など

    # E_formの損失を追加＋softmax温度のスケジュール化
    temp_schedule = [
        T_min + (T_max - T_min) / num_steps * i for i in range(num_steps + 1)
    ][::-1]

    return temp_schedule


class Angle_Scaler():
    def __init__(self, min_angle:float, max_angle:float):
        self.min_angle = min_angle #'degree'
        self.max_angle = max_angle #'degree'
    def scale(self, angles):
        return (angles-self.min_angle)/(self.max_angle-self.min_angle)
    def rescale(self, scaled_batch_angle):
        return torch.clip(scaled_batch_angle,min=0.,max=1.)*(self.max_angle-self.min_angle)+self.min_angle


class ABC_Scaler():
    def __init__(self, init_batch_abc, min_length:float, max_length:float, device:str):
        self.max_length = (torch.max(init_batch_abc, dim=1).values).view(-1,1).to(device)
        self.eps = 1E-6
        self.min_length = min_length
        self.max_length = max_length

    def scale(self, batch_abc):
        return batch_abc/self.max_length
    def rescale(self, scaled_batch_abc):
        return torch.clip(scaled_batch_abc*self.max_length, min=self.min_length, max=self.max_length)
    


def temperature_softmax(logits, temperature=1.0, dim=-1):
    """Applies a temperature-scaled softmax to the input logits.

    This function modifies the softmax operation by scaling the logits with
    a temperature parameter before applying softmax. The temperature parameter
    can adjust the sharpness of the output distribution. A higher temperature
    makes the distribution more uniform, while a lower temperature makes it
    sharper.

    Args:
        logits (torch.Tensor): The input logits to which softmax will be applied.
        temperature (float, optional): The temperature to scale the logits. Default is 1.0.

    Returns:
        torch.Tensor: The softmax output after applying temperature scaling.

    Raises:
        ValueError: If the temperature is non-positive.

    Example:
        >>> logits = torch.tensor([2.0, 1.0, 0.1])
        >>> temperature = 0.5
        >>> softmax_outputs = temperature_softmax(logits, temperature)
        >>> print(softmax_outputs)
    """
    if temperature <= 0:
        raise ValueError("Temperature must be positive.")
    # Adjust logits based on the temperature
    adjusted_logits = logits / temperature
    # Apply softmax to the adjusted logits
    return torch.softmax(adjusted_logits, dim=dim)


def calculate_onehot(
    output_dict:Dict[str, torch.Tensor],
    atom_feat_matrix:Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Convert atomic distribution to onehot.
    '''
    max_val, max_index = torch.max(output_dict['normalized_dist'], dim=1)
    onehot_x = torch.zeros(output_dict['normalized_dist'].shape,device=output_dict['normalized_dist'].device)
    onehot_x[torch.arange(output_dict['normalized_dist'].shape[0]), max_index] = 1
    assert (torch.max(onehot_x, dim=1)[1] == max_index).all()

    _, max_ox_index = torch.max(output_dict['sharpened_ox_mask'], dim=1)
    onehot_ox_mask = torch.zeros_like(output_dict['sharpened_ox_mask'])
    onehot_ox_mask[torch.arange(output_dict['sharpened_ox_mask'].shape[0]), max_ox_index] = 1
    assert (torch.max(onehot_ox_mask, dim=1)[1] == max_ox_index).all()

    if atom_feat_matrix is not None:
        onehot_atom_feat = torch.matmul(onehot_x, atom_feat_matrix)
    else:
        onehot_atom_feat = torch.zeros_like(onehot_x)*torch.nan
    return max_val, onehot_x, onehot_atom_feat, onehot_ox_mask


def elec_neutral_check_SUPER_COMMON(num_i:int, total:int, elements: List[str], stoichs: List[List[int]], return_all_ox_states:bool=False) -> Tuple[bool, Optional[Tuple[int]]]:
    """
    Check for electrical neutrality using PyMatGen icsd_oxidation_states method by evaluating possible oxidation states combinations.

    Args:
        num_i (int): Index of the structure (for tqdm display)
        total (int): Total number of structures. (for tqdm display)
        elements (List[str]): List of element symbols.
        stoichs (List[List[int]]): List of lists containing stoichiometries.
        return_all_ox_states (bool): Whether to return all possible oxidation states combinations.

    Returns:
        Tuple[bool, Optional[Tuple[int]]]: A tuple where the first element is a boolean indicating 
                                           whether the input is electrically neutral, and the second 
                                           element is a tuple of oxidation states that make it neutral 
                                           (or None if no neutral combination is found).

    Examples:
        >>> elec_neutral_check_SUPER_COMMON(5, 10, elements=['Ti', 'O'], stoichs=[[1], [2]])
        (True, , ['Ti', 'O', 'O'], (4, -2, -2)))
        >>> elec_neutral_check_SUPER_COMMON(5, 10, elements = ['Ti', 'Al', 'O'], stoichs = [[1],[1],[1]])
        (False, ['Ti', 'Al', 'O'], None)
        >>> elec_neutral_check_SUPER_COMMON(5, 10, elements=['He', 'O'], stoichs=[[1], [2]])
        (False, ['H', 'O', 'O'], None)
    """   
    all_elements = []
    for elem, stoi in zip(elements, stoichs):
        assert len(stoi) == 1
        all_elements.extend([elem]*stoi[0])
    ox_combos = [
        list(set(Element_pmg(elem).icsd_oxidation_states) & set(Element_pmg(elem).oxidation_states) & set(Element(elem).oxidation_states) & set(Element_pmg(elem).common_oxidation_states))
        #list(set(Element_pmg(elem).icsd_oxidation_states) & set(Element_pmg(elem).oxidation_states) & set(Element(elem).oxidation_states))
        for elem in all_elements    
    ]

    # check excluding non-oxidation state elements
    if any([len(ox) == 0 for ox in ox_combos]):
        return False, all_elements, None

    lengths = np.array([len(sublist) for sublist in ox_combos])
    product_of_lengths = np.prod(lengths)

    if return_all_ox_states:
        all_neutral_ox_states = []
        for ox_states in tqdm(itertools.product(*ox_combos), total=product_of_lengths,leave=False, desc=f"neutral check ({num_i+1}/{total}) by PMG"):
            if sum(ox_states) == 0:
                all_neutral_ox_states.append(ox_states)
        return len(all_neutral_ox_states)>0, all_elements, all_neutral_ox_states

    else:
        for ox_states in tqdm(itertools.product(*ox_combos), total=product_of_lengths,leave=False, desc=f"neutral check ({num_i+1}/{total}) by PMG"):
            if sum(ox_states) == 0:
                return True, all_elements, ox_states
            
        return False, all_elements, None

def count_elements(elements_list):
    elements = {}
    for element in elements_list:
        if element not in elements:
            elements[element] = 0
        elements[element] += 1
    return elements


def remove_files_in_directory(directory: str):
    """Deletes all files and subdirectories within the specified directory.

    This function recursively deletes all files, symbolic links, and subdirectories within the specified directory.
    It outputs an error message if it fails to delete a file, symbolic link, or directory.

    Args:
        directory (str): The path to the directory from which files and directories are to be deleted.

    Raises:
        Exception: Outputs an error message if there is an issue during file deletion.
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)  # Remove file or symbolic link
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)  # Recursively remove directory
        except Exception as e:
            print(f'Failed to delete. File: {file_path}. Error: {e}')


def define_initial_dataset_dir_path(
    crystal_system: Optional[str],
    initial_dataset,
    neutral: bool,
    max_atoms: Optional[int] = 10,
    perovskite_mode: Optional[str] = None,
    perovskite_size: Optional[str] = None,
):
    raw_data_dir = os.path.join('./data/raw_data')
    
    if max_atoms is None:
        max_atoms = 100000000
        max_strings = ''
    else:
        max_strings = f'_max_{max_atoms}atoms'
    
    if neutral == 'neutral':
        suffix = '_neutral'
    elif neutral == 'common':
        suffix = '_neutral_common'
    elif neutral == 'super_common':
        suffix = '_neutral_super_common'
    elif not neutral:
        suffix = ''
    else:
        raise ValueError(f"Invalid value for neutral: {neutral}")

    if crystal_system is None:
        dir_path = os.path.join(raw_data_dir, f'initial_candidates_from_{initial_dataset}{max_strings}{suffix}')
        dataset_name = f'{initial_dataset}{max_strings}{suffix}'
    elif crystal_system == 'perovskite':
        dataset_name = f'initial_{perovskite_mode}_perovskite'
        dir_path = os.path.join(raw_data_dir, dataset_name)
    elif crystal_system in ['perovskite2x2x2', 'perovskite3x3x3', 'perovskite4x4x4']:
        size_str = f'_{crystal_system.replace("perovskite", "")}'
        dataset_name = f'initial_{perovskite_mode}_perovskite{size_str}'
        dir_path = os.path.join(raw_data_dir, dataset_name)

    else:
        raise NotImplementedError
    assert os.path.exists(dir_path), f"{dir_path} does not exist."
    
    return dir_path, dataset_name

def define_experiment_name(
    model_name:str,
    dataset_name: str,
    target_bandgap: float,
    bandgap_margin: float,
    use_formation_energy: bool,
    e_form_min: Optional[float],
    learning_rates: List[float],
    learning_cycles: List[float],
    adding_noise_scale: Optional[float],
    num_steps: int,
    num_graph_update: Optional[float],
):  
    lattice_lr, atom_lr, coords_lr = learning_rates
    lattice_cycle, atom_cycle, coords_cycle = learning_cycles
    if lattice_cycle<1:
        lattice_cycle = 0
    if atom_cycle<1:
        atom_cycle = 0
    if coords_cycle<1:
        coords_cycle = 0

    if e_form_min is None:
        e_form_suffix = ''
    else:
        e_form_suffix = f'-min{e_form_min:.4f}'
    
    if adding_noise_scale is not None:
        noise_suffix = f'__noise{adding_noise_scale:.4f}'
    else:
        noise_suffix = ''

    if num_graph_update is not None and model_name == 'ALIGNN':
        model_name = f'{model_name}up{num_graph_update}'

    exp_name = f'{model_name}__{dataset_name}__bg{target_bandgap:.2f}pm{bandgap_margin:.2f}__EfCoef{float(use_formation_energy)}{e_form_suffix}__Atomlr{atom_lr}_c{atom_cycle}__Latticelr{lattice_lr}_c{lattice_cycle}__Coordslr{coords_lr}_c{coords_cycle}__ns{num_steps}{noise_suffix}'
    return exp_name


def save_optimization_results(
    npz_path: str,
    optimized_dict_list: List[Dict[str, torch.Tensor]],
    prediction_loss_setting_dict:dict,
    atomic_dictribution_loss_setting_dict:dict,
):
    common_data_dict = {
        "lattice_vectors": torch.cat([opt_dict['lattice_vectors'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        "dir_coords": torch.cat([opt_dict['normed_batch_dir_coords'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        'init_coords': torch.cat([opt_dict['init_coords'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        "scaled_batch_abc": torch.cat([opt_dict['scaled_batch_abc'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        "scaled_batch_angle": torch.cat([opt_dict['scaled_batch_angle'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        'ox_states_used_mask': torch.cat([opt_dict['ox_states_used_mask'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        "atomic_mask": torch.cat([opt_dict['atomic_mask'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        "sharpened_ox_mask": torch.cat([opt_dict['sharpened_ox_mask'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        "atomic_distribution": torch.cat([opt_dict['atomic_distribution'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        "ox_mask_learnable_tensor_per_crystal": torch.cat([opt_dict['ox_mask_learnable_tensor_per_crystal'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        "normalized_dist": torch.cat([opt_dict['normalized_dist'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        "onehot_x": torch.cat([opt_dict['onehot_x'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        "batch_abc": torch.cat([opt_dict['batch_abc'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        "batch_angle": torch.cat([opt_dict['batch_angle'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        "init_abc": torch.cat([opt_dict['init_abc'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        "init_angles": torch.cat([opt_dict['init_angles'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        "size": torch.cat([opt_dict['size'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        "site_ids": torch.cat([opt_dict['site_ids'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        "num_atoms": torch.cat([opt_dict['size'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy(),
        "original_fnames": np.concatenate([opt_dict['fnames'] for opt_dict in optimized_dict_list]),
    }

    #for key in ['bandgap_onehot', 'bandgap_dist', 'e_form_onehot', 'e_form_dist', 'loss_bandgap_onehot', 'tolerance_onehot', 'loss_tolerance_onehot']:
    #    if key in optimized_dict_list[0].keys():
    #        common_data_dict[key] = torch.cat([opt_dict[key].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy()
        
    for key in list(prediction_loss_setting_dict.keys()) + list(atomic_dictribution_loss_setting_dict.keys()):
        common_data_dict[f'{key}_onehot'] = torch.cat([opt_dict[f'{key}_onehot'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy()
        common_data_dict[f'loss_{key}_onehot'] = torch.cat([opt_dict[f'loss_{key}_onehot'].squeeze() for opt_dict in optimized_dict_list]).detach().cpu().numpy()
        
        

    # History
    for key in ['gap_history', 'e_form_history', 't_history']:
        if key in optimized_dict_list[0].keys():
            common_data_dict[key] = np.concatenate([opt_dict[key].squeeze() for opt_dict in optimized_dict_list]).T 


    np.savez(
        npz_path,
        **common_data_dict
    )

    
def extract_conditions(experiment_name):
    """
    実験名の文字列から各条件を抽出し、辞書形式で返す関数。

    Args:
    experiment_name (str): 実験名の文字列

    Returns:
    dict: 条件が格納された辞書。マッチしない場合は空の辞書を返す。
    """
    # 条件を抽出するための正規表現パターン
    # Splitting the name into parts based on '__' and other identifiable markers
    parts = experiment_name.split('__')
    print(parts)
    model_name = parts[0]
    dataset_name = parts[1]
    bg_info = parts[2].split('bg')[1].split('pm')
    target_bandgap = float(bg_info[0])
    bandgap_margin = float(bg_info[1])
    use_formation_energy = float(parts[3].split('EfCoef')[1])
    atom_info = parts[4].split('Atomlr')[1].split('_c')
    atom_lr = float(atom_info[0])
    atom_cycle = float(atom_info[1])
    lattice_info = parts[5].split('Latticelr')[1].split('_c')
    lattice_lr = float(lattice_info[0])
    lattice_cycle = float(lattice_info[1])
    coords_info = parts[6].split('Coordslr')[1].split('_c')
    coords_lr = float(coords_info[0])
    coords_cycle = float(coords_info[1])
    num_steps = int(parts[7].split('ns')[1])

    # Constructing the dictionary
    result = {
        'model_name': model_name,
        'dataset_name': dataset_name,
        'target_bandgap': target_bandgap,
        'bandgap_margin': bandgap_margin,
        'use_formation_energy': use_formation_energy,
        'atom_lr': atom_lr,
        'atom_cycle': atom_cycle,
        'lattice_lr': lattice_lr,
        'lattice_cycle': lattice_cycle,
        'coords_lr': coords_lr,
        'coords_cycle': coords_cycle,
        'num_steps': num_steps,
    }
    
    # Handling optional noise component if exists
    if len(parts) > 8:
        noise_scale = float(parts[8].split('noise')[1])
        result['adding_noise_scale'] = noise_scale

    return result