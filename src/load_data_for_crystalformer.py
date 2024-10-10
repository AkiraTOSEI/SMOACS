import copy
import glob
import os
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from jarvis.core.atoms import Atoms

from .create_oxidation_mask import (
    create_learnable_oxides_mask,
    get_atom_masks_for_oxidation_states,
)
from .utils import ABC_Scaler, Angle_Scaler


def create_crystalformer_init_structure_from_directory(
    dir_path:str,
    mask_data_path:str,
    radii_data_path:Optional[str],
    max_ox_states:int,
    site_atom_oxidation_dict:Dict[str,Any],
    num_candidate:Optional[int] = None,
):
    """
    (ja) dir_pathにあるPOSCARファイルを読み込んで、初期構造を作成する
    (en) Read the POSCAR files in dir_path and create the initial structure
    """


    print("<info> load initial structures from", dir_path)
    mask_d = dict(np.load(mask_data_path), allow_pickle=True)
    if radii_data_path is not None:
        radii_d = dict(np.load(radii_data_path), allow_pickle=True)
    else:
        radii_d = None

    batch_list, size_list, angles_list, coords_list, abc_list, atom_dist_list,fnames = [], [], [], [], [], [],[]
    atom_mask_for_all_ox_states_list, radii_for_all_ox_states_list, ox_states_used_mask_list, site_ids_list = [], [], [], []
    num_total_crystals = 0
    
    file_paths = glob.glob(os.path.join(dir_path, '*.vasp'))
    if num_candidate is not None:
        file_paths = file_paths[:num_candidate]

    for cry_i, path in enumerate(file_paths):
        atoms = Atoms.from_poscar(path)
        batch_list.append(torch.ones(len(atoms.elements),dtype=torch.long)*num_total_crystals)
        size_list.append(len(atoms.elements))
        angles_list.append(atoms.lattice.angles)
        coords_list.append(torch.tensor(atoms.coords))
        abc_list.append(atoms.lattice.abc)
        atom_dist_list.append(torch.ones((len(atoms.elements), 98))/98)
        fnames.append(os.path.basename(path))

        _atom_mask_for_all_ox_states, _radii_for_all_ox_states, _ox_states_used_mask, _site_ids = get_atom_masks_for_oxidation_states(
            atoms=atoms,
            mask_d=mask_d,
            max_ox_states=max_ox_states,
            site_atom_oxidation_dict=site_atom_oxidation_dict,
            graph_update_mode=False,
            radii_d=radii_d
        )
        atom_mask_for_all_ox_states_list.append(_atom_mask_for_all_ox_states)
        radii_for_all_ox_states_list.append(_radii_for_all_ox_states)
        ox_states_used_mask_list.append(torch.stack([_ox_states_used_mask]*len(atoms.elements),dim=0))
        site_ids_list.append(_site_ids)
        num_total_crystals += 1

    batch = torch.concat(batch_list,dim=0).to(torch.long)
    size = torch.tensor(size_list).to(torch.long)
    angles = torch.tensor(angles_list).to(torch.float32)
    coords = torch.concat(coords_list, dim=0).to(torch.float32)
    abc = torch.tensor(abc_list).to(torch.float32)
    atom_dist = torch.concat(atom_dist_list,dim=0) # 一様分布で初期化
    atom_mask_for_all_ox_states = torch.concat(atom_mask_for_all_ox_states_list).type(torch.get_default_dtype())
    radii_for_all_ox_states = torch.concat(radii_for_all_ox_states_list).type(torch.get_default_dtype())
    ox_states_used_mask = torch.concat(ox_states_used_mask_list, dim=0).type(torch.get_default_dtype())
    site_ids = torch.concat(site_ids_list, dim=0)

    assert torch.sum(size) == coords.shape[0] == atom_dist.shape[0] == len(batch)
    assert len(size) == len(angles) == len(abc)

    return abc, atom_dist, batch, coords, size, angles, fnames, atom_mask_for_all_ox_states, radii_for_all_ox_states, ox_states_used_mask, site_ids


def load_crystalformer_initial_data(
    settings_dict:Dict[str,Any],
    dir_path:str,
    mask_data_path:str,
    radii_data_path:Optional[str],
    batch_size:int,
    use_atomic_mask:bool,
    max_ox_states:int,
    angle_range:Tuple[float,float],
    abc_range:Tuple[float,float],
    device:str,
    test_mode:bool,
    num_candidate:Optional[int] = None,

    ):

    out_data = create_crystalformer_init_structure_from_directory(
        dir_path = dir_path, 
        mask_data_path = mask_data_path,
        radii_data_path = radii_data_path,
        max_ox_states = max_ox_states,
        num_candidate = num_candidate,
        site_atom_oxidation_dict = settings_dict['site_atom_oxidation_dict'],
    )

    all_opt_abc, all_opt_x, all_batch, all_coords4input, all_size, all_angles, fnames, atom_mask_for_all_ox_states, radii_for_all_ox_states, ox_states_used_mask, site_ids = out_data
    crystalformer_all_inputs_dict ={
        'x':all_opt_x,
        'batch':all_batch,
        'coords4input':all_coords4input,
        'size':all_size,
        'abc':all_opt_abc,
        'angles':all_angles,
        'init_coords':copy.deepcopy(all_coords4input),
        'init_abc':copy.deepcopy(all_opt_abc),
        'init_angles':copy.deepcopy(all_angles),
        'atomic_mask':None,
        'radii_tensor':None,
        'atom_mask_for_all_ox_states':atom_mask_for_all_ox_states,
        'radii_for_all_ox_states':radii_for_all_ox_states,
        'ox_states_used_mask':ox_states_used_mask,
        'site_ids':site_ids,
        'fnames':fnames,
    }

    if use_atomic_mask:
        crystalformer_all_inputs_dict['atomic_mask'] = atom_mask_for_all_ox_states.to(device)
        ox_mask_learnable_tensor_per_crystal  = create_learnable_oxides_mask(crystalformer_all_inputs_dict)
        crystalformer_all_inputs_dict['ox_mask_learnable_tensor_per_crystal'] = ox_mask_learnable_tensor_per_crystal
        crystalformer_all_inputs_dict['radii_tensor'] = radii_for_all_ox_states.to(device)

    # divide the initial data into small batches
    minibatch_datas = []
    for min_id in range(0, all_opt_abc.shape[0], batch_size):
        # select batch candidate crystals
        mini_batch_inputs_dict = select_batch_candidates_crystalformer(
            min_id=min_id,
            num_batch_crystal=batch_size,
            crystalformer_all_inputs_dict=crystalformer_all_inputs_dict
        )
        scalers = {
            "abc_scaler": ABC_Scaler(
                init_batch_abc=mini_batch_inputs_dict['batch_abc'],
                min_length=abc_range[0],
                max_length=abc_range[1],
                device=device
            ),
            "angle_scaler": Angle_Scaler(
                min_angle=angle_range[0],
                max_angle=angle_range[1],
            )
        }
        minibatch_datas.append(
            [mini_batch_inputs_dict, scalers]
        )

    return minibatch_datas


def pickup_candidates_crystalformer(
    min_id:int, 
    max_id:int,
    crystalformer_all_inputs_dict: dict,
    ):
    assert min_id < max_id

    x = crystalformer_all_inputs_dict['x']
    batch = crystalformer_all_inputs_dict['batch']
    coords4input = crystalformer_all_inputs_dict['coords4input']
    size = crystalformer_all_inputs_dict['size']
    abc = crystalformer_all_inputs_dict['abc']
    angles = crystalformer_all_inputs_dict['angles']
    atomic_mask = crystalformer_all_inputs_dict['atomic_mask']
    radii_tensor = crystalformer_all_inputs_dict['radii_tensor']
    fnames = np.array(crystalformer_all_inputs_dict['fnames'])
    ox_mask_learnable_tensor_per_crystal = crystalformer_all_inputs_dict['ox_mask_learnable_tensor_per_crystal']
    ox_states_used_mask = crystalformer_all_inputs_dict['ox_states_used_mask']
    site_ids = crystalformer_all_inputs_dict['site_ids']


    selected_atom_ids = (min_id <= batch.detach().cpu().numpy())&(batch.detach().cpu().numpy()< max_id) 
    selected_x = x[selected_atom_ids]
    selected_batch = batch[selected_atom_ids]
    selected_coords4input = coords4input[selected_atom_ids]
    selected_site_ids = site_ids[selected_atom_ids]

    unique_ids = np.unique(batch.detach().cpu().numpy())
    selected_ids = (min_id <= unique_ids)&(unique_ids < max_id)
    selected_abc= abc[selected_ids]
    selected_angles= angles[selected_ids]
    selected_size = size[selected_ids]
    selected_fnames = fnames[selected_ids]
    

    if atomic_mask is not None:
        selected_atomic_mask = atomic_mask[selected_atom_ids]
        selected_ox_mask_learnable_tensor_per_crystal = ox_mask_learnable_tensor_per_crystal[selected_ids]
        selected_ox_states_used_mask = ox_states_used_mask[selected_atom_ids]
    else:
        selected_atomic_mask = None
        selected_ox_mask_learnable_tensor_per_crystal = None
        selected_ox_states_used_mask = None
        
    if radii_tensor is not None:
        selected_radii_tensor = radii_tensor[selected_atom_ids]
    else:
        selected_radii_tensor = None

    minibatch_inputs_dict = {
        'atomic_distribution': selected_x,
        'batch': selected_batch,
        'batch_dir_coords': selected_coords4input,
        'size': selected_size,
        'batch_abc': selected_abc,
        'batch_angle' : selected_angles,
        'init_coords':copy.deepcopy(selected_coords4input),
        'init_abc':copy.deepcopy(selected_abc),
        'init_angles':copy.deepcopy(selected_angles),
        'fnames': selected_fnames,
        'atomic_mask':selected_atomic_mask,
        'radii_tensor':selected_radii_tensor,
        'ox_mask_learnable_tensor_per_crystal':selected_ox_mask_learnable_tensor_per_crystal,
        'ox_states_used_mask':selected_ox_states_used_mask,
        'site_ids':selected_site_ids,
    }
    
    return minibatch_inputs_dict


def select_batch_candidates_crystalformer(
    min_id:int,
    num_batch_crystal:int,
    crystalformer_all_inputs_dict: dict,
    ):
    '''
    Divide the optimization target into small chunks, i.e. mini-batches
    Args:
        min_id: The minimum ID of the optimization target to be sorted
        num_batch_crystal: The number of crystals to be optimized at once in the optimization
        crystalformer_all_inputs_dict: Input to Crystalformer
    Returns:
        opt_x: distribution of atomic species of the optimization target. shape = (size, num_atom_type) 
        batch: ID of the optimization target crystal. shape = (num_batch_crystal,)
        coords4input: Lattice coordinates of the atoms to be optimized. shape =(total number of atoms in num_batch_crystal, 3)
        size: Number of atoms in the optimization target crystal shape = (num_batch_crystal,)
        opt_abc: Length of the lattice vectors of the optimization target crystal. shape = (num_batch_crystal, 3)
        angles: Angle of the lattice vectors of the optimization target crystal.. shape = (num_batch_crystal, 3)
    '''

    max_id = min_id+num_batch_crystal
    mini_batch_inputs_dict = pickup_candidates_crystalformer(
        min_id=min_id, 
        max_id=max_id,
        crystalformer_all_inputs_dict=crystalformer_all_inputs_dict
    )
    return mini_batch_inputs_dict

