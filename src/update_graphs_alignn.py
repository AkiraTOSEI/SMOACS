
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn

from .create_oxidation_mask import stacking_learnable_oxsides_mask
from .lattice_utils import compute_lattice_vectors, masking_atomic_distribution
from .load_data import load_initial_data
from .optimizer import define_optimizer_and_scheduler
from .save_data import save_poscar
from .utils import calculate_onehot, create_element_list, temperature_softmax


def calculate_update_steps(num_steps: int, num_update: int) -> List[int]:
    """Calculates the step indices where the sine values first exceed the specified thresholds.

    This function computes the sine value for a sequence of steps and finds the first step
    where the sine value exceeds each of a set of decreasing thresholds.

    Args:
        num_steps (int): The total number of steps to compute sine values for. Default is 200.
        num_update (int): The number of threshold values to compute, which are evenly spaced and decreasing. Default is 10.

    Returns:
        List[int]: A list of indices indicating the first step where the sine value exceeds each threshold.

    Raises:
        ValueError: If `num_update` is zero or negative, which would lead to division by zero.
    """
    if num_update <= 0:
        raise ValueError("num_update must be greater than 1 to avoid division by zero.")
        
    # Generate the step numbers
    steps = np.arange(num_steps)
    
    # Calculate the sine values
    sin_val_steps = np.sin(np.pi / 2 * steps/num_steps)
    
    # Calculate update thresholds
    update_thres = np.linspace(0, 1, num_update+1, endpoint=False)[1:]
    
    # Find the first step that exceeds each threshold
    update_steps = []
    for thres in update_thres:
        idx = np.argmax(sin_val_steps > thres)
        update_steps.append(idx if sin_val_steps[idx] > thres else -1)  # -1 if no step exceeds the threshold
    
    return update_steps

def matching_by_hash_table(
    a:np.ndarray, 
    b:np.ndarray
)->List[int]:

    """ A function to find the indices where each row of a and b match
    """
    # ハッシュテーブルを作成
    hash_map = {tuple(row): idx for idx, row in enumerate(a)}

    # bの各行に対して対応するaのインデックスを見つける
    matching_indices = [hash_map[tuple(row)] for row in b]
    return matching_indices


def get_alignment_ids(
    fnames:np.ndarray,
    batch_abc:torch.Tensor,
    batch_angle:torch.Tensor,
    batch_dir_coords:torch.Tensor,
    atomic_distribution:torch.Tensor,
    size:torch.Tensor,
    updated_fnames:np.ndarray,
    updated_batch_abc:np.ndarray,
    updated_batch_dir_coords:np.ndarray,
    updated_size:np.ndarray,
):
    '''
    Once the POSCAR is created and converted to a graph, the positions of the atoms change. Therefore, an ID array is created to match the order of the original POSCAR and the optimized order.
    '''

    # Get lattice alignment id array
    ### all fnames are unique
    assert len(fnames) == np.unique(fnames).shape[0]
    assert len(updated_fnames) == np.unique(updated_fnames).shape[0]
    # Convert the index of the old data to the index of the new data
    lattice_alignment_array = np.array([list(fnames).index(fn) for fn in updated_fnames])
    ### Check if the order of the data is the same
    assert (fnames[lattice_alignment_array] == updated_fnames).all()
    ### Check if the data is the same
    #nan_data_bool = ~torch.isnan(batch_abc[lattice_alignment_array].detach().cpu()).any(axis=1)
    lattice_vectors = compute_lattice_vectors(batch_abc, batch_angle)
    assert lattice_vectors.shape == torch.Size([len(batch_abc), 3, 3])
    nan_data_bool = (torch.isnan(lattice_vectors).sum(dim=[1,2])==0).detach().cpu()
    torch.testing.assert_close(batch_abc[lattice_alignment_array][nan_data_bool].detach().cpu(), updated_batch_abc[nan_data_bool].detach().cpu())
    lattice_vectors = lattice_vectors.detach().cpu().numpy()
    assert (size[lattice_alignment_array] == updated_size).all()

    # compute lattice vectors
    

    # get atom-wise alignment id
    coordinate_lattice_id = np.concatenate([np.array([lattice_id]*num) for lattice_id, num in enumerate(size)])
    aligned_size = size.detach().cpu().numpy()[lattice_alignment_array] 
    aligned_coordinate_lattice_id = np.concatenate([np.array([lattice_id]*num) for lattice_id, num in enumerate(aligned_size)])
    ### get the index of the new data in the old data
    updated_batch_dir_coords = updated_batch_dir_coords.detach().cpu().numpy()
    ### Get the index to align the atomic order for each crystal)
    atom_wise_alignment_list = []
    no_nan_bool = np.ones(shape=(len(batch_dir_coords),), dtype=bool)
    for lattice_id in range(len(size)):
        ### get new lattice id
        aligned_lattice_id = lattice_alignment_array[lattice_id]
        ### get atom-wise lattice id
        aligned_atom_bool = aligned_coordinate_lattice_id==aligned_lattice_id
        atom_bool = coordinate_lattice_id==lattice_id
        ### get lattice data
        lattice_abc = batch_abc[lattice_id].detach().cpu().numpy()
        lattice_angle = batch_angle[lattice_id].detach().cpu().numpy()
        lattice_batch_dir_coords = batch_dir_coords[atom_bool].detach().cpu().numpy()
        lattice_atom_distribution = atomic_distribution[atom_bool].detach().cpu().numpy()
        if np.isnan(lattice_vectors[lattice_id]).any() or np.isnan(lattice_batch_dir_coords).any() or np.isnan(lattice_abc).any() or np.isnan(lattice_angle).any() or np.isnan(lattice_atom_distribution).any():
            ### if nan exists, the alignment order is no longer valid
            print(f"lattice_id: {fnames[lattice_id]} has nan")
            index = np.arange(len(lattice_batch_dir_coords))
            no_nan_bool[atom_bool] = False
        else:
            ### get the index to align the atomic order
            index = np.array(matching_by_hash_table(lattice_batch_dir_coords, updated_batch_dir_coords[aligned_atom_bool]))
        index = list(index + len(atom_wise_alignment_list))
        atom_wise_alignment_list.extend(index)
        
    atom_wise_alignment_array = np.array(atom_wise_alignment_list)

    assert (batch_dir_coords[atom_wise_alignment_array].detach().cpu().numpy()[no_nan_bool] == updated_batch_dir_coords[no_nan_bool]).all()

    return lattice_alignment_array, atom_wise_alignment_array, no_nan_bool


def get_alignment_ids_from_current_structure(
    optimized_mini_batch_inputs_dict:Dict[str, Any],
    settings_dict:Dict[str, Any],
    tmp_poscar_dir:str = './tmp_poscars',
):


    ### update graph data
    all_inputs_dict = load_initial_data(
        settings_dict = settings_dict,
        model_name = 'ALIGNN',
        dir_path = tmp_poscar_dir,
        mask_data_path=settings_dict['mask_data_path'],
        radii_data_path=settings_dict['radii_data_path'],
        max_ox_states = settings_dict['max_ox_states'],
        num_candidate = settings_dict['num_batch_crystal'],
        batch_size = settings_dict['num_batch_crystal'],
        use_atomic_mask = settings_dict['use_atomic_mask'],
        test_mode = settings_dict['test_mode'],
        angle_range = settings_dict['angle_range'],
        abc_range = settings_dict['abc_range'],
        graph_update_mode = True, # graph更新モード、原子マスクを作成しない
        device = settings_dict['device'],
        
        #file_type = 'cif'
    )
    updated_minibatch_dict, updated_scalers = all_inputs_dict[0]

    # delete unnecessary data that is replaced by the original data
    del updated_minibatch_dict['atomic_distribution'], updated_minibatch_dict['ox_mask_learnable_tensor_per_crystal']
    del updated_minibatch_dict['ox_states_used_mask'], updated_minibatch_dict['atomic_mask']

    ### get alignment ids
    lattice_alignment_array, atom_wise_alignment_array, no_nan_bool = get_alignment_ids(
        fnames=optimized_mini_batch_inputs_dict['fnames'],
        batch_abc=optimized_mini_batch_inputs_dict['batch_abc'],
        batch_angle=optimized_mini_batch_inputs_dict['batch_angle'],
        batch_dir_coords=torch.remainder(optimized_mini_batch_inputs_dict['batch_dir_coords'],1.),
        atomic_distribution=optimized_mini_batch_inputs_dict['atomic_distribution'],
        size=optimized_mini_batch_inputs_dict['size'],
        updated_fnames=updated_minibatch_dict['fnames'],
        updated_batch_abc=updated_minibatch_dict['batch_abc'],
        updated_batch_dir_coords=updated_minibatch_dict['batch_dir_coords'],
        updated_size=updated_minibatch_dict['size'],
    )
    return lattice_alignment_array, atom_wise_alignment_array, updated_minibatch_dict, updated_scalers, no_nan_bool


def update_graphs_and_trainable_parameters(
    optimized_mini_batch_inputs_dict:Dict[str, Any],
    settings_dict:Dict[str, Any],
    tmp_poscar_dir:str,
    device:str,
    onehot_temperature:float = 1E-8,

):
    '''
    Updated tranable parameters
    '''
    # get settings parameters
    learning_rates = settings_dict['learning_rates']
    learning_rate_cycle = settings_dict['learning_rate_cycle']
    num_steps = settings_dict['num_steps']
    angle_optimization = settings_dict['angle_optimization']
    length_optimization = settings_dict['length_optimization']
    print(" ----------------------------\n   <info> Update graph and trainable parameters....")
    
    ### save current structures to poscar files
    save_poscar(optimized_mini_batch_inputs_dict, tmp_poscar_dir)

    # get alignment ids
    lattice_alignment_array, atom_wise_alignment_array, updated_minibatch_dict, updated_scaler, no_nan_bool = get_alignment_ids_from_current_structure(
        optimized_mini_batch_inputs_dict=optimized_mini_batch_inputs_dict,
        settings_dict=settings_dict,
        tmp_poscar_dir=tmp_poscar_dir,
    )
    # get data from optimized_mini_batch_inputs_dict
    atomic_distribution = optimized_mini_batch_inputs_dict['atomic_distribution']
    ox_mask_learnable_tensor_per_crystal = optimized_mini_batch_inputs_dict['ox_mask_learnable_tensor_per_crystal']
    ox_states_used_mask = optimized_mini_batch_inputs_dict['ox_states_used_mask']
    atomic_mask = optimized_mini_batch_inputs_dict['atomic_mask']
    site_ids = optimized_mini_batch_inputs_dict['site_ids']
    radii_tensor = optimized_mini_batch_inputs_dict['radii_tensor']
    init_coords = optimized_mini_batch_inputs_dict['init_coords']
    

    # align current atomic distribution for the updated structure
    updated_minibatch_dict['atomic_distribution'] = atomic_distribution[atom_wise_alignment_array]
    updated_minibatch_dict['ox_mask_learnable_tensor_per_crystal'] = ox_mask_learnable_tensor_per_crystal[lattice_alignment_array]
    updated_minibatch_dict['ox_states_used_mask'] = ox_states_used_mask[atom_wise_alignment_array] 

    updated_minibatch_dict['atomic_mask'] = atomic_mask[atom_wise_alignment_array]
    updated_minibatch_dict['site_ids'] = site_ids[atom_wise_alignment_array]
    updated_minibatch_dict['radii_tensor'] = radii_tensor[atom_wise_alignment_array]
    updated_minibatch_dict['init_coords'] = init_coords[atom_wise_alignment_array]
    size = updated_minibatch_dict['size']
    assert (updated_minibatch_dict['size'] == optimized_mini_batch_inputs_dict['size'][lattice_alignment_array]).all()


    # get onehot atomic distribution
    ### masking and sharpening
    stacked_learnable_ox_weight = stacking_learnable_oxsides_mask(updated_minibatch_dict['ox_mask_learnable_tensor_per_crystal'].clone().detach().cpu(), size)
    atomic_distribution, sharpened_ox_mask = masking_atomic_distribution(
        updated_minibatch_dict['atomic_distribution'].clone().detach().cpu(), 
        updated_minibatch_dict['atomic_mask'].clone().detach().cpu(),
        updated_minibatch_dict['ox_states_used_mask'].clone().detach().cpu(),
        stacked_learnable_ox_weight, 
        onehot_temperature)
    quasi_onehot_dist = temperature_softmax(atomic_distribution, temperature=onehot_temperature)
    ### get onehot atomic distribution
    _, onehot_x, _, _= calculate_onehot(
        {
            'normalized_dist':quasi_onehot_dist,
            'sharpened_ox_mask':sharpened_ox_mask,
       }, 
       updated_minibatch_dict['atom_feat_matrix']
    )  

    ### Check if the elements match the updated structure
    VALID_ELEMENTS98 = create_element_list()
    updated_elements1 = [VALID_ELEMENTS98[a_idx-1] for a_idx in updated_minibatch_dict['g'].ndata['atomic_number'].detach().cpu().numpy().astype(int).reshape(-1)]
    updated_elements2 = [VALID_ELEMENTS98[a_idx] for a_idx in torch.argmax(onehot_x,axis=1).detach().cpu().numpy().astype(int).reshape(-1)]
    assert (np.array(updated_elements1)[no_nan_bool] == np.array(updated_elements2)[no_nan_bool]).all()

    # Get updated trainable parameters etc.
    abc_scaler = updated_scaler['abc_scaler']
    angle_scaler = updated_scaler['angle_scaler']
    atomic_distribution = nn.Parameter(updated_minibatch_dict['atomic_distribution'].clone().to(device), requires_grad=True)
    scaled_batch_abc = nn.Parameter(abc_scaler.scale(updated_minibatch_dict['batch_abc'].clone().to(device)), requires_grad=True)
    scaled_batch_angle = nn.Parameter(angle_scaler.scale(updated_minibatch_dict['batch_angle'].clone().to(device)), requires_grad=True)
    batch_dir_coords = nn.Parameter(updated_minibatch_dict['batch_dir_coords'].clone().to(device), requires_grad=True)
    ox_mask_learnable_tensor_per_crystal = nn.Parameter(updated_minibatch_dict['ox_mask_learnable_tensor_per_crystal'].to(device).clone(), requires_grad=True)
    updated_minibatch_dict['radii_tensor'] = updated_minibatch_dict['radii_tensor'].to(device)
    updated_minibatch_dict['site_ids'] = updated_minibatch_dict['site_ids'].to(device)
    learnable_parameters_dict = {
        'scaled_batch_abc':scaled_batch_abc,
        'scaled_batch_angle':scaled_batch_angle,
        'batch_dir_coords':batch_dir_coords,
        'atomic_distribution':atomic_distribution,
        'ox_mask_learnable_tensor_per_crystal':ox_mask_learnable_tensor_per_crystal,
    }


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
        length_optimization=length_optimization ,
        num_steps=num_steps,
        lattice_cycle=learning_rate_cycle[0],
        atom_cycle=learning_rate_cycle[1],
        coords_cycle=learning_rate_cycle[2],
    )
    optimiers_dict = {
        'optimizer_lattice':optimizer_lattice,
        'optimizer_atom':optimizer_atom,
        'optimizer_coords':optimizer_coords,
        'scheduler_lattice':scheduler_lattice,
        'scheduler_atom':scheduler_atom,
        'scheduler_coords':scheduler_coords,
    }
    print("    <info> End Updating graph and trainable parameters....\n ----------------------------")

    return updated_minibatch_dict, updated_scaler, optimiers_dict, learnable_parameters_dict
