import os
import shutil
from typing import Optional, Dict, Tuple

import numpy as np
from jarvis.core.atoms import Atoms
import torch

from src.utils.common import create_element_list
from src.utils.coord import compute_lattice_vectors
from src.utils.feature_utils import calculate_onehot
from src.utils.mask_utils import (
    masking_atomic_distribution,
    stacking_learnable_oxsides_mask,
)
from src.utils.nn_utils import temperature_softmax
from src.spg_wp.wyckoff_expander import WyckoffExpander
from src.preprocess.prepro_utils import compute_normalized_dist_and_ox_mask


def save_poscar(
    optimized_mini_batch_inputs_dict, 
    tmp_poscar_dir,
    wyckoff_expander: Optional[WyckoffExpander],
    onehot_temperature:float = 1E-8
):
    """
    Save optimized crystal structures as POSCAR files.

    This function converts batched crystal optimization results into 
    per-structure POSCAR files, using one-hot atomic type selection and 
    reconstructed lattice vectors. If invalid values (NaNs) are found 
    in the structure data, the original initial POSCAR is copied instead.

    Args:
        optimized_mini_batch_inputs_dict (Dict): Dictionary containing optimized outputs, including:
            - 'batch_abc', 'batch_angle', 'atomic_distribution', 'batch_dir_coords', 'size', etc.
        tmp_poscar_dir (str): Path to directory where POSCAR files will be saved.
        onehot_temperature (float, optional): Temperature used to compute one-hot atomic distributions. Default is 1E-8.

    Notes:
        - This function assumes initial POSCARs are stored under `./init_poscars/`.
        - Atomic types are selected via softmax masking and converted to element symbols.
        - Only fractional coordinates are used; outputs are not in Cartesian format.
    """
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
    wp_size = optimized_mini_batch_inputs_dict['wp_size'].detach().cpu().numpy()
    fnames = optimized_mini_batch_inputs_dict['fnames']
    atomic_mask = optimized_mini_batch_inputs_dict['atomic_mask'].detach().cpu()
    ox_states_used_mask = optimized_mini_batch_inputs_dict['ox_states_used_mask'].detach().cpu()
    atom_feat_matrix = optimized_mini_batch_inputs_dict['atom_feat_matrix'].detach().cpu()
    site_ids = optimized_mini_batch_inputs_dict['site_ids'].detach().cpu().numpy()
    ox_mask_learnable_tensor_per_crystal = optimized_mini_batch_inputs_dict['ox_mask_learnable_tensor_per_crystal'].detach().cpu()

    # compute lattice vectors
    lattice_vectors = compute_lattice_vectors(batch_abc, batch_angle).numpy()
    batch_abc = batch_abc.numpy()
    batch_angle = batch_angle.numpy()
    
    # Compute normalized atomic distribution and sharpened oxidation state mask.
    normalized_dist, sharpened_ox_mask = compute_normalized_dist_and_ox_mask(
        atomic_distribution=atomic_distribution,
        atomic_mask=atomic_mask,
        ox_states_used_mask=ox_states_used_mask,
        temperature=onehot_temperature,
        size=wp_size,
        ox_mask_learnable_tensor_per_crystal=ox_mask_learnable_tensor_per_crystal,  # can be None
    )
    # Expand Wyckoff positions if a WyckoffExpander is provided.
    if wyckoff_expander is not None:
        device = wyckoff_expander.device
        normed_batch_dir_coords , normalized_dist = wyckoff_expander.wyckoff_expand(
            wp_coords = torch.tensor(normed_batch_dir_coords).to(device),
            wp_atom_dist = torch.tensor(normalized_dist).to(device),
            site_ids = torch.tensor(site_ids).to(device),
        )
        normed_batch_dir_coords = normed_batch_dir_coords.detach().cpu()
        normalized_dist = normalized_dist.detach().cpu()
    assert np.sum(size) == normalized_dist.shape[0] == normed_batch_dir_coords.shape[0], f"n_nodes: {n_nodes}, normalized_dist.shape[0]: {normalized_dist.shape[0]}, normed_batch_dir_coords.shape[0]: {normed_batch_dir_coords.shape[0]}"


    ### get onehot atomic distribution
    _, onehot_x, _, _= calculate_onehot(
        {
            "normalized_dist": normalized_dist,
            "sharpened_ox_mask": sharpened_ox_mask,
            "atomic_distribution": atomic_distribution,
            "atomic_mask": atomic_mask,
            "site_ids": torch.tensor(site_ids),
            "masking_temperature": onehot_temperature,
            "wyckoff_expander": wyckoff_expander,
        },
        atom_feat_matrix,
    )

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
                    'props':['']*len(normed_batch_dir_coords[coordinate_lattice_id==lattice_id])

        }
        ### if nan exists, use initial poscar. Otherwise, use optimized poscar
        if np.isnan(lattice_vectors[lattice_id]).any() or np.isnan(batch_abc[lattice_id]).any() or np.isnan(batch_angle[lattice_id]).any() or np.isnan(normalized_dist[coordinate_lattice_id==lattice_id]).any() or np.isnan(normed_batch_dir_coords[coordinate_lattice_id==lattice_id]).any():
            print(f"lattice_id: {fnames[lattice_id]} has nan. use initial poscar")
            shutil.copy(f"./init_poscars/{fnames[lattice_id]}", os.path.join(tmp_poscar_dir,f"{fnames[lattice_id]}"))
        else:
            Atoms.from_dict(atom_dic).write_poscar(os.path.join(tmp_poscar_dir,f"{fnames[lattice_id]}")) 
