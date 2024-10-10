import glob
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from jarvis.core.atoms import Atoms
from torch.utils.data import DataLoader

from .alignn_preprocess import get_torch_dataset, graph_to_tensors
from .create_oxidation_mask import get_atom_masks_for_oxidation_states
from .utils import ABC_Scaler, Angle_Scaler


def create_ALIGNN_init_structure_from_directory(
    dir_path:str,
    mask_data_path:str,
    radii_data_path:Optional[str],
    max_ox_states:int,
    site_atom_oxidation_dict:Optional[dict],
    num_candidate:Optional[int] = None,
    graph_update_mode:bool = False,
    ):
    
    
    mask_d = dict(np.load(mask_data_path, allow_pickle=True))
    if radii_data_path is not None:
        radii_d = dict(np.load(radii_data_path, allow_pickle=True))
    else:
        radii_d = None

    print("<info> load initial structures from", dir_path)
    dataset_optimize = []
    num_total_crystals = 0
    file_paths = glob.glob(os.path.join(dir_path, '*.vasp'))
    
    if num_candidate is not None:
        file_paths = file_paths[:num_candidate]
    
    for path in file_paths:
        atoms = Atoms.from_poscar(path)

        atom_mask_for_all_ox_states, radii_for_all_ox_states, ox_states_used_mask, site_ids = get_atom_masks_for_oxidation_states(
            atoms=atoms,
            mask_d=mask_d,
            max_ox_states=max_ox_states,
            site_atom_oxidation_dict=site_atom_oxidation_dict,
            graph_update_mode=graph_update_mode,
            radii_d=radii_d
        )
        dataset_optimize.append({
            'jid':os.path.basename(path),
            'atoms':{
                'lattice_mat':atoms.lattice.matrix,
                'coords':atoms.coords,
                "elements":atoms.elements,
                'abc':atoms.lattice.abc,
                'angles':atoms.lattice.angles,
                'props':atoms.props,
                'cartesian':False,

            },
            'dummy_target':0, # 0 はダミー
            'elements':atoms.elements,
            'atom_mask_for_all_ox_states':atom_mask_for_all_ox_states,
            'radii_for_all_ox_states':radii_for_all_ox_states,
            'ox_states_used_mask':ox_states_used_mask,
            'site_ids':site_ids,
        })


        
    num_total_crystals = len(dataset_optimize)
    return dataset_optimize, num_total_crystals


def load_ALIGNN_initial_data(
        settings_dict:Dict[str,Any],
        dir_path:str,
        mask_data_path:str,
        radii_data_path:Optional[str],
        batch_size:int,
        use_atomic_mask:bool,
        max_ox_states:int,
        device:str,
        angle_range:Tuple[float,float],
        abc_range:Tuple[float,float],
        num_candidate:Optional[int] = None,
        max_neighbors:int = 12,
        cutoff:float = 8.0,
        cutoff_extra:float = 3.0,
        test_mode:bool = True,
        graph_update_mode:bool = False,
    ):

    # load the dataset from the directory
    dataset_optimize, num_total = create_ALIGNN_init_structure_from_directory(
        dir_path=dir_path,
        mask_data_path=mask_data_path,
        radii_data_path=radii_data_path,
        max_ox_states=max_ox_states,
        num_candidate=num_candidate,
        site_atom_oxidation_dict=settings_dict['site_atom_oxidation_dict'],
        graph_update_mode = graph_update_mode,
    )

    # create the dataset for the ALIGNN model
    train_data, train_df = get_torch_dataset(
        dataset=dataset_optimize,
        id_tag='jid',
        atom_features="cgcnn",
        target="dummy_target",
        target_atomwise="",
        target_grad="",
        target_stress="",
        neighbor_strategy= "k-nearest",
        use_canonize=True,
        name='dft_3d',
        line_graph=True,
        cutoff=cutoff,
        cutoff_extra=cutoff_extra,
        max_neighbors=max_neighbors,
        classification=False,
        output_dir="./",
        tmp_name="train_data",
    )

    # create the dataloader for the ALIGNN model
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_data.collate_line_graph,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    # create the ALIGNN input data
    ALIGNN_all_inputs = []
    for i, g_x in enumerate(train_loader):
        g = g_x[0]
        lg = g_x[1]

        g = g.local_var()
        lg = lg.local_var()
        # initial node features: atom feature network...
        atom_features = g.ndata.pop("atom_features")
        bondlength = torch.norm(g.edata.pop("r"), dim=1)
        angle_features = lg.edata.pop("h")

        data_dict_for_test = {
            'atom_features':atom_features,
            'bondlength':bondlength,
            'angle_features':angle_features,
        }
        
        alignn_minibatch_dict = graph_to_tensors(
            g=g, 
            lg=lg, 
            train_df=train_df.loc[i*batch_size:(i+1)*batch_size-1], 
            data_dict_for_test=data_dict_for_test, 
            device=device, 
            use_atomic_mask=use_atomic_mask,
            test_mode=test_mode
        )

        scalers = {
            "abc_scaler": ABC_Scaler(
                init_batch_abc=alignn_minibatch_dict['batch_abc'],
                min_length=abc_range[0],
                max_length=abc_range[1],
                device=device,
            ),
            "angle_scaler": Angle_Scaler(
                min_angle=angle_range[0],
                max_angle=angle_range[1],
            ),
        }

        ALIGNN_all_inputs.append(
            [alignn_minibatch_dict, scalers]
        )
        #if i*batch_size>=num_total:
        #    break


    return ALIGNN_all_inputs
