import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from jarvis.core.atoms import Atoms
from torch.utils.data import DataLoader

from src.graph.graph_utils import get_torch_dataset, graph_to_tensors
from src.utils.oxidation import get_atom_masks_for_oxidation_states
from src.utils.scalers import ABC_Scaler, Angle_Scaler


def create_ALIGNN_init_structure_from_directory(
    dir_path: str,
    intermid_dir_path: str,
    max_ox_states: int,
    mask_method: str,
    crystal_system: Optional[str],
    initial_dataset: str,
    num_candidate: Optional[int] = None,
    graph_update_mode: bool = False,
):
    print("<info> load initial structures from", dir_path)
    if crystal_system is None:
        print("<info> crystal_system is None")
        mask_d = dict(
            np.load(
                os.path.join(intermid_dir_path, f"mask_dict_{mask_method}.npz"),
                allow_pickle=True,
            )
        )
        radii_d = None
    elif crystal_system == "perovskite":
        print("<info> crystal_system is perovskite")
        mask_d = dict(
            np.load(
                os.path.join(intermid_dir_path, f"ionic_mask_dict_{mask_method}.npz"),
                allow_pickle=True,
            )
        )
        radii_d = dict(
            np.load(
                os.path.join(intermid_dir_path, f"ionic_radii_dict_{mask_method}.npz"),
                allow_pickle=True,
            )
        )

    dataset_optimize = []
    num_total_crystals = 0
    file_paths = glob.glob(os.path.join(dir_path, "*.vasp"))
    for path in file_paths:
        atoms = Atoms.from_poscar(path)

        (
            atom_mask_for_all_ox_states,
            radii_for_all_ox_states,
            ox_states_used_mask,
            site_ids,
        ) = get_atom_masks_for_oxidation_states(
            atoms=atoms,
            mask_d=mask_d,
            radii_d=radii_d,
            max_ox_states=max_ox_states,
            crystal_system=crystal_system,
            initial_dataset=initial_dataset,
            graph_update_mode=graph_update_mode,
        )
        dataset_optimize.append(
            {
                "jid": os.path.basename(path),
                "atoms": {
                    "lattice_mat": atoms.lattice.matrix,
                    "coords": atoms.coords,
                    "elements": atoms.elements,
                    "abc": atoms.lattice.abc,
                    "angles": atoms.lattice.angles,
                    "props": atoms.props,
                    "cartesian": False,
                },
                "dummy_target": 0,  # 0 はダミー
                "elements": atoms.elements,
                "atom_mask_for_all_ox_states": atom_mask_for_all_ox_states,
                "radii_for_all_ox_states": radii_for_all_ox_states,
                "ox_states_used_mask": ox_states_used_mask,
                "site_ids": site_ids,
            }
        )

    if num_candidate is not None:
        dataset_optimize = dataset_optimize[:num_candidate]

    num_total_crystals = len(dataset_optimize)
    return dataset_optimize, num_total_crystals


def load_ALIGNN_initial_data(
    settings_dict: Dict[str, Any],
    dir_path: str,
    intermid_dir_path: str,
    batch_size: int,
    use_atomic_mask: bool,
    max_ox_states: int,
    mask_method: str,
    crystal_system: Optional[str],
    initial_dataset: str,
    angle_range: Tuple[float, float],
    abc_range: Tuple[float, float],
    num_candidate: Optional[int] = None,
    max_neighbors: int = 12,
    cutoff: float = 8.0,
    cutoff_extra: float = 3.0,
    base_lattice_index: int = 0,
    test_mode: bool = True,
    graph_update_mode: bool = False,
) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]:
    """
    Load and preprocess a batch of crystal structures for ALIGNN training.

    This function reads crystal structures from VASP files, converts them into DGL graphs and line graphs,
    extracts bond and angle features, and prepares batched tensor inputs and scaling utilities
    (abc and angle scalers) for ALIGNN training.

    Args:
        settings_dict (Dict): General configuration dictionary (may be unused here).
        dir_path (str): Path to the directory containing .vasp structure files.
        intermid_dir_path (str): Path to intermediate mask and radii files.
        batch_size (int): Number of crystals to include in each batch.
        use_atomic_mask (bool): Whether to use oxidation-state-based atomic masks.
        max_ox_states (int): Maximum number of oxidation states to consider per element.
        mask_method (str): Identifier for mask file naming.
        crystal_system (Optional[str]): Crystal system filter (e.g., 'perovskite' or None).
        angle_range (Tuple[float, float]): Min and max bond angles for normalization.
        abc_range (Tuple[float, float]): Min and max lattice length values for normalization.
        num_candidate (Optional[int]): Maximum number of structures to use (if specified).
        max_neighbors (int): Max neighbors per atom in graph construction.
        cutoff (float): Distance cutoff for neighbor search.
        cutoff_extra (float): Extra distance margin.
        base_lattice_index (int): Placeholder (not used).
        test_mode (bool): If True, test assertions are run to validate tensors.
        graph_update_mode (bool): If True, graphs are recomputed during optimization.

    Returns:
        List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]: A list of ALIGNN-compatible batches,
        each consisting of (input_tensor_dict, scaler_dict).
    """

    dataset_optimize, num_total = create_ALIGNN_init_structure_from_directory(
        dir_path=dir_path,
        crystal_system=crystal_system,
        intermid_dir_path=intermid_dir_path,
        max_ox_states=max_ox_states,
        num_candidate=num_candidate,
        mask_method=mask_method,
        initial_dataset=initial_dataset,
        graph_update_mode=graph_update_mode,
    )

    train_data, train_df = get_torch_dataset(
        dataset=dataset_optimize,
        id_tag="jid",
        atom_features="cgcnn",
        target="dummy_target",
        target_atomwise="",
        target_grad="",
        target_stress="",
        neighbor_strategy="k-nearest",
        use_canonize=True,
        name="dft_3d",
        line_graph=True,
        cutoff=cutoff,
        cutoff_extra=cutoff_extra,
        max_neighbors=max_neighbors,
        classification=False,
        output_dir="./",
        tmp_name="train_data",
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_data.collate_line_graph,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

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
            "atom_features": atom_features,
            "bondlength": bondlength,
            "angle_features": angle_features,
        }

        alignn_minibatch_dict = graph_to_tensors(
            g=g,
            lg=lg,
            train_df=train_df.loc[i * batch_size : (i + 1) * batch_size - 1],
            data_dict_for_test=data_dict_for_test,
            use_atomic_mask=use_atomic_mask,
            test_mode=test_mode,
        )

        scalers = {
            "abc_scaler": ABC_Scaler(
                init_batch_abc=alignn_minibatch_dict["batch_abc"],
                min_length=abc_range[0],
                max_length=abc_range[1],
                device="cuda",
            ),
            "angle_scaler": Angle_Scaler(
                min_angle=angle_range[0],
                max_angle=angle_range[1],
            ),
        }

        ALIGNN_all_inputs.append([alignn_minibatch_dict, scalers])
        # if i*batch_size>=num_total:
        #    break

    return ALIGNN_all_inputs
