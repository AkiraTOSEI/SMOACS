import copy
import glob
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from jarvis.core.atoms import Atoms

from src.utils.mask_utils import create_learnable_oxides_mask
from src.utils.oxidation import get_atom_masks_for_oxidation_states
from src.utils.scalers import ABC_Scaler, Angle_Scaler


def pickup_candidates_crystalformer(
    min_id: int,
    max_id: int,
    crystalformer_all_inputs_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Selects atoms and crystal-level information for a given batch range.

    Given start and end indices of crystal IDs, this function extracts the corresponding subset
    of atomic features, coordinates, lattice parameters, masks, and identifiers. It supports
    both masked and unmasked modes, returning a dictionary of batched inputs.

    Args:
        min_id (int): Minimum crystal ID to include (inclusive).
        max_id (int): Maximum crystal ID to include (exclusive).
        crystalformer_all_inputs_dict (Dict[str, Any]): Full dataset dictionary for Crystalformer.

    Returns:
        Dict[str, Any]: Dictionary of selected atom and crystal information, containing:
            - 'atomic_distribution': Batched atomic distributions.
            - 'batch': Batch indices per atom.
            - 'batch_dir_coords': Fractional coordinates of atoms.
            - 'size': Number of atoms per structure.
            - 'batch_abc': Lattice lengths (a, b, c).
            - 'batch_angle': Lattice angles (alpha, beta, gamma).
            - 'init_coords', 'init_abc', 'init_angles': Cloned versions for optimization.
            - 'fnames': Structure file names or identifiers.
            - 'atomic_mask', 'radii_tensor', 'ox_mask_learnable_tensor_per_crystal', 'ox_states_used_mask': Optional mask data.
            - 'site_ids': Site type indicators (e.g., A/B/O site).
    """
    assert min_id < max_id

    x = crystalformer_all_inputs_dict["x"]
    batch = crystalformer_all_inputs_dict["batch"]
    coords4input = crystalformer_all_inputs_dict["coords4input"]
    size = crystalformer_all_inputs_dict["size"]
    abc = crystalformer_all_inputs_dict["abc"]
    angles = crystalformer_all_inputs_dict["angles"]
    atomic_mask = crystalformer_all_inputs_dict["atomic_mask"]
    radii_tensor = crystalformer_all_inputs_dict["radii_tensor"]
    fnames = np.array(crystalformer_all_inputs_dict["fnames"])
    ox_mask_learnable_tensor_per_crystal = crystalformer_all_inputs_dict[
        "ox_mask_learnable_tensor_per_crystal"
    ]
    ox_states_used_mask = crystalformer_all_inputs_dict["ox_states_used_mask"]
    site_ids = crystalformer_all_inputs_dict["site_ids"]

    selected_atom_ids = (min_id <= batch.detach().cpu().numpy()) & (
        batch.detach().cpu().numpy() < max_id
    )
    selected_x = x[selected_atom_ids]
    selected_batch = batch[selected_atom_ids]
    selected_coords4input = coords4input[selected_atom_ids]
    selected_site_ids = site_ids[selected_atom_ids]

    unique_ids = np.unique(batch.detach().cpu().numpy())
    selected_ids = (min_id <= unique_ids) & (unique_ids < max_id)
    selected_abc = abc[selected_ids]
    selected_angles = angles[selected_ids]
    selected_size = size[selected_ids]
    selected_fnames = fnames[selected_ids]

    if atomic_mask is not None:
        selected_atomic_mask = atomic_mask[selected_atom_ids]
        selected_ox_mask_learnable_tensor_per_crystal = (
            ox_mask_learnable_tensor_per_crystal[selected_ids]
        )
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
        "atomic_distribution": selected_x,
        "batch": selected_batch,
        "batch_dir_coords": selected_coords4input,
        "size": selected_size,
        "batch_abc": selected_abc,
        "batch_angle": selected_angles,
        "init_coords": copy.deepcopy(selected_coords4input),
        "init_abc": copy.deepcopy(selected_abc),
        "init_angles": copy.deepcopy(selected_angles),
        "fnames": selected_fnames,
        "atomic_mask": selected_atomic_mask,
        "radii_tensor": selected_radii_tensor,
        "ox_mask_learnable_tensor_per_crystal": selected_ox_mask_learnable_tensor_per_crystal,
        "ox_states_used_mask": selected_ox_states_used_mask,
        "site_ids": selected_site_ids,
    }

    return minibatch_inputs_dict


def select_batch_candidates_crystalformer(
    min_id: int,
    num_batch_crystal: int,
    crystalformer_all_inputs_dict: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Select a subset of candidate crystals and their corresponding data for a single optimization batch.

    This function extracts a slice of the full dataset, returning a dictionary of atomic coordinates,
    lattice vectors, atomic distributions, masks, and other relevant features required by Crystalformer.

    Args:
        min_id (int): Index of the first crystal to include in the batch.
        num_batch_crystal (int): Number of crystals to include in the batch.
        crystalformer_all_inputs_dict (Dict[str, Any]): Dictionary containing full dataset of inputs for Crystalformer.

    Returns:
        Dict[str, Any]: A dictionary containing the batched inputs for the selected crystals.
    """
    max_id = min_id + num_batch_crystal
    mini_batch_inputs_dict = pickup_candidates_crystalformer(
        min_id=min_id,
        max_id=max_id,
        crystalformer_all_inputs_dict=crystalformer_all_inputs_dict,
    )
    return mini_batch_inputs_dict


def create_crystalformer_init_structure_from_directory(
    dir_path: str,
    intermid_dir_path: str,
    max_ox_states: int,
    mask_method: str,
    crystal_system: Optional[str],
    initial_dataset: str,
    num_candidate: Optional[int] = None,
) -> Tuple[
    torch.Tensor,  # abc
    torch.Tensor,  # atomic_distribution
    torch.Tensor,  # batch
    torch.Tensor,  # coords
    torch.Tensor,  # size
    torch.Tensor,  # angles
    List[str],  # fnames
    torch.Tensor,  # atom_mask_for_all_ox_states
    torch.Tensor,  # radii_for_all_ox_states
    torch.Tensor,  # ox_states_used_mask
    torch.Tensor,  # site_ids
]:
    """
    Loads initial crystal structures from VASP-format files and prepares input tensors for Crystalformer.

    This function reads POSCAR (.vasp) files in the specified directory, constructs atomic coordinate tensors,
    lattice parameters, atomic distributions, oxidation masks, and other auxiliary inputs. These tensors are
    returned for use as initialization in Crystalformer-based optimization.

    Args:
        dir_path (str): Path to the directory containing POSCAR (.vasp) files.
        intermid_dir_path (str): Path to directory containing oxidation mask and radius .npz files.
        max_ox_states (int): Maximum number of oxidation states considered per atom.
        mask_method (str): Identifier used to select the appropriate mask file (e.g., 'default', 'ionic').
        crystal_system (Optional[str]): Crystal system constraint (e.g., 'perovskite'), or None.
        num_candidate (Optional[int]): If specified, limits the number of candidate structures returned.

    Returns:
        Tuple containing:
            - abc (torch.Tensor): Lattice lengths (a, b, c) for each structure. Shape: (N_crystals, 3).
            - atomic_distribution (torch.Tensor): Uniform atomic distribution per atom. Shape: (N_atoms, 98).
            - batch (torch.Tensor): Batch ID for each atom. Shape: (N_atoms,).
            - coords (torch.Tensor): Cartesian coordinates of atoms. Shape: (N_atoms, 3).
            - size (torch.Tensor): Number of atoms in each crystal. Shape: (N_crystals,).
            - angles (torch.Tensor): Lattice angles (alpha, beta, gamma). Shape: (N_crystals, 3).
            - fnames (List[str]): List of structure file names.
            - atom_mask_for_all_ox_states (torch.Tensor): Oxidation-state masks per atom. Shape: (N_atoms, num_types, max_ox_states).
            - radii_for_all_ox_states (torch.Tensor): Radius tensors per atom. Shape: (N_atoms, num_types, max_ox_states).
            - ox_states_used_mask (torch.Tensor): Binary mask indicating used oxidation states. Shape: (N_atoms, max_ox_states).
            - site_ids (torch.Tensor): Categorical site identifiers per atom (e.g., A/B/O site in perovskites). Shape: (N_atoms,).
    """

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
    else:
        raise NotImplementedError(f"crystal_system {crystal_system} is not implemented")

    (
        batch_list,
        size_list,
        angles_list,
        coords_list,
        abc_list,
        atom_dist_list,
        fnames,
    ) = [], [], [], [], [], [], []
    (
        atom_mask_for_all_ox_states_list,
        radii_for_all_ox_states_list,
        ox_states_used_mask_list,
        site_ids_list,
    ) = [], [], [], []
    num_total_crystals = 0
    file_paths = glob.glob(os.path.join(dir_path, "*.vasp"))
    for cry_i, path in enumerate(file_paths):
        atoms = Atoms.from_poscar(path)
        batch_list.append(
            torch.ones(len(atoms.elements), dtype=torch.long) * num_total_crystals
        )
        size_list.append(len(atoms.elements))
        angles_list.append(atoms.lattice.angles)
        coords_list.append(torch.tensor(atoms.coords))
        abc_list.append(atoms.lattice.abc)
        atom_dist_list.append(torch.ones((len(atoms.elements), 98)) / 98)
        fnames.append(os.path.basename(path))
        (
            _atom_mask_for_all_ox_states,
            _radii_for_all_ox_states,
            _ox_states_used_mask,
            site_ids,
        ) = get_atom_masks_for_oxidation_states(
            atoms=atoms,
            mask_d=mask_d,
            radii_d=radii_d,
            max_ox_states=max_ox_states,
            initial_dataset=initial_dataset,
            crystal_system=crystal_system,
        )
        atom_mask_for_all_ox_states_list.append(_atom_mask_for_all_ox_states)
        radii_for_all_ox_states_list.append(_radii_for_all_ox_states)
        ox_states_used_mask_list.append(
            torch.stack([_ox_states_used_mask] * len(atoms.elements), dim=0)
        )
        site_ids_list.append(site_ids)
        num_total_crystals += 1

    batch = torch.concat(batch_list, dim=0).to(torch.long)
    size = torch.tensor(size_list).to(torch.long)
    angles = torch.tensor(angles_list).to(torch.float32)
    coords = torch.concat(coords_list, dim=0).to(torch.float32)
    abc = torch.tensor(abc_list).to(torch.float32)
    atom_dist = torch.concat(atom_dist_list, dim=0)  # 一様分布で初期化
    atom_mask_for_all_ox_states = torch.concat(atom_mask_for_all_ox_states_list).type(
        torch.get_default_dtype()
    )
    radii_for_all_ox_states = torch.concat(radii_for_all_ox_states_list).type(
        torch.get_default_dtype()
    )
    ox_states_used_mask = torch.concat(ox_states_used_mask_list, dim=0).type(
        torch.get_default_dtype()
    )
    site_ids = torch.concat(site_ids_list, dim=0)

    if num_candidate is not None:
        size = size[:num_candidate]
        angles = angles[:num_candidate]
        coords = coords[: torch.sum(size)]
        abc = abc[:num_candidate]
        fnames = fnames[:num_candidate]
        atom_dist = atom_dist[: torch.sum(size)]
        batch = batch[: torch.sum(size)]
        atom_mask_for_all_ox_states = atom_mask_for_all_ox_states[: torch.sum(size)]
        radii_for_all_ox_states = radii_for_all_ox_states[: torch.sum(size)]
        ox_states_used_mask = ox_states_used_mask[: torch.sum(size)]
        site_ids = site_ids[: torch.sum(size)]
    assert torch.sum(size) == coords.shape[0] == atom_dist.shape[0] == len(batch)
    assert len(size) == len(angles) == len(abc)

    return (
        abc,
        atom_dist,
        batch,
        coords,
        size,
        angles,
        fnames,
        atom_mask_for_all_ox_states,
        radii_for_all_ox_states,
        ox_states_used_mask,
        site_ids,
    )


def load_crystalformer_initial_data(
    settings_dict: Dict[str, Any],
    dir_path: str,
    intermid_dir_path: str,
    crystal_system: Optional[str],
    initial_dataset: str,
    batch_size: int,
    use_atomic_mask: bool,
    max_ox_states: int,
    mask_method: str,
    angle_range: Tuple[float, float],
    abc_range: Tuple[float, float],
    test_mode: bool,
    num_candidate: Optional[int] = None,
    device: str = "cuda",
) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]:
    """
    Load and prepare initial structure inputs for Crystalformer training.

    This function loads per-crystal atomic and structural data from files,
    creates batched dictionaries including coordinates, masks, and labels,
    and returns batches with corresponding abc/angle scalers for optimization.

    Args:
        settings_dict (Dict): General configuration dictionary (may be unused here).
        dir_path (str): Path to directory containing structure files.
        intermid_dir_path (str): Path to oxidation mask and radii files.
        crystal_system (Optional[str]): Crystal system name (e.g., 'perovskite').
        batch_size (int): Number of crystals per batch.
        use_atomic_mask (bool): Whether to apply oxidation-state-based atomic masks.
        max_ox_states (int): Maximum number of oxidation states per element.
        mask_method (str): Key used for file naming of masks/radii.
        angle_range (Tuple[float, float]): Normalization range for angles.
        abc_range (Tuple[float, float]): Normalization range for lattice constants.
        test_mode (bool): If True, additional consistency checks are enabled.
        num_candidate (Optional[int]): Total number of candidate structures to use.
        device (str): Device to which tensors should be moved ('cuda' or 'cpu').

    Returns:
        List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]: A list of batches,
        each containing a Crystalformer input dictionary and corresponding scaler dictionary.
    """

    out_data = create_crystalformer_init_structure_from_directory(
        dir_path=dir_path,
        intermid_dir_path=intermid_dir_path,
        max_ox_states=max_ox_states,
        crystal_system=crystal_system,
        initial_dataset=initial_dataset,
        num_candidate=num_candidate,
        mask_method=mask_method,
    )

    (
        all_opt_abc,
        all_opt_x,
        all_batch,
        all_coords4input,
        all_size,
        all_angles,
        fnames,
        atom_mask_for_all_ox_states,
        radii_for_all_ox_states,
        ox_states_used_mask,
        site_ids,
    ) = out_data
    crystalformer_all_inputs_dict = {
        "x": all_opt_x,
        "batch": all_batch,
        "coords4input": all_coords4input,
        "size": all_size,
        "abc": all_opt_abc,
        "angles": all_angles,
        "init_coords": copy.deepcopy(all_coords4input),
        "init_abc": copy.deepcopy(all_opt_abc),
        "init_angles": copy.deepcopy(all_angles),
        "atomic_mask": None,
        "radii_tensor": None,
        "atom_mask_for_all_ox_states": atom_mask_for_all_ox_states,
        "radii_for_all_ox_states": radii_for_all_ox_states,
        "ox_states_used_mask": ox_states_used_mask,
        "site_ids": site_ids,
        "fnames": fnames,
    }

    if use_atomic_mask:
        crystalformer_all_inputs_dict["atomic_mask"] = atom_mask_for_all_ox_states.to(
            device
        )
        ox_mask_learnable_tensor_per_crystal = create_learnable_oxides_mask(
            crystalformer_all_inputs_dict
        )
        crystalformer_all_inputs_dict["ox_mask_learnable_tensor_per_crystal"] = (
            ox_mask_learnable_tensor_per_crystal
        )
        crystalformer_all_inputs_dict["radii_tensor"] = radii_for_all_ox_states.to(
            device
        )

    # divide the initial data into small batches
    minibatch_datas = []
    for min_id in range(0, num_candidate, batch_size):
        # select batch candidate crystals
        mini_batch_inputs_dict = select_batch_candidates_crystalformer(
            min_id=min_id,
            num_batch_crystal=batch_size,
            crystalformer_all_inputs_dict=crystalformer_all_inputs_dict,
        )
        scalers = {
            "abc_scaler": ABC_Scaler(
                init_batch_abc=mini_batch_inputs_dict["batch_abc"],
                min_length=abc_range[0],
                max_length=abc_range[1],
                device="cuda",
            ),
            "angle_scaler": Angle_Scaler(
                min_angle=angle_range[0],
                max_angle=angle_range[1],
            ),
        }
        minibatch_datas.append([mini_batch_inputs_dict, scalers])

    return minibatch_datas
