import itertools
import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from jarvis.core.atoms import Atoms
from pymatgen.core import Element as Element_pmg
from smact import Element
from tqdm import tqdm
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
try:
    from pymatgen.symmetry.analyzer import SymmetryUndeterminedError
except ImportError:  # pragma: no cover - fallback for older/newer pymatgen
    class SymmetryUndeterminedError(Exception):
        pass
from src.spg_wp.wyckoff_expander import WyckoffExpander

from src.utils.common import create_element_list, get_master_dir
from src.utils.coord import compute_abc_angle


def count_elements(elements_list):
    elements = {}
    for element in elements_list:
        if element not in elements:
            elements[element] = 0
        elements[element] += 1
    return elements


def _load_level_forbidden_sets() -> dict[int, set[str]]:
    md_path = (
        Path(get_master_dir())
        / "workspace"
        / "mosso_ws"
        / "rare_critical_free_levels_with_Z.md"
    )
    if not md_path.is_file():
        return {}
    text = md_path.read_text(encoding="utf-8")
    level_sets: dict[int, set[str]] = {}
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        parts = [p.strip() for p in line.strip().strip("|").split("|")]
        if len(parts) < 3:
            continue
        if not parts[0].isdigit():
            continue
        level = int(parts[0])
        if level < 1 or level > 7:
            continue
        list_col = parts[2]
        elements = set(re.findall(r"[A-Z][a-z]?", list_col))
        if elements:
            level_sets[level] = elements
    return level_sets


def _calculate_elem_level(elements_list: list[str], level_sets: dict[int, set[str]]) -> int:
    elems = set(elements_list)
    for level in range(7, 0, -1):
        forbidden = level_sets.get(level)
        if not forbidden:
            continue
        if elems.isdisjoint(forbidden):
            return level
    return 0


def elec_neutral_check_SUPER_COMMON(
    num_i: int,
    total: int,
    elements: List[str],
    stoichs: List[List[int]],
    return_all_ox_states: bool = False,
) -> Tuple[bool, Optional[Tuple[int]]]:
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
        all_elements.extend([elem] * stoi[0])
    ox_combos = [
        list(
            set(Element_pmg(elem).icsd_oxidation_states)
            & set(Element_pmg(elem).oxidation_states)
            & set(Element(elem).oxidation_states)
            & set(Element_pmg(elem).common_oxidation_states)
        )
        for elem in all_elements
    ]

    # check excluding non-oxidation state elements
    if any([len(ox) == 0 for ox in ox_combos]):
        return False, all_elements, None

    lengths = np.array([len(sublist) for sublist in ox_combos])
    product_of_lengths = np.prod(lengths)

    if return_all_ox_states:
        all_neutral_ox_states = []
        for ox_states in tqdm(
            itertools.product(*ox_combos),
            total=product_of_lengths,
            leave=False,
            desc=f"neutral check ({num_i + 1}/{total}) by PMG",
        ):
            if sum(ox_states) == 0:
                all_neutral_ox_states.append(ox_states)
        return len(all_neutral_ox_states) > 0, all_elements, all_neutral_ox_states

    else:
        for ox_states in tqdm(
            itertools.product(*ox_combos),
            total=product_of_lengths,
            leave=False,
            desc=f"neutral check ({num_i + 1}/{total}) by PMG",
        ):
            if sum(ox_states) == 0:
                return True, all_elements, ox_states

        return False, all_elements, None


def check_nearest_neighbor(atoms_data: Atoms, r: float = 0.5, verbose: int = 0) -> bool:
    """
    Checks for any atom pairs within a specified distance in a supercell.

    Args:
        atoms_data (Atoms): An instance of Atoms containing atomic data.
        r (float): Radius to check for neighbor proximity. Default is 0.5.
        verbose (int): Verbosity level. A non-zero value prints detailed bond information.

    Returns:
        bool: True if no atom pairs are found within the specified distance, False otherwise.
    """
    try:
        # Generate supercell
        sc_atoms_data = atoms_data.make_supercell_matrix([2, 2, 2])
        coords = (
            sc_atoms_data.coords
        )  # Assuming this now correctly handles supercell coordinates

        # Compute distance matrix efficiently
        from scipy.spatial import distance_matrix

        dist_matrix = distance_matrix(coords, coords)
        np.fill_diagonal(dist_matrix, np.inf)

        # get original atom ids
        atom_ids = np.array(
            [[atoms_data.elements.index(e)] * (2 * 2 * 2) for e in atoms_data.elements]
        ).reshape(-1)

        # Identify pairs with distance less than r
        close_pairs = np.where(dist_matrix < r)

        if verbose > 0:
            for _row, _col in zip(*close_pairs):
                coord0 = coords[_row]
                coord1 = coords[_col]
                print(
                    f"bond between: Atom-{atom_ids[_row]} {sc_atoms_data.elements[_row]}({coord0[0]:.3f},{coord0[1]:.3f},{coord0[2]:.3f}) ----- Atom-{atom_ids[_col]} {sc_atoms_data.elements[_col]}({coord1[0]:.3f},{coord1[1]:.3f},{coord1[2]:.3f}), distance:{dist_matrix[_row, _col]:.3f}"
                )
                original_coord0 = atoms_data.coords[atom_ids[_row]]
                original_coord1 = atoms_data.coords[atom_ids[_col]]
                print(
                    f"original coord:  Atom-{atom_ids[_row]} {atoms_data.elements[atom_ids[_row]]}({original_coord0[0]:.3f},{original_coord0[1]:.3f},{original_coord0[2]:.3f}) ----- Atom-{atom_ids[_col]} {atoms_data.elements[atom_ids[_col]]}({original_coord1[0]:.3f},{original_coord1[1]:.3f},{original_coord1[2]:.3f})"
                )
                print("")

        return not any(dist_matrix.flatten() < r)

    except Exception as e:
        print(f"An error occurred: {e}")
        return False


def perovskite_coordinate_check(
    init_coords: np.ndarray,
    dir_coords: np.ndarray,
    size: np.ndarray,
    limit_of_displacement: float = 0.15,
) -> np.ndarray:
    """
    Check whether atomic displacements in perovskite optimization stay within a valid range
    under periodic boundary conditions (PBC). This is especially important to detect
    anomalously displaced atoms in large unit cells.

    Args:
        init_coords (np.ndarray): Initial fractional coordinates. Shape: [num_atoms, 3].
        dir_coords (np.ndarray): Optimized fractional coordinates. Shape: [num_atoms, 3].
        size (np.ndarray): Number of atoms per crystal. Shape: [num_structures].
        limit_of_displacement (float, optional): Max allowed displacement per axis. Default: 0.15.

    Returns:
        np.ndarray: Boolean array of shape (num_structures,) indicating whether all atoms
                    in each structure are within the allowed displacement range.
    """
    displacement = np.minimum(
        np.abs(init_coords - dir_coords),
        np.abs(init_coords - dir_coords + 1),
        np.abs(init_coords - dir_coords - 1),
    )
    assert (size[0] == size).all()
    return (
        (displacement <= limit_of_displacement)
        .all(axis=1)
        .reshape(len(size), size[0])
        .all(axis=1)
    )


def check_neurality_bondlength_and_save_structure(
    npz_path:str,
    csv_path:str,
    saved_dir:str,
    crystal_system:str,
    perovskite_size:str,
    acceptable_margin:float,
    wyckoff_expander:Optional[WyckoffExpander],
    neutral_check:bool = True,
    poscar_magmom_min: Optional[float] = None,
) -> pd.DataFrame:
    """
    Check charge neutrality and bondlength validity for optimized crystal structures, then save the valid ones.

    This function:
      - Loads structure data from `.npz` and `.csv`.
      - Checks charge neutrality using SMACT or a common oxidation state rule.
      - Validates atomic bondlengths to detect unrealistic structures.
      - Saves structures in POSCAR format (and optionally CIF).
      - Returns a DataFrame summarizing neutrality and bond checks.

    Args:
        npz_path (str): Path to `.npz` file with optimization results.
        csv_path (str): Path to `.csv` file with loss and ranking information.
        saved_dir (str): Directory to save valid POSCAR files.
        crystal_system (str): Name of crystal system (e.g., 'perovskite').
        perovskite_size (str): Size string (e.g., '2x2x2') to determine if neutrality check is skipped.
        acceptable_margin (float): Threshold for loss to consider a structure acceptable.
        neutral_check (bool, optional): Whether to check for charge neutrality. Defaults to True.

    Returns:
        pd.DataFrame: DataFrame containing neutrality and bondlength check results for all evaluated structures.
    """
    # clean the directory
    for file in os.listdir(saved_dir):
        os.remove(os.path.join(saved_dir, file))

    VALID_ELEMENTS98=create_element_list()
    elem_level_sets = _load_level_forbidden_sets()
    os.makedirs(f'{saved_dir}', exist_ok=True)

    d = np.load(npz_path, allow_pickle=True)

    # formation energyが低く、指定のバンドギャップを満たすものから順に並べる
    sorted_df = pd.read_csv(csv_path)
    if "loss_total" in sorted_df.columns:
        sorted_df = sorted_df.sort_values(["loss_total"])
    else:
        sorted_df = sorted_df.sort_values(['loss_onehot','ef_onehot'])

    sorted_index = sorted_df['lattice_index'].values
    lattice_vectors = np.array(d["lattice_vectors"])
    if lattice_vectors.ndim == 2:
        lattice_vectors = lattice_vectors[None, ...]
    elif lattice_vectors.ndim == 1 and lattice_vectors.size == 9:
        lattice_vectors = lattice_vectors.reshape(1, 3, 3)
    opt_abc, opt_angle = compute_abc_angle(torch.tensor(lattice_vectors))
    coordinate_lattice_id = np.concatenate([np.array([lattice_id]*num) for lattice_id, num in enumerate(d['num_atoms'])])

    os.makedirs(f'{saved_dir}', exist_ok=True)
    os.makedirs('./cif', exist_ok=True)

    neutralities = []
    pred_magmom = d.get("pred_magmom")
    for num_i, lattice_id in enumerate(tqdm(sorted_index, desc='save_structure', total=len(sorted_index))):

        # exclude nan data
        if np.isnan(d['dir_coords'][coordinate_lattice_id==lattice_id]).any() or np.isnan(lattice_vectors[lattice_id]).any():
            continue

        # Neutral check
        elements_list = [VALID_ELEMENTS98[atom_idx] for atom_idx in np.argmax(d['onehot_x'][coordinate_lattice_id==lattice_id], axis=1)]
        atoms, stoichs = [], []
        for _atom, _stoich in count_elements(elements_list).items():
            atoms.append(_atom)
            stoichs.append([_stoich])

        max_neutral_atoms = 10
        if neutral_check and len(elements_list) <= max_neutral_atoms:
            try:
                (
                    is_neutral_cmn,
                    all_elements_cmn,
                    ox_states_cmn,
                ) = elec_neutral_check_SUPER_COMMON(
                    num_i, len(sorted_index), atoms, stoichs, True
                )
                if crystal_system == 'perovskite':
                    sorted_ox_states = [tuple(sorted(tup)) for tup in ox_states_cmn]
                    is_neutral_cmn = (
                        (-1, -1, -1, 1, 2) in sorted_ox_states
                    ) or ((-2, -2, -2, 2, 4) in sorted_ox_states)
            except UnicodeDecodeError:
                is_neutral_cmn, all_elements_cmn, ox_states_cmn = (
                    np.nan,
                    [np.nan],
                    [np.nan],
                )
        else:
            is_neutral_cmn, all_elements_cmn, ox_states_cmn = np.nan, [np.nan], [np.nan]

        if (
            poscar_magmom_min is not None
            and pred_magmom is not None
            and float(pred_magmom[lattice_id]) < poscar_magmom_min
        ):
            continue

        elem_level = None
        if elem_level_sets:
            elem_level = _calculate_elem_level(elements_list, elem_level_sets)
        # create Atoms object
        atom_dic  = {
            'lattice_mat': [list(vec) for vec in lattice_vectors[lattice_id]],
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
        # space group check
        if wyckoff_expander is not None:
            try:
                proposed_spg = SpacegroupAnalyzer(
                    atoms_data.pymatgen_converter()
                ).get_space_group_number()
                spg_check = wyckoff_expander.spg_number == proposed_spg
            except (TypeError, SymmetryUndeterminedError):
                proposed_spg = -1
                spg_check = False

        # bondlength check 
        bondlength_check = check_nearest_neighbor(atoms_data)

        # add to data dictionary
        data_dict = {   
                'original_fnames':d['original_fnames'][lattice_id],
                'elements':atoms,
                'is_neutral_cmn':is_neutral_cmn,
                'elements_cmn':all_elements_cmn,
                'ox_states_cmn':ox_states_cmn,
                'minbond_less_than_0.5':bondlength_check,
                'one_atom':len(atoms)==1,
            }
        if elem_level is not None:
            data_dict['elem_level'] = elem_level
        if wyckoff_expander is not None:
            data_dict['spg_check'] = spg_check
            data_dict['proposed_spg'] = proposed_spg
        print("DEBUG: ", data_dict)

        neutralities.append(
            data_dict
        )        

        # structure saved
        fname = f"Optimized_based_on_{d['original_fnames'][lattice_id]}"
        atoms_data.write_poscar(os.path.join(saved_dir, f'{fname}'))


    return pd.DataFrame(neutralities)
