import itertools
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from jarvis.core.atoms import Atoms
from pymatgen.core.periodic_table import Element as Element_pmg
from smact import Element as Element_smact


def elec_neutral_all_common(
    elements: List[str], stoichs: List[List[int]], max_ox_states: int
) -> List[Tuple[int]]:
    """
    Determine all possible combinations of oxidation states for a list of elements that result in electrical neutrality.

    Args:
        elements (List[str]): A list of element symbols.
        stoichs (List[List[int]]): A list of stoichiometries corresponding to each element. Each sublist should contain exactly one integer.
        max_ox_states (int): The maximum number of oxidation state combinations to return.

    Returns:
        List[Tuple[int]]: A list of tuples, where each tuple contains a combination of oxidation states that sum to zero.

     Examples:
        >>> elec_neutral_all_common(['Ti', 'O'], [[1], [2]])
        [(4, -2, -2)]
        >>> elec_neutral_all_common(['Fe', 'O'], [[1], [1]])
        [(2, -2)]
        >>> elec_neutral_all_common(['H', 'O'], [[1], [2]])
        []
    """
    all_elements = []
    for elem, stoi in zip(elements, stoichs):
        assert len(stoi) == 1
        all_elements.extend([elem] * stoi[0])
    ox_combos = [
        list(
            set(Element_pmg(elem).icsd_oxidation_states)
            & set(Element_pmg(elem).oxidation_states)
            & set(Element_smact(elem).oxidation_states)
        )
        for elem in all_elements
    ]

    all_ox_states = []
    for ox_states in itertools.product(*ox_combos):
        if sum(ox_states) == 0:
            all_ox_states.append(ox_states)
        if len(all_ox_states) == max_ox_states:
            return all_ox_states, all_elements
    return all_ox_states, all_elements


def elec_neutral_all_common_loose(
    elements: List[str], stoichs: List[List[int]], max_ox_states: int
) -> List[Tuple[int]]:
    """
    Determine all possible combinations of oxidation states for a list of elements that result in electrical neutrality.

    Args:
        elements (List[str]): A list of element symbols.
        stoichs (List[List[int]]): A list of stoichiometries corresponding to each element. Each sublist should contain exactly one integer.
        max_ox_states (int): The maximum number of oxidation state combinations to return.

    Returns:
        List[Tuple[int]]: A list of tuples, where each tuple contains a combination of oxidation states that sum to zero.

     Examples:
        >>> elec_neutral_all_common(['Ti', 'O'], [[1], [2]])
        [(4, -2, -2)]
        >>> elec_neutral_all_common(['Fe', 'O'], [[1], [1]])
        [(2, -2)]
        >>> elec_neutral_all_common(['H', 'O'], [[1], [2]])
        []
    """
    all_elements = []
    for elem, stoi in zip(elements, stoichs):
        assert len(stoi) == 1
        all_elements.extend([elem] * stoi[0])
    ox_combos = [
        # list(set(Element_pmg(elem).icsd_oxidation_states) & set(Element_pmg(elem).oxidation_states) & set(Element(elem).oxidation_states))
        list(set(Element_smact(elem).oxidation_states))
        for elem in all_elements
    ]

    all_ox_states = []
    for ox_states in itertools.product(*ox_combos):
        if sum(ox_states) == 0:
            all_ox_states.append(ox_states)
        if len(all_ox_states) == max_ox_states:
            return all_ox_states, all_elements
    return all_ox_states, all_elements


def get_atom_masks_for_oxidation_states(
    atoms: Atoms,
    mask_d: Dict[str, np.ndarray],
    max_ox_states: int,
    crystal_system: Optional[str],
    initial_dataset: str,
    graph_update_mode: bool = False,
    radii_d: Optional[Dict[str, np.ndarray]] = None,
    num_type_of_atoms: int = 98,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate atom masks for all possible oxidation states combinations ensuring electrical neutrality, specific to the crystal system provided.

    Args:
        atoms (jarvis.core.atoms.Atoms): An object containing element information.
        mask_d (Dict[str, np.ndarray]): Dictionary mapping oxidation states (str) to masks (np.ndarray).
        max_ox_states (int): Maximum number of oxidation states to consider.
        crystal_system (Optional[str]): Type of crystal system, e.g., 'perovskite' or None.
        radii_d (Optional[Dict[str, np.ndarray]]): Dictionary of radii associated with elements, required for certain crystal systems.
        num_type_of_atoms (int): Total number of unique atom types considered.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - torch.Tensor: A 3D tensor where each slice along the third axis represents the atom masks for a particular combination of oxidation states. Shape is [num_atoms, num_type_of_atoms, max_ox_states].
            - torch.Tensor: A 3D tensor corresponding to radii for each element based on oxidation states. Shape is [num_atoms, num_type_of_atoms, max_ox_states].
            - torch.Tensor: A 1D tensor indicating which oxidation states were used (1) or not used (0) across combinations. Shape is [max_ox_states].
    """
    # choose elec_neutral_all_common function based on initial_dataset
    if initial_dataset == "megnet":
        elec_neutral_all_common_func = elec_neutral_all_common
    elif initial_dataset == "jarvis_supercon":
        elec_neutral_all_common_func = elec_neutral_all_common_loose
    else:
        raise NotImplementedError(
            f"initial_dataset, {initial_dataset}, is not implemented yet."
        )

    # Get neutral oxidation states
    if graph_update_mode:
        # graph updadeのときは、原子マスクを引き継ぐので、Noneにする
        assert crystal_system is None, (
            "crystal_system must be None for graph update mode."
        )
        site_ids = [np.nan] * len(atoms.elements)
        neutral_ox_states = []

    elif crystal_system == "perovskite":
        # Ti(4+), Ba(2+), O(2-)
        # print("perovskite_test",(elem_list, stoichs))
        p1_list, p2_list = [], []
        site_ids = []
        for elem in atoms.elements:
            if elem == "Ti":
                p1_list.append(2)
                p2_list.append(4)
                site_ids.append(1)  # B-site
            elif elem == "Ba":
                p1_list.append(1)
                p2_list.append(2)
                site_ids.append(0)  # A-site
            elif elem == "O":
                p1_list.append(-1)
                p2_list.append(-2)
                site_ids.append(2)  # O-site
            else:
                raise ValueError("Invalid element for perovskite crystal mode.")
        neutral_ox_states = [tuple(p1_list), tuple(p2_list)]
        assert radii_d is not None, (
            "Radii mask dictionary is required for perovskite crystal mode."
        )
    elif crystal_system is None:
        elem_list = list(set(atoms.elements))
        stoichs = [[atoms.elements.count(elem)] for elem in elem_list]
        neutral_ox_states, all_elements = elec_neutral_all_common_func(
            elem_list, stoichs, max_ox_states=max_ox_states
        )
        assert len(neutral_ox_states) > 0, (
            "No neutral oxidation states found."
        )  # あとで消す
        site_ids = [np.nan] * len(atoms.elements)
    else:
        raise ValueError("Invalid crystal_system. Choose from ['perovskite', None].")

    # Add dummy tuples until the length is max_ox_states
    dummy_tuple = tuple([-999] * len(atoms.elements))
    while len(neutral_ox_states) < max_ox_states:
        neutral_ox_states.append(dummy_tuple)

    atom_mask_for_all_ox_states = []
    radii_tensor_for_all_ox_states = []
    ox_states_used_mask = []
    for ox_states in neutral_ox_states:
        # Initialize mask for each oxidation state
        ox_mask = np.zeros((len(atoms.elements), num_type_of_atoms))
        radii_tensor = np.zeros((len(atoms.elements), num_type_of_atoms))

        # Skip dummy oxidation states
        if ox_states == dummy_tuple:
            ox_states_used_mask.append(0)
            atom_mask_for_all_ox_states.append(ox_mask)
            radii_tensor_for_all_ox_states.append(radii_tensor)
            continue
        else:
            ox_states_used_mask.append(1)

        #  Apply masks for each element based on oxidation states
        for pos, ox_val in enumerate(ox_states):
            if str(ox_val) not in mask_d:
                raise ValueError(
                    f"Oxidation state {ox_val} not found in mask dictionary. Element: {all_elements[pos]}"
                )
            ox_mask[pos] = mask_d[str(ox_val)]
            if radii_d is not None:
                radii_tensor[pos] = radii_d[str(ox_val)]

        atom_mask_for_all_ox_states.append(ox_mask)
        radii_tensor_for_all_ox_states.append(radii_tensor)

    return (
        torch.tensor(np.stack(atom_mask_for_all_ox_states, axis=-1)),
        torch.tensor(np.stack(radii_tensor_for_all_ox_states, axis=-1)),
        torch.tensor(ox_states_used_mask),
        torch.tensor(site_ids),
    )
