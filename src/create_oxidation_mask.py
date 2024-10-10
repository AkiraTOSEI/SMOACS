import itertools
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from jarvis.core.atoms import Atoms
from pymatgen.core.periodic_table import Element as Element_pmg
from smact import Element

from .utils import create_element_list


def elec_neutral_all_common(elements: List[str], stoichs: List[List[int]],max_ox_states:int) -> Tuple[List[Tuple[int]], List[str]]:
    """
    Determine all possible combinations of oxidation states for a list of elements that result in electrical neutrality.

    Args:
        elements (List[str]): A list of element symbols.
        stoichs (List[List[int]]): A list of stoichiometries corresponding to each element. Each sublist should contain exactly one integer.
        max_ox_states (int): The maximum number of oxidation state combinations to return.

    Returns:
        List[Tuple[int]]: A list of tuples, where each tuple contains a combination of oxidation states that sum to zero.

     Examples:
        >>> elec_neutral_all_common(['Ti', 'O'], [[1], [2]], 10)
        [(4, -2, -2)], ['Ti', 'O', 'O']
        >>> elec_neutral_all_common(['N', 'Ru'], [[1], [1]], 1)
        ([(-2, 2)], ['N', 'Ru'])
        >>> elec_neutral_all_common(['N', 'Ru'], [[1], [1]], 5)
        ([(-2, 2), (-3, 3)], ['N', 'Ru'])
        >>> elec_neutral_all_common(['H', 'O'], [[1], [2]])
        ([], ['H', 'O', 'O'])
    """
    all_elements = []
    for elem, stoi in zip(elements, stoichs):
        assert len(stoi) == 1
        all_elements.extend([elem]*stoi[0])
    ox_combos = [
        list(set(Element_pmg(elem).icsd_oxidation_states) & set(Element_pmg(elem).oxidation_states) & set(Element(elem).oxidation_states))
        for elem in all_elements    
    ]

    all_ox_states = []
    for ox_states in itertools.product(*ox_combos):
        if sum(ox_states) == 0:
            all_ox_states.append(ox_states)
        if len(all_ox_states) == max_ox_states:
            return all_ox_states, all_elements
    return all_ox_states, all_elements


def find_positions(lst: List[str], element: str) -> List[int]:
    """Find all positions of an element in a list."""
    return [index for index, value in enumerate(lst) if value == element]


def get_oxidation_states_patterns_and_site_ids(
    element_list: List[str],
    atom_settings_dict: Dict[str, Dict[str, Any]],
    max_ox_states: int,
    graph_update_mode: bool,
    dummy_number: int = -999,
)-> Tuple[List[Tuple[int]], List[int]]:
    """

    Args:
        element_list (List[str]): List of elements. e.g. ['Ti', 'O', 'O']
        site_atom_oxidation_dict (Dict[str, Dict[str, Any]]): Dictionary containing oxidation states patterns and site ids for each element.
        max_ox_states (int): Maximum number of oxidation state patterns to consider.
        crystal_system (Optional[str]): Type of crystal system, e.g., 'perovskite' or None.
        graph_update_mode (bool): Whether to update the graph or not.
    Output:
        Tuple[List[Tuple[int]], List[int]]: A tuple containing a list of oxidation states patterns and a list of site ids.
    """

    # Get neutral oxidation states

    if graph_update_mode:
        # When graph update is True, set to None to inherit the previous atom mask. So the oxidation states are not needed.
        neutral_ox_states = [([dummy_number] * len(element_list))]
        site_ids = [np.nan] * len(element_list)

    elif '*' in atom_settings_dict.keys():        # When graph update is True, set to None to inherit the atom mask. 
        # When using wildcard '*', set to None to create the atom mask later using 'element'.
        if atom_settings_dict.get('*') is not None:
            assert atom_settings_dict['*'].get('element') is not None, " 'element' must be provided for wildcard."

        site_ids = [dummy_number*2] * len(element_list)
        neutral_ox_states = [([dummy_number*2] * len(element_list))]

    
    elif len(atom_settings_dict)==0:
        # Get unique elements and their stoichiometries and then find all possible oxidation states combinations

        elem_list = list(set(element_list))
        stoichs = [[element_list.count(elem)] for elem in elem_list]
        neutral_ox_states, _ = elec_neutral_all_common(elem_list, stoichs, max_ox_states=max_ox_states)
        site_ids = [np.nan] * len(element_list)
        # Add dummy tuples until the length is max_ox_states
        dummy_tuple = tuple([dummy_number] * len(element_list))
        while len(neutral_ox_states) < max_ox_states:
            neutral_ox_states.append(dummy_tuple)

    elif len(atom_settings_dict)>0:

        # all elements have the same length
        num_ox_patterns = len(next(iter(atom_settings_dict.values()))['ox_patterns'])
        assert all(len(v['ox_patterns']) == num_ox_patterns for v in atom_settings_dict.values()), "All # of ox_patterns for elements must have the same."

        # initialize the oxidation state list with NaN
        ox_list_array = np.ones((num_ox_patterns, len(element_list)))*np.nan

        # fill the oxidation state list with the values from the dictionary
        for e_i, elem in enumerate(element_list):
            ox_list_array[:,e_i] = atom_settings_dict[elem]['ox_patterns']
        assert np.isnan(ox_list_array).sum() == 0, "All oxidation states must be filled."

        # neutral check
        assert (np.sum(ox_list_array,axis=1)==0).all()

        # convert the array to a list of tuples
        neutral_ox_states = [tuple(ox_list_array[i].astype(int)) for i in range(ox_list_array.shape[0])]

        # site ids
        site_ids = [atom_settings_dict[elem]['site_id'] for elem in element_list]
                    
    else:
        raise Exception("Invalid input for site_atom_oxidation_dict")
    
    return neutral_ox_states, site_ids   



def get_site_wise_atom_specification_dict(
        atom_settings_dict:dict,
        dummy_number:int,
        ) -> Dict[int, np.ndarray]:
    """
    Get site-wise atom specification dictionary.
    """
    site_wise_atom_specification_dict = {}
    all_element_list = create_element_list()

    if '*' in atom_settings_dict.keys():
        #
        # e.g. atom_settings_dict: {'*': {'element': 'He'}} -> site_wise_atom_specification_dict: {0: np.array([0., 1., 0., ...])},...}
        #
        specified_elem = atom_settings_dict['*']['element']
        site_wise_atom_specification = np.zeros(len(all_element_list)) # initialize
        idx = all_element_list.index(specified_elem)
        site_wise_atom_specification[idx] = 1
        site_wise_atom_specification_dict[dummy_number*2] = site_wise_atom_specification
        return site_wise_atom_specification_dict

    for elem in atom_settings_dict.keys():
        #
        # e.g. atom_settings_dict: {'Ti': {'element': ['Li'], 'site_id': 1}} -> site_wise_atom_specification_dict: {1: np.array([0., 0., 1., 0., ...])}
        #
        if 'element' in atom_settings_dict[elem]:
            site_wise_atom_specification = np.zeros(len(all_element_list)) # initialize
            #print(atom_settings_dict[elem]['element'])
            site_id = atom_settings_dict[elem]['site_id']
            for specified_elem in atom_settings_dict[elem]['element']:
                idx = all_element_list.index(specified_elem)
                site_wise_atom_specification[idx] = 1
            site_wise_atom_specification_dict[site_id] = site_wise_atom_specification
    
    return site_wise_atom_specification_dict



def get_radii_and_atom_mask(
        element_list: List[str],
        site_ids: List[int],
        neutral_ox_states: List[Tuple[int]],
        mask_d: Dict[str, np.ndarray],
        radii_d: Optional[Dict[str, np.ndarray]],
        site_wise_atom_specification_dict: Dict[int, np.ndarray],
        num_type_of_atoms: int,
        use_atomic_radii: bool,
        dummy_number: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate atom masks and corresponding radii tensor. If site-specific atom specification is provided in site_wise_atom_specification_dict, use it. Otherwise, create masks based on oxidation states.
    Args:
        element_list (List[str]): List of elements. e.g. ['Ti', 'Ba', 'O', 'O', 'O']
        site_ids (List[int]): List of site ids. e.g. [0, 1, 2, 2, 2]
        neutral_ox_states (List[Tuple[int]]): List of oxidation states patterns. e.g. [(4, 2, -2, -2, -2)]
        mask_d (Dict[str, np.ndarray]): Dictionary mapping oxidation states (str) to masks (np.ndarray).
        radii_d (Dict[str, np.ndarray]): Dictionary of radii associated with elements.
        site_wise_atom_specification_dict (Dict[int, np.ndarray]): Dictionary of site-wise atom specifications.
        dummy_number (int): Dummy number to skip certain oxidation states.
        num_type_of_atoms (int): Total number of unique atom types considered.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        1st: A 3D tensor where each slice along the third axis represen ts the atom masks for a particular combination of oxidation states. Shape is [num_atoms_in_crystal, num_type_of_atoms, max_ox_states].
        2nd: A 3D tensor corresponding to radii for each element based on oxidation states. Shape is [num_atoms_in_crystal, num_type_of_atoms, max_ox_states].
        3rd: A 1D tensor indicating which oxidation states were used (1) or not used (0) across combinations. Shape is [max_ox_states].

    """

    atom_mask_for_all_ox_states = []
    radii_tensor_for_all_ox_states = []
    ox_states_used_mask = []

    for ox_states in neutral_ox_states:
        # Initialize mask for each oxidation state
        ox_mask = np.zeros((len(element_list), num_type_of_atoms))
        radii_tensor = np.zeros((len(element_list), num_type_of_atoms))

        # Skip dummy oxidation states
        if dummy_number in ox_states:
            ox_states_used_mask.append(0)
            atom_mask_for_all_ox_states.append(ox_mask)
            radii_tensor_for_all_ox_states.append(radii_tensor)
            continue
        else:
            ox_states_used_mask.append(1)

        #  Apply masks for each element based on oxidation states
        for pos, (ox_val, s_id) in enumerate(zip(ox_states,site_ids)):

            # サイトごとの原子種の指定がある場合はそれを使う。ない場合は、酸化数をもとにマスクを作成する. 
            # If there is a specification of atomic species for each site, use it. Otherwise, create a mask based on the oxidation number.
            
            if s_id in site_wise_atom_specification_dict.keys():
                ox_mask[pos] = site_wise_atom_specification_dict[s_id]
            else:
                if str(ox_val) not in mask_d:
                    raise ValueError(f"Oxidation state {ox_val} not found in mask dictionary.")
                ox_mask[pos] = mask_d[str(ox_val)]
            if use_atomic_radii:
                radii_tensor[pos] = radii_d[str(ox_val)]

        atom_mask_for_all_ox_states.append(ox_mask)
        radii_tensor_for_all_ox_states.append(radii_tensor)

    return torch.tensor(np.stack(atom_mask_for_all_ox_states,axis=-1)), torch.tensor(np.stack(radii_tensor_for_all_ox_states,axis=-1)), torch.tensor(ox_states_used_mask)


def get_atom_masks_for_oxidation_states(
        atoms: Atoms, 
        mask_d: Dict[str, np.ndarray],
        max_ox_states:int ,
        site_atom_oxidation_dict: Optional[Dict[str, Dict[str, Any]]],
        graph_update_mode: bool,
        radii_d: Optional[Dict[str, np.ndarray]],
        num_type_of_atoms: int = 98,
        dummy_number: int = -999,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate atom masks for all possible oxidation states combinations ensuring electrical neutrality, specific to the crystal system provided.
    Args
        atoms (jarvis.core.atoms.Atoms): An object containing element information.
        mask_d (Dict[str, np.ndarray]): Dictionary mapping oxidation states (str) to masks (np.ndarray).
        max_ox_states (int): Maximum number of oxidation states to consider.
        site_atom_oxidation_dict (Dict[str, Dict[str, Any]]): Dictionary containing oxidation states patterns and site ids for each element.
        graph_update_mode (bool): Whether to update the graph or not.
        radii_d (Optional[Dict[str, np.ndarray]]): Dictionary of ionic radii associated with elements, required for certain crystal systems.
        num_type_of_atoms (int): Total number of unique atom types considered.
        dummy_number (int): Dummy number to skip certain oxidation states.
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            - torch.Tensor: A 3D tensor where each slice along the third axis represents the atom masks for a particular combination of oxidation states. Shape is [num_atoms, num_type_of_atoms, max_ox_states].
            - torch.Tensor: A 3D tensor corresponding to radii for each element based on oxidation states. Shape is [num_atoms, num_type_of_atoms, max_ox_states].
            - torch.Tensor: A 1D tensor indicating which oxidation states were used (1) or not used (0) across combinations. Shape is [max_ox_states].
            - torch.Tensor: A 1D tensor indicating site ids for each atom. Shape is [num_atoms].
    """

    # Get element list from Crystal structures
    element_list = atoms.elements

    # Get oxidation states patterns and site ids
    neutral_ox_states, site_ids  = get_oxidation_states_patterns_and_site_ids(
        element_list=element_list, 
        atom_settings_dict=site_atom_oxidation_dict['Atom_settings'], 
        max_ox_states=max_ox_states, 
        graph_update_mode=graph_update_mode,
        dummy_number = dummy_number
    )
        
    # Get site-wise atom specification dictionary
    site_wise_atom_specification_dict = get_site_wise_atom_specification_dict(site_atom_oxidation_dict['Atom_settings'], dummy_number)

    # Get atom masks and ion radii for all oxidation states
    atom_mask_for_all_ox_states, radii_tensor_for_all_ox_states, ox_states_used_mask = get_radii_and_atom_mask(
        element_list=element_list,
        site_ids=site_ids,
        neutral_ox_states=neutral_ox_states,
        mask_d=mask_d,
        radii_d=radii_d,
        site_wise_atom_specification_dict=site_wise_atom_specification_dict,
        use_atomic_radii=site_atom_oxidation_dict['use_ionic_radii'],
        dummy_number=dummy_number,
        num_type_of_atoms=num_type_of_atoms
    )

    return atom_mask_for_all_ox_states, radii_tensor_for_all_ox_states, ox_states_used_mask, torch.tensor(site_ids)


def stacking_learnable_oxsides_mask(
    ox_mask_learnable_tensor_per_crystal:torch.Tensor,
    size:torch.Tensor
    )->torch.Tensor:
    """
    Args:
        ox_mask_learnable_tensor_per_crystal: torch.Tensor, shape=(num_crystals, num_oxides)
    """
    return torch.concat([torch.stack([ox_mask_learnable_tensor_per_crystal[idx]]* size) for idx, size in enumerate(size)],dim=0)


def create_learnable_oxides_mask(mini_batch_inputs_dict):
    """
    a function to create learnable mask tensor for oxides used in the model
    """
    _ox_mask_learnable_tensor_per_crystal  = []
    total_idx = 0

    for idx in mini_batch_inputs_dict['size']:
        total_idx += idx
        _ox_mask_learnable_tensor_per_crystal.append(mini_batch_inputs_dict['ox_states_used_mask'][total_idx-1])
    ox_mask_learnable_tensor_per_crystal = torch.stack(_ox_mask_learnable_tensor_per_crystal)

    return ox_mask_learnable_tensor_per_crystal
        