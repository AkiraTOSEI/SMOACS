import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from jarvis.core.atoms import Atoms
from pymatgen.core.periodic_table import Element as Element_pmg
from smact import Element as Element_smact

from scripts.data_generations.oxidation_sampler import sample_k_per_pos, build_ways_per_pos
from scripts.data_generations.neutral_ox_by_n import load_or_build_neutral_ox_dict


def elec_neutral_all_common(
    elements: List[str],
    stoichs: List[List[int]],
    max_ox_states: int,
    timeout_sec: Optional[float] = None,
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
    states_per_pos = [
        list(
            set(Element_pmg(elem).icsd_oxidation_states)
            & set(Element_pmg(elem).oxidation_states)
            & set(Element_smact(elem).oxidation_states)
        )
        for elem in all_elements
    ]
    start_time = time.perf_counter()
    if timeout_sec is not None and timeout_sec <= 0:
        return [], all_elements
    ways, offset = build_ways_per_pos(states_per_pos, target=0)
    total = ways[-1][offset]
    if total == 0:
        return [], all_elements
    if timeout_sec is not None and (time.perf_counter() - start_time) >= timeout_sec:
        return [], all_elements
    sampled = sample_k_per_pos(
        states_per_pos,
        k=min(max_ox_states, total),
        target=0,
        seed=None,
        unique=True,
    )
    # The sampler traverses positions backwards; return oxidation states in atom order.
    return [tuple(reversed(seq)) for seq in sampled], all_elements


def elec_neutral_all_common_loose(
    elements: List[str],
    stoichs: List[List[int]],
    max_ox_states: int,
    timeout_sec: Optional[float] = None,
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
    states_per_pos = [
        list(set(Element_smact(elem).oxidation_states))
        for elem in all_elements
    ]
    start_time = time.perf_counter()
    if timeout_sec is not None and timeout_sec <= 0:
        return [], all_elements
    ways, offset = build_ways_per_pos(states_per_pos, target=0)
    total = ways[-1][offset]
    if total == 0:
        return [], all_elements
    if timeout_sec is not None and (time.perf_counter() - start_time) >= timeout_sec:
        return [], all_elements
    sampled = sample_k_per_pos(
        states_per_pos,
        k=min(max_ox_states, total),
        target=0,
        seed=None,
        unique=True,
    )
    # The sampler traverses positions backwards; return oxidation states in atom order.
    return [tuple(reversed(seq)) for seq in sampled], all_elements


def get_atom_masks_for_oxidation_states(
        atoms: Atoms,
        mask_d: Dict[str, np.ndarray],
        max_ox_states:int ,
        crystal_system: Optional[str],
        graph_update_mode: bool =  False,
        radii_d: Optional[Dict[str, np.ndarray]] = None,
        num_type_of_atoms: int = 98,
        profile: Optional[Dict[str, float]] = None,
        timeout_sec: Optional[float] = None,
        mask_name: Optional[str] = None,
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
    # Get neutral oxidation states
    if profile is not None:
        profile.setdefault("ox_state_search_time", 0.0)
        profile.setdefault("ox_mask_loop_time", 0.0)
        profile.setdefault("ox_tensor_time", 0.0)

    if graph_update_mode:
        # graph updadeのときは、原子マスクを引き継ぐので、Noneにする
        assert crystal_system is None, "crystal_system must be None for graph update mode."
        site_ids = [np.nan] * len(atoms.elements)
        neutral_ox_states = []

    elif crystal_system == 'perovskite':
        # Ti(4+), Ba(2+), O(2-)
        #print("perovskite_test",(elem_list, stoichs))
        p1_list, p2_list = [],[]
        site_ids = []
        for elem in atoms.elements:
            if elem == 'Ti':
                p1_list.append(2)
                p2_list.append(4)
                site_ids.append(1) # B-site
            elif elem == 'Ba':
                p1_list.append(1)
                p2_list.append(2)
                site_ids.append(0) # A-site
            elif elem == 'O':
                p1_list.append(-1)
                p2_list.append(-2)
                site_ids.append(2) # O-site
            else:
                raise ValueError("Invalid element for perovskite crystal mode.")
        neutral_ox_states = [tuple(p1_list), tuple(p2_list)]
        assert radii_d is not None, "Radii mask dictionary is required for perovskite crystal mode."
    elif crystal_system is None:
        n_sites = len(atoms.elements)
        t0 = time.perf_counter()
        if mask_name is not None:
            states = sorted(int(k) for k in mask_d.keys())
            out_dir = Path(__file__).resolve().parents[2] / "data" / "intermidiate"
            payload = load_or_build_neutral_ox_dict(
                out_dir=out_dir,
                mask_name=mask_name,
                states=states,
                n_max=135,
                k_per_n=1000,
                target=0,
                seed=None,
                overwrite=True,
            )
            choices = payload.get("data", {}).get(str(n_sites), [])
            if not choices:
                raise TimeoutError("No neutral oxidation states found within timeout.")
            rng = np.random.default_rng()
            count = min(max_ox_states, len(choices))
            selected = rng.choice(len(choices), size=count, replace=False)
            neutral_ox_states = []
            for idx in selected:
                ox_seq = list(choices[idx])
                rng.shuffle(ox_seq)
                neutral_ox_states.append(tuple(ox_seq))
            all_elements = list(atoms.elements)
        else:
            # Keep oxidation-state assignment aligned to atom-site order.
            all_elements = list(atoms.elements)
            stoichs = [[1] for _ in all_elements]
            neutral_ox_states, all_elements = elec_neutral_all_common(
                all_elements,
                stoichs,
                max_ox_states=max_ox_states,
                timeout_sec=timeout_sec,
            )
            if len(neutral_ox_states) == 0:
                raise TimeoutError("No neutral oxidation states found within timeout.")
        if profile is not None:
            profile["ox_state_search_time"] += time.perf_counter() - t0
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
    t0 = time.perf_counter()
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
                raise ValueError(f"Oxidation state {ox_val} not found in mask dictionary. Element: {all_elements[pos]}")
            ox_mask[pos] = mask_d[str(ox_val)]
            if radii_d is not None:
                radii_tensor[pos] = radii_d[str(ox_val)]

        atom_mask_for_all_ox_states.append(ox_mask)
        radii_tensor_for_all_ox_states.append(radii_tensor)
    if profile is not None:
        profile["ox_mask_loop_time"] += time.perf_counter() - t0

    t0 = time.perf_counter()
    atom_mask_tensor = torch.tensor(np.stack(atom_mask_for_all_ox_states, axis=-1))
    radii_tensor = torch.tensor(np.stack(radii_tensor_for_all_ox_states, axis=-1))
    ox_states_used_tensor = torch.tensor(ox_states_used_mask)
    site_ids_tensor = torch.tensor(site_ids)
    if profile is not None:
        profile["ox_tensor_time"] += time.perf_counter() - t0

    return atom_mask_tensor, radii_tensor, ox_states_used_tensor, site_ids_tensor
