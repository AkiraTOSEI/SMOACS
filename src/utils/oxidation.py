import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from jarvis.core.atoms import Atoms
from pymatgen.core.periodic_table import Element as Element_pmg
from smact import Element as Element_smact

from scripts.data_generations.neutral_ox_by_n import load_or_build_neutral_ox_dict
from scripts.data_generations.oxidation_sampler import (
    build_ways_per_pos,
    sample_k_per_pos,
)

OX_SOLUTION_MODE_STRICT_UNIQUE = "strict_unique"
OX_SOLUTION_MODE_UP_TO_K = "up_to_k"


def _validate_ox_solution_mode(ox_solution_mode: str) -> None:
    if ox_solution_mode not in {
        OX_SOLUTION_MODE_STRICT_UNIQUE,
        OX_SOLUTION_MODE_UP_TO_K,
    }:
        raise ValueError(
            "ox_solution_mode must be one of "
            f"{OX_SOLUTION_MODE_STRICT_UNIQUE!r}, {OX_SOLUTION_MODE_UP_TO_K!r}. "
            f"Got: {ox_solution_mode!r}"
        )


def _count_neutral_solutions(states_per_pos: List[List[int]]) -> int:
    ways, offset = build_ways_per_pos(states_per_pos, target=0)
    n = len(states_per_pos)
    idx = offset
    if not ways or not (0 <= idx < len(ways[n])):
        return 0
    return int(ways[n][idx])


def _sample_neutral_sequences(
    states_per_pos: List[List[int]],
    max_ox_states: int,
    timeout_sec: Optional[float],
    ox_solution_mode: str,
) -> List[Tuple[int]]:
    _validate_ox_solution_mode(ox_solution_mode)
    start_time = time.perf_counter()

    if timeout_sec is not None and timeout_sec <= 0:
        return []

    # Legacy count semantics: treat max_ox_states as an upper bound, not a
    # requirement. When fewer unique neutral assignments exist, use all of them
    # and let the caller pad the remainder with dummy entries.
    total_solutions = _count_neutral_solutions(states_per_pos)
    if total_solutions == 0:
        return []
    if timeout_sec is not None and (time.perf_counter() - start_time) >= timeout_sec:
        return []

    try:
        k = min(max_ox_states, total_solutions)
        if k <= 0:
            return []

        sampled = sample_k_per_pos(
            states_per_pos,
            k=k,
            target=0,
            seed=None,
            unique=True,
        )
        # sample_k_per_pos reconstructs solutions from the tail position first.
        # Reverse here so oxidation states remain aligned with the input site order.
        return [tuple(reversed(seq)) for seq in sampled]
    except ValueError:
        return []


def elec_neutral_all_common(
    elements: List[str],
    stoichs: List[List[int]],
    max_ox_states: int,
    timeout_sec: Optional[float] = None,
    ox_solution_mode: str = OX_SOLUTION_MODE_STRICT_UNIQUE,
) -> List[Tuple[int]]:
    """
    Determine all possible combinations of oxidation states for a list of
    elements that result in electrical neutrality.
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
    sampled = _sample_neutral_sequences(
        states_per_pos=states_per_pos,
        max_ox_states=max_ox_states,
        timeout_sec=timeout_sec,
        ox_solution_mode=ox_solution_mode,
    )
    return sampled, all_elements


def elec_neutral_all_common_loose(
    elements: List[str],
    stoichs: List[List[int]],
    max_ox_states: int,
    timeout_sec: Optional[float] = None,
    ox_solution_mode: str = OX_SOLUTION_MODE_STRICT_UNIQUE,
) -> List[Tuple[int]]:
    """
    Determine all possible combinations of oxidation states for a list of
    elements that result in electrical neutrality using the loose SMACT-only
    oxidation-state set.
    """
    all_elements = []
    for elem, stoi in zip(elements, stoichs):
        assert len(stoi) == 1
        all_elements.extend([elem] * stoi[0])
    states_per_pos = [list(set(Element_smact(elem).oxidation_states)) for elem in all_elements]
    sampled = _sample_neutral_sequences(
        states_per_pos=states_per_pos,
        max_ox_states=max_ox_states,
        timeout_sec=timeout_sec,
        ox_solution_mode=ox_solution_mode,
    )
    return sampled, all_elements


def get_atom_masks_for_oxidation_states(
    atoms: Atoms,
    mask_d: Dict[str, np.ndarray],
    max_ox_states: int,
    crystal_system: Optional[str],
    initial_dataset: str = "megnet",
    graph_update_mode: bool = False,
    radii_d: Optional[Dict[str, np.ndarray]] = None,
    num_type_of_atoms: int = 98,
    profile: Optional[Dict[str, float]] = None,
    timeout_sec: Optional[float] = None,
    mask_name: Optional[str] = None,
    ox_solution_mode: str = OX_SOLUTION_MODE_STRICT_UNIQUE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate atom masks for all possible oxidation state combinations
    consistent with charge neutrality for the provided crystal.
    """
    if profile is not None:
        profile.setdefault("ox_state_search_time", 0.0)
        profile.setdefault("ox_mask_loop_time", 0.0)
        profile.setdefault("ox_tensor_time", 0.0)

    if initial_dataset == "megnet":
        elec_neutral_all_common_func = elec_neutral_all_common
    elif initial_dataset == "jarvis_supercon":
        elec_neutral_all_common_func = elec_neutral_all_common_loose
    else:
        raise NotImplementedError(
            f"initial_dataset, {initial_dataset}, is not implemented yet."
        )

    if graph_update_mode:
        assert crystal_system is None, "crystal_system must be None for graph update mode."
        site_ids = [np.nan] * len(atoms.elements)
        neutral_ox_states = []
    elif crystal_system == "perovskite":
        p1_list, p2_list = [], []
        site_ids = []
        for elem in atoms.elements:
            if elem == "Ti":
                p1_list.append(2)
                p2_list.append(4)
                site_ids.append(1)
            elif elem == "Ba":
                p1_list.append(1)
                p2_list.append(2)
                site_ids.append(0)
            elif elem == "O":
                p1_list.append(-1)
                p2_list.append(-2)
                site_ids.append(2)
            else:
                raise ValueError("Invalid element for perovskite crystal mode.")
        neutral_ox_states = [tuple(p1_list), tuple(p2_list)]
        assert radii_d is not None, "Radii mask dictionary is required for perovskite crystal mode."
        all_elements = list(atoms.elements)
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
            all_elements = list(atoms.elements)
            stoichs = [[1] for _ in all_elements]
            neutral_ox_states, all_elements = elec_neutral_all_common_func(
                all_elements,
                stoichs,
                max_ox_states=max_ox_states,
                timeout_sec=timeout_sec,
                ox_solution_mode=ox_solution_mode,
            )
            if len(neutral_ox_states) == 0:
                raise TimeoutError("No neutral oxidation states found within timeout.")
        if profile is not None:
            profile["ox_state_search_time"] += time.perf_counter() - t0
        site_ids = [np.nan] * len(atoms.elements)
    else:
        raise ValueError("Invalid crystal_system. Choose from ['perovskite', None].")

    dummy_tuple = tuple([-999] * len(atoms.elements))
    while len(neutral_ox_states) < max_ox_states:
        neutral_ox_states.append(dummy_tuple)

    atom_mask_for_all_ox_states = []
    radii_tensor_for_all_ox_states = []
    ox_states_used_mask = []
    t0 = time.perf_counter()
    for ox_states in neutral_ox_states:
        ox_mask = np.zeros((len(atoms.elements), num_type_of_atoms))
        radii_tensor = np.zeros((len(atoms.elements), num_type_of_atoms))

        if ox_states == dummy_tuple:
            ox_states_used_mask.append(0)
            atom_mask_for_all_ox_states.append(ox_mask)
            radii_tensor_for_all_ox_states.append(radii_tensor)
            continue

        ox_states_used_mask.append(1)
        for pos, ox_val in enumerate(ox_states):
            if str(ox_val) not in mask_d:
                raise ValueError(
                    "Oxidation state "
                    f"{ox_val} not found in mask dictionary. Element: {all_elements[pos]}"
                )
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
