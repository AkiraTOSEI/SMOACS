#!/usr/bin/env python
# coding: utf-8
from __future__ import annotations

#
# ## Wyckoff座領からatomic_distribution, atomic_coordinateを取得する
# - materials projectからダウンロードした構造と、materials projectのWyckoff座標を参照しながら作った構造で、structure matchしなかった。何かがおかしいっぽい
#     - だから、最小原子間距離がかなり小さい構造が頻発していたのだと思う
# - materials project のBaTiO3の構造を再現するようにしてみたが、うまく行かない

# In[ ]:


# # データ生成
# ## 関連関数

# In[ ]:


# ### Wyckoff位置に応じた酸化数パターンの生成関数

# In[30]:


import itertools
import os
import random
import shutil
import time
from pathlib import Path
from typing import Iterable, Sequence, Tuple, Union, Any, Optional

import numpy as np
import pandas as pd
import torch
from IPython.display import clear_output, display
from jarvis.core.atoms import pmg_to_atoms
from periodictable import elements as pd_elements
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element as PmgElement
from pymatgen.core import Lattice, Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.analyzer import SymmetryUndeterminedError
from smact import Element as Element_smact
from tqdm.auto import tqdm


import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.spg_wp.P321 import P321Wyckoff
from src.spg_wp.Fbar43m import Fbar43mWyckoff
from src.spg_wp.Amm2 import Amm2Wyckoff
from src.spg_wp.R3m import R3mWyckoff
from src.spg_wp.P6_3m import P63mWyckoff
from src.spg_wp.P4mmm import P4mmmWyckoff
from src.spg_wp.P2m import P2mWyckoff
from src.spg_wp.wyckoff_expander import WyckoffExpander, wyckoff_letter_index


API_KEY_ENV = "MP_API_KEY"


# In[31]:


class MaterialsError(Exception):
    pass


# In[32]:


def check_nearest_neighbors_distance(
    structure, r=1.0, verbose=0, return_shortest_dist=False
):
    """Check whether pairwise distances satisfy a cutoff.

    Distances are computed from Pymatgen's periodic distance matrix. When
    ``return_shortest_dist`` is true, the shortest off-diagonal distance is
    returned directly.

    Args:
        structure: A Pymatgen ``Structure`` (or an object accepted by ``pmg_to_atoms``).
        r (float): Cutoff distance in angstrom. Pairs with distance ``<= r`` are flagged.
        verbose (int): If ``1``, print each unique violating pair; otherwise, print nothing.

    Returns:
        bool: ``True`` if **no** pairs are found within the cutoff (i.e., all interatomic
        distances are ``> r``), ``False`` if at least one pair violates the cutoff.

    Raises:
        ValueError: If ``r`` is not positive.

    """
    if r <= 0:
        raise ValueError("r must be positive.")
    n_sites = len(structure)
    if n_sites <= 1:
        return float("inf") if return_shortest_dist else True

    dm = np.asarray(structure.distance_matrix, dtype=float)
    tri = np.triu_indices(n_sites, k=1)
    dists = dm[tri]

    if return_shortest_dist:
        return float(np.min(dists))

    violating = np.where(dists <= r)[0]
    if verbose == 1 and violating.size > 0:
        for idx in violating:
            i = int(tri[0][idx])
            j = int(tri[1][idx])
            print(
                f"bond between Atom-{i} {structure[i].species_string} "
                f"({structure[i].x:.3f},{structure[i].y:.3f},{structure[i].z:.3f}) "
                f"--- Atom-{j} {structure[j].species_string} "
                f"({structure[j].x:.3f},{structure[j].y:.3f},{structure[j].z:.3f}), "
                f"distance: {dm[i, j]:.3f} Å"
            )
    return violating.size == 0


# In[33]:


def charge_neutral_states(num_atom_at_WP, min_ox=-4, max_ox=6):
    """
    Enumerate all possible oxidation states for each Wyckoff position
    in the integer range [-4, +6] and return the combinations that
    satisfy Σ n_i * ox_i = 0.

    Parameters
    ----------
    num_atom_at_WP : (k,) array-like
        Number of atoms at each Wyckoff position.
    min_ox, max_ox : int
        Minimum and maximum allowed oxidation states (inclusive).

    Returns
    -------
    combos_neutral : (m, k) ndarray
        Oxidation-state combinations (m rows) that satisfy charge neutrality.
    """
    n = np.asarray(num_atom_at_WP, dtype=int)
    ox_range = np.arange(min_ox, max_ox + 1, dtype=int)  # [-4, -3, ..., 6]
    ox_range.remove(0)  # Exclude oxidation state 0 to enforce charge neutrality
    k = n.size

    # (1) Construct all oxidation-state combinations as a k-dimensional meshgrid
    grid = np.indices((ox_range.size,) * k, dtype=int)
    # shape = (k, ox_range.size**k). Transpose and reshape to (total, k)
    combos = ox_range[grid].reshape(k, -1).T

    # (2) Compute total charge (= Σ n_i * ox_i) in a vectorized manner via dot product
    total_charge = combos @ n

    # (3) Keep only rows with total charge 0
    combos_neutral = combos[total_charge == 0]
    return combos_neutral


# In[34]:


def neutral_oxidation_states(num_atom_at_WP, min_ox=-4, max_ox=6):
    """
    Enumerate oxidation-state combinations that satisfy charge neutrality
    (Σ n_i * ox_i = 0) using only vectorized operations. Oxidation state 0
    is excluded from the candidates.

    Parameters
    ----------
    num_atom_at_WP : array-like, shape (k,)
        Number of atoms at each Wyckoff position.
    min_ox, max_ox : int
        Minimum and maximum oxidation states (inclusive).

    Returns
    -------
    combos_neutral : ndarray, shape (m, k)
        m combinations of oxidation states that satisfy charge neutrality
        and contain no zero oxidation states.
    """
    n = np.asarray(num_atom_at_WP, dtype=np.int64)
    ox_range = np.concatenate((np.arange(min_ox, 0), np.arange(1, max_ox + 1))).astype(
        np.int64
    )
    k = n.size

    # Build all combinations (total, k) at once
    grid = np.indices((ox_range.size,) * k, dtype=np.int64)
    combos = ox_range[grid].reshape(k, -1).T  # (total, k)

    # Filter by charge-neutrality condition
    return combos[(combos @ n) == 0]


# In[35]:


def remove_equivalent_states(
    combos: np.ndarray,
    site_ids: np.ndarray,
) -> np.ndarray:
    """Remove duplicate oxidation-state vectors based on interchangeable site IDs.

    Given a matrix whose rows are candidate oxidation-state assignments and
    a parallel list of integer ``site_ids`` that label which Wyckoff positions
    are interchangeable, this function collapses rows that differ only by
    permuting columns with the same ``site_id``.

    Args:
        combos (np.ndarray):
            An **(m, k)** integer array produced by
            ``neutral_oxidation_states``. Each row is one oxidation-state
            assignment; each column corresponds to a Wyckoff position.
        site_ids (Sequence[int]):
            Length-k sequence of integers. Columns that share the same
            ID are considered equivalent and may be swapped without
            producing a new, distinct assignment.

    Returns:
        np.ndarray:
            A deduplicated **(m', k)** array where each equivalence
            class of oxidation-state vectors is represented exactly once
            ``(m' ≤ m)``.

    Raises:
        ValueError: If the number of columns in ``combos`` does not match
            the length of ``site_ids``.
    """
    combos = np.asarray(combos, dtype=np.int8)
    site_ids = np.asarray(site_ids, dtype=int)

    if combos.shape[1] != site_ids.size:
        raise ValueError("Mismatch between combos.shape[1] and the length of site_ids.")

    # --- Step 1 ─ Canonicalize rows by sorting within each equal-ID group ----
    combos_norm = combos.copy()
    for sid in np.unique(site_ids):
        cols = np.where(site_ids == sid)[0]  # indices of the current ID
        if cols.size > 1:  # skip single-column groups
            combos_norm[:, cols] = np.sort(combos_norm[:, cols], axis=1)

    # --- Step 2 ─ Remove exact duplicates -----------------------------------
    unique_combos = np.unique(combos_norm, axis=0)
    return unique_combos


# In[36]:


def stochastic_sort_weighted(
    combos: Sequence[Sequence[int]],
    *,
    gamma: float = 1.0,
    temperature: Optional[float] = None,
    eps: float = 1e-3,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return `combos` in a stochastic order where lighter |Σox| vectors
    are *more likely* to appear early.

    Args
    ----
    combos : Sequence[Sequence[int]]
        (m, k) array-like of oxidation-state assignments.
    gamma : float, default 1.0
        Strength of the inverse-score weighting:
        ``weight ∝ (1 / (|Σox| + eps)) ** gamma``.
    temperature : float | None, default None
        If provided, an additional Boltzmann re-weight is applied:
        ``weight ← exp(log(weight)/T)``.
        Lower T (= colder) sharpens the preference; higher T flattens it.
    eps : float, default 1e-3
        Small constant to avoid divide-by-zero.
    rng : np.random.Generator | None
        Optional NumPy random generator; falls back to ``np.random.default_rng()``.

    Returns
    -------
    np.ndarray
        Stochastically ordered copy of `combos`, shape (m, k).
    """
    rng = rng or np.random.default_rng()
    combos = np.asarray(combos, dtype=np.int8)
    scores = np.sum(np.abs(combos), axis=1)  # |Σ ox|
    weights = (1.0 / (scores + eps)) ** gamma  # power-law bias

    if temperature is not None:
        # Boltzmann tempering (optional second bias)
        weights = np.exp(np.log(weights) / temperature)

    out = []
    while combos.size:
        probs = weights / weights.sum()
        idx = rng.choice(len(combos), p=probs)
        out.append(combos[idx])
        combos = np.delete(combos, idx, axis=0)
        weights = np.delete(weights, idx)
    return np.stack(out)


# In[37]:


def assert_consistent_multiplicities(
    num_atom_at_WP: Sequence[int | np.integer],
    site_ids: Sequence[int | np.integer],
) -> None:
    """Assert that all Wyckoff positions with the same ``site_id`` share
    an identical multiplicity.

    Parameters
    ----------
    num_atom_at_WP : Sequence[int]
        Multiplicity (number of atoms) assigned to each Wyckoff position.
    site_ids : Sequence[int]
        Integer identifier of each Wyckoff position.  Columns with the
        same ID are considered the *same Wyckoff letter* (e.g. all
        “4a” sites).

    Raises
    ------
    AssertionError
        If at least one ``site_id`` maps to more than one distinct
        multiplicity.
    ValueError
        If the two input sequences differ in length.
    """
    num_atom_at_WP = np.asarray(num_atom_at_WP, dtype=int)
    site_ids = np.asarray(site_ids, dtype=int)

    if num_atom_at_WP.size != site_ids.size:
        raise ValueError("num_atom_at_WP and site_ids must have the same length.")

    for sid in np.unique(site_ids):
        mask = site_ids == sid
        uniq = np.unique(num_atom_at_WP[mask])
        assert (
            uniq.size == 1
        ), f"Inconsistent multiplicities for site_id {sid}: {uniq.tolist()}"


# In[38]:


def generate_sorted_oxidation_states(
    num_atom_at_WP: Sequence[int],
    site_ids: Sequence[int],
    *,
    gamma: float = 2.0,
) -> np.ndarray:
    """Pipeline: multiplicity check → neutral states → deduplication
    by site IDs → stochastic, score-biased sorting.

    Args
    ----
    num_atom_at_WP : Sequence[int]
        Multiplicity (number of atoms) for each Wyckoff position.
    site_ids : Sequence[int]
        Integer site ID for each Wyckoff position; columns with the same
        ID are interchangeable.
    gamma : float, default 2.0
        Weight-shaping exponent for ``stochastic_sort_weighted``.
        Larger values push low-|Σox| vectors even earlier.

    Returns
    -------
    np.ndarray
        Stochastically ordered, deduplicated, charge-neutral oxidation
        states.  Shape = (m, k).

    Notes
    -----
    The function assumes the following helper utilities are already
    defined in the namespace:

    * ``assert_consistent_multiplicities``
    * ``neutral_oxidation_states``
    * ``remove_equivalent_states``
    * ``stochastic_sort_weighted``
    """
    # 1. Consistency check (raises AssertionError / ValueError on failure)
    assert_consistent_multiplicities(num_atom_at_WP, site_ids)

    # 2. Enumerate all charge-neutral assignments (0 excluded internally)
    all_combos = neutral_oxidation_states(num_atom_at_WP)

    # 3. Collapse rows that differ only by swapping equal site IDs
    unique_combos = remove_equivalent_states(all_combos, site_ids)

    # 4. Stochastic, gamma-weighted ordering (copy as int32 for safety)
    sorted_combos = stochastic_sort_weighted(
        unique_combos.astype(np.int32), gamma=gamma
    )
    return sorted_combos


def canonicalize_oxidation_pattern_by_site_ids(
    oxidation_pattern: Sequence[int | np.integer],
    site_ids: Sequence[int | np.integer],
) -> np.ndarray:
    """Canonicalize one oxidation vector by sorting values within each equal-site_id block."""
    ox = np.asarray(oxidation_pattern, dtype=np.int32).copy()
    site_ids = np.asarray(site_ids, dtype=int)
    if ox.size != site_ids.size:
        raise ValueError("oxidation_pattern and site_ids must have the same length.")
    for sid in np.unique(site_ids):
        cols = np.where(site_ids == sid)[0]
        if cols.size > 1:
            ox[cols] = np.sort(ox[cols])
    return ox


def is_all_metal_wyckoff_sites(symm_conv: Any) -> bool:
    """Return True if all representative Wyckoff-site elements are metals."""
    try:
        wp_elements = [
            eq_sites[0].species_string for eq_sites in symm_conv.equivalent_sites
        ]
        return all(PmgElement(elem).is_metal for elem in wp_elements)
    except Exception:
        return False


def infer_neutral_wp_oxidation_pattern(
    structure: Structure,
    symm_conv: Any,
    site_ids: Sequence[int | np.integer],
    num_atom_at_WP: Sequence[int | np.integer],
    min_ox: int = -4,
    max_ox: int = 6,
) -> np.ndarray | None:
    """Infer one neutral WP oxidation pattern using SMACT oxidation states.

    The search is constrained per-WP by element-specific oxidation states
    from SMACT, then solved with a DFS + charge-bound pruning under:
    ``sum(num_atom_at_WP[i] * ox_i) == 0``.
    """
    site_ids = np.asarray(site_ids, dtype=int)
    num_atom_at_WP = np.asarray(num_atom_at_WP, dtype=int)
    if site_ids.size != num_atom_at_WP.size:
        raise ValueError("site_ids and num_atom_at_WP must have the same length.")

    wp_elements = [
        eq_sites[0].species_string for eq_sites in symm_conv.equivalent_sites
    ]
    if len(wp_elements) != num_atom_at_WP.size:
        return None

    candidate_states: list[list[int]] = []
    for elem in wp_elements:
        try:
            smact_states = {int(round(v)) for v in Element_smact(elem).oxidation_states}
        except Exception:
            return None

        usable_states = {
            ox for ox in smact_states if ox != 0 and min_ox <= ox <= max_ox
        }
        if not usable_states:
            return None

        # Smaller |ox| first to get a deterministic, mild-charge solution.
        candidate_states.append(sorted(usable_states, key=lambda x: (abs(x), x)))

    k = num_atom_at_WP.size
    contributions = [
        np.array([int(num_atom_at_WP[i] * ox) for ox in candidate_states[i]], dtype=int)
        for i in range(k)
    ]

    suffix_min = np.zeros(k + 1, dtype=int)
    suffix_max = np.zeros(k + 1, dtype=int)
    for i in range(k - 1, -1, -1):
        suffix_min[i] = suffix_min[i + 1] + int(np.min(contributions[i]))
        suffix_max[i] = suffix_max[i + 1] + int(np.max(contributions[i]))

    inferred = np.zeros(k, dtype=np.int32)

    def _dfs(i: int, charge: int) -> bool:
        if i == k:
            return charge == 0

        for ox in candidate_states[i]:
            next_charge = charge + int(num_atom_at_WP[i] * ox)
            if next_charge + suffix_min[i + 1] > 0:
                continue
            if next_charge + suffix_max[i + 1] < 0:
                continue

            inferred[i] = int(ox)
            if _dfs(i + 1, next_charge):
                return True
        return False

    if not _dfs(0, 0):
        return None

    inferred = canonicalize_oxidation_pattern_by_site_ids(inferred, site_ids)
    if int(np.dot(num_atom_at_WP, inferred)) != 0:
        return None
    return inferred


# In[39]:


def get_num_WP(d):
    """Return the number of Wyckoff positions in a document."""
    return len(
        (SpacegroupAnalyzer(d["structure"]).get_symmetrized_structure()).wyckoff_symbols
    )


# In[40]:


def structure_check(
    structure: Structure,
    wyckoff_expander: WyckoffExpander,
    dist_min: float,
    dist_max: Optional[float] = None,
):
    """
    The function applies two consecutive tests:

    1. **Nearest-neighbor distance test** –
       Calls ``check_nearest_neighbors_distance`` to confirm that every
       interatomic distance is larger than or equal to *r* Å.

    2. **Space-group consistency test** –
       Re-computes the space-group number with
       :class:`pymatgen.symmetry.analyzer.SpacegroupAnalyzer` and checks it
       against ``wyckoff_expander.spg_number``.

    The function returns ``True`` only if **both** tests pass.
    A mismatch in the space-group number
    """
    if dist_max is None:
        dist_max = 6.0  # set a default maximum distance
    #
    if minimum_distance_within_bounds(structure, dist_min, dist_max):
        try:
            spg_number = SpacegroupAnalyzer(structure).get_space_group_number()
        except SymmetryUndeterminedError:
            print("Could not determine space group.")
            return False
        if spg_number == wyckoff_expander.spg_number:
            return True
        else:
            # print(f"Space group number {spg_number} does not match expected {wyckoff_expander.spg_number}.")
            return False
    else:
        # print("Nearest neighbor distance check failed.")
        return False


# In[41]:


def generate_valid_structure(
    wp_atom_dist_vec: torch.Tensor,
    site_ids: torch.Tensor,
    wyckoff_expander: WyckoffExpander,
    structure: Structure,
    max_trials: int = 1_000,
    r_min: float = 1.0,
    r_max: Optional[float] = None,
    a_min: float = 3.0,
    a_max: float = 12.0,
    device: Union[torch.device, str] = "cpu",
    random_abc: bool = False,
) -> Tuple[Structure, torch.Tensor, torch.Tensor]:
    """Return the first randomly generated structure that passes `structure_check`.

    Args:
        wp_atom_dist_vec (torch.Tensor): (N_wp, N_elem) probability matrix.
        site_ids (torch.Tensor): Wyckoff site IDs (N_wp,).
        wyckoff_expander (object): Must expose ``wyckoff_expand`` method.
        max_trials (int, optional): Max attempts. Defaults to 1 000.
        r_min (float, optional): Cut-off forwarded to `structure_check`. Defaults to 1.0.
        r_max (float | None, optional): Max distance for `structure_check`. Defaults to None.
        a_min (float, optional): Min lattice length. Defaults to 3.0 Å.
        a_max (float, optional): Max lattice length. Defaults to 10.0 Å.
        device (torch.device | str, optional): Tensor device. Defaults to "cpu".

    Returns:
        tuple[Structure, torch.Tensor, torch.Tensor]:
            * structure – Valid `pymatgen` structure.
            * random_wp_coords – (N_wp, 3) fractional coordinates.
            * random_abc – (3,) lattice lengths.

    Raises:
        RuntimeError: No valid structure found within `max_trials`.
    """
    wp_atom_dist_vec, site_ids = wp_atom_dist_vec.to(device), site_ids.to(device)

    wp_atom_dist_vec, site_ids = wp_atom_dist_vec.to(device), site_ids.to(device)

    for _ in tqdm(range(max_trials), desc="max_trials in WP", leave=False):
        # randomly generate wp_coords
        wp_coords = torch.rand(len(wp_atom_dist_vec), 3, device=device)
        coords, atom_vec = wyckoff_expander.wyckoff_expand(
            wp_coords, wp_atom_dist_vec, site_ids
        )
        species = [
            pd_elements[int(i) + 1].symbol for i in torch.argmax(atom_vec, 1).cpu()
        ]

        if random_abc:
            lattice = randomize_lattice(structure)
        else:
            lattice = structure.lattice

        struct = Structure(lattice, species, coords)
        try:
            if structure_check(
                struct, wyckoff_expander, dist_min=r_min, dist_max=r_max
            ):
                abc = torch.tensor(
                    [
                        structure.lattice.a,
                        structure.lattice.b,
                        structure.lattice.c,
                    ],
                    device=device,
                )
                return struct, wp_coords, abc
        except TypeError:
            # sometimes occours in pymatgen's structure check
            continue

    raise MaterialsError(
        f"Failed to generate a valid structure in {max_trials} trials."
    )


# In[42]:


def build_atomic_mask_for_oxidation_patterns(
    ox_mask_path: Union[str, Path],
    oxidation_table: Sequence[Sequence[int]],
    num_atom_at_WP: Sequence[int],
    max_ox_patterns: int = 100,
) -> np.ndarray:
    """Create a 3-D mask tensor for every oxidation-state pattern.

    Each oxidation pattern in ``oxidation_table`` is mapped to an
    `(N_wp, 98)` binary mask stored on disk (*ox_mask_path*).
    Patterns are stacked along the third axis; missing patterns are padded
    with an all-zero mask so that the final array always has
    shape ``(N_wp, 98, max_ox_patterns)``.

    Parameters
    ----------
    ox_mask_path : str | pathlib.Path
        Path to the ``.npy`` file containing a dictionary that maps
        *str(oxidation_state)* → mask array.
    oxidation_table : Sequence[Sequence[int]]
        List of oxidation-state patterns; each pattern is a list/tuple of
        integer oxidation states (e.g. ``[2, -2]``).
    num_atom_at_WP : Sequence[int]
        Multiplicity (number of atoms) for each Wyckoff position; used only
        for sizing the zero-padding mask.
    max_ox_patterns : int, default 100
        Maximum number of patterns to include in the output tensor.
        If ``len(oxidation_table)`` exceeds this value, the list is truncated;
        if it is shorter, zero masks are appended.

    Returns
    -------
    np.ndarray
        Binary mask tensor with shape ``(N_wp, 98, max_ox_patterns)`` where
        *N_wp* = ``len(num_atom_at_WP)``.

    Notes
    -----
    * The second dimension (98) assumes oxidation masks were pre-computed
      for the first 98 elements (H–Cf). Adapt as needed.
    * The function does **not** validate that the oxidation states in
      *oxidation_table* exist in *ox_mask_path*; missing keys will raise
      ``KeyError`` from NumPy.
    """
    mask_dict = np.load(ox_mask_path, allow_pickle=True)

    # ------------------------------------------------------------------
    # Build mask list for the requested patterns
    # ------------------------------------------------------------------
    mask_list = []
    for ox_pattern in oxidation_table[:max_ox_patterns]:
        mask_list.append([mask_dict[str(ox)] for ox in ox_pattern])

    # ------------------------------------------------------------------
    # Zero-pad if we have fewer than max_ox_patterns
    # ------------------------------------------------------------------
    n_missing = max_ox_patterns - len(mask_list)
    if n_missing > 0:
        zero_mask = np.zeros((len(num_atom_at_WP), 98))
        mask_list.extend([zero_mask] * n_missing)

    # Convert to ndarray and reorder axes: (patterns, N_wp, 98) → (N_wp, 98, patterns)
    mask_tensor = np.transpose(np.array(mask_list), (1, 2, 0))
    return mask_tensor


# In[43]:


def prepare_data(
    structure: Structure,
    sample_id: Union[int, str],
    material_id: str,
    oxidation_table: Sequence[Sequence[int]],
    ox_mask_path: Union[str, Path],
    num_atom_at_WP: Sequence[int],
    max_ox_patterns: int,
    site_ids: torch.Tensor,
    wp_coords: torch.Tensor,
    saved_dir: Union[str, Path],
) -> dict:
    """Pack all per-sample information required by ``load_graph``.

    Args:
        structure: Full *pymatgen* structure.
        sample_id: Arbitrary ID (used in ``jid``).
        material_id: e.g. Materials Project ID.
        oxidation_table: List of oxidation-state patterns.
        ox_mask_path: ``.npy`` file: ``str(ox_state) → (N_wp, 98) mask``.
        num_atom_at_WP: Wyckoff multiplicities (length =N_wp).
        max_ox_patterns: Tensor depth for oxidation patterns.
        site_ids: Wyckoff site IDs (shape = ``(N_wp,)``).
        wp_coords: Wyckoff fractional coords (shape = ``(N_wp, 3)``).
        saved_dir: Directory where the POSCAR will be written.

    Returns:
        dict: Keys = ``['atoms', 'dummy_target', 'jid', 'ox_states_used_mask',
        'atom_mask_for_all_ox_states', 'radii_for_all_ox_states',
        'elements', 'site_ids', 'wp_coords']``.
    """
    nWp_site = len(num_atom_at_WP)
    # --- geometry & basic arrays -----------------------------------------
    ## atoms
    atoms = pmg_to_atoms(structure)

    ## dummy target
    dummy_target = 0

    ## jid
    jid = f"Sample-{sample_id}__{material_id}.vasp"

    ## mask for oxidation patterns
    ox_used_mask = np.zeros(max_ox_patterns)
    ox_used_mask[: len(oxidation_table)] = 1.0

    ## atom_mask_for_all_ox_states
    atomic_mask_for_all_pattern = build_atomic_mask_for_oxidation_patterns(
        ox_mask_path=ox_mask_path,
        oxidation_table=oxidation_table,
        num_atom_at_WP=num_atom_at_WP,
        max_ox_patterns=max_ox_patterns,
    )

    # tensor for ionic radii
    radii_for_all_ox_states = np.zeros(
        shape=atomic_mask_for_all_pattern.shape, dtype=np.float32
    )

    ## elements
    elements = [site.species_string for site in structure.sites]

    ## site_ids
    site_ids = site_ids.numpy()  # Wyckoff position ID (0-based)

    # --- sanity checks ----------------------------------------------------
    assert (
        atomic_mask_for_all_pattern.shape[0] == nWp_site == len(wp_coords)
    ), "atomic_mask_for_all_pattern.shape[0] must match nWp_site"
    assert len(site_ids) == nWp_site, "site_ids must have the same length as nWp_site"
    assert (
        atomic_mask_for_all_pattern.shape[2] == max_ox_patterns
    ), "atomic_mask_for_all_pattern.shape[2] must match max_ox_patterns"
    assert torch.equal(
        torch.sum(torch.tensor(atomic_mask_for_all_pattern), dim=[0, 1]) > 0,
        torch.tensor(ox_used_mask, dtype=torch.bool),
    )

    # --- save POSCAR ------------------------------------------------------
    atoms.write_poscar(os.path.join(saved_dir, f"{jid}"))

    # --- bundle and return -----------------------------------------------
    dict_data = {
        "atoms": atoms.to_dict(),
        "dummy_target": dummy_target,
        "jid": jid,
        "ox_states_used_mask": ox_used_mask,
        "atom_mask_for_all_ox_states": atomic_mask_for_all_pattern,
        "radii_for_all_ox_states": radii_for_all_ox_states,
        "elements": elements,
        "site_ids": site_ids,
        "wp_coords": wp_coords.numpy(),  # (N_wp, 3)
        "atomic_density": atoms.num_atoms / atoms.volume,
    }
    print(
        f"Atomic density: {atoms.num_atoms / atoms.volume:.3f} ({atoms.num_atoms} / {atoms.volume:.3f}) atoms/Å³"
    )

    return dict_data


# In[ ]:


# In[44]:


def compute_min_max_nn_distance(
    all_data_rows: Iterable[Sequence[Any]],
) -> Tuple[float, float]:
    """Compute the minimum and maximum of the shortest nearest-neighbor distances.

    For each row in `all_data_rows`, this function expects the 5th element (index 4)
    to be a mapping (e.g., dict) containing a "structure" entry. It calls
    `check_nearest_neighbors_distance(structure, return_shortest_dist=True, r=50)`
    to obtain the shortest distance for that structure, then returns the global
    minimum and maximum across all rows.

    Args:
        all_data_rows: An iterable of sequences. Each sequence must have at least
            five elements, where the element at index 4 is a mapping with key
            "structure" holding the target structure object.

    Returns:
        A tuple (dist_min, dist_max) where:
            - dist_min: The minimum shortest nearest-neighbor distance.
            - dist_max: The maximum shortest nearest-neighbor distance.

    Raises:
        ValueError: If `all_data_rows` is empty, or a row does not provide a
            mapping at index 4 with a "structure" key.
        Any exception raised by `check_nearest_neighbors_distance` will propagate
            unless wrapped by the ValueError above.
    """
    shortest_lengths: list[float] = []

    for row in all_data_rows:
        doc = row[5]
        structure = doc["structure"]
        d = check_nearest_neighbors_distance(structure, return_shortest_dist=True, r=50)
        shortest_lengths.append(float(d))

    dist_min = min(shortest_lengths)
    dist_max = max(shortest_lengths)
    display(pd.DataFrame(shortest_lengths, columns=["shortest_nn_distance"]).describe())
    return dist_min, dist_max


# In[45]:


def minimum_distance_within_bounds(
    struct: Structure, dist_min: float, dist_max: float
) -> bool:
    """Check if the shortest nearest-neighbor distance of a structure is within specified bounds.

    Args:
        struct: The structure to check.
        dist_min: The minimum acceptable distance.
        dist_max: The maximum acceptable distance.

    Returns:
        True if the shortest nearest-neighbor distance is greater than dist_min
        and less than dist_max, False otherwise.
    """
    return (
        dist_min
        < check_nearest_neighbors_distance(structure=struct, return_shortest_dist=True)
        < dist_max
    )


# In[46]:


def group_equal(values: np.ndarray, atol: float) -> list[list[int]]:
    """Group indices of approximately equal values.

    Elements are grouped if they satisfy ``np.isclose`` with the provided
    absolute tolerance. Each index appears in exactly one group.

    Args:
        values: 1D array of numeric values to group.
        atol: Absolute tolerance for equality testing.

    Returns:
        A list of groups, where each group is a list of indices that are
        approximately equal.
    """
    groups, used = [], set()
    for i, ref in enumerate(values):
        if i in used:
            continue
        group = [i]
        for j in range(i + 1, len(values)):
            if j in used:
                continue
            if np.isclose(values[j], ref, rtol=1e-6, atol=atol):
                group.append(j)
        groups.append(group)
        used.update(group)
    return groups


def pinned_angle_value(x: float, tol_ang: float) -> float | None:
    """Return the pinned angle (60, 90 or 120) if x is within tolerance.

    Args:
        x: Angle in degrees.
        tol_ang: Absolute tolerance for deciding pinning.

    Returns:
        90.0 or 120.0 if ``x`` is within ``tol_ang`` of those values; otherwise ``None``.
    """
    if abs(x - 60.0) <= tol_ang:
        return 60.0
    if abs(x - 90.0) <= tol_ang:
        return 90.0
    if abs(x - 120.0) <= tol_ang:
        return 120.0

    return None


def randomize_lattice(structure: Structure) -> Lattice:
    """Return a randomized lattice while preserving equality relations and pinned angles.

    This function constructs a new ``pymatgen.core.lattice.Lattice`` from the input
    ``Structure`` under the following rules:

    - If any of ``a``, ``b``, ``c`` are (approximately) equal in the input lattice,
      that equality is preserved by assigning the *same* new length to the whole group.
    - If any of ``alpha``, ``beta``, ``gamma`` are (approximately) equal in the input lattice,
      that equality is preserved by assigning the *same* new angle to the whole group.
    - Any angle within ±0.001° of 90° or 120° is treated as *pinned* and kept exactly
      at 90° or 120° (group-wide if one member is pinned).
    - Non-pinned angle groups are assigned a random value uniformly from [40°, 140°].
    - Each length group (among a, b, c) is assigned a random value uniformly from [2 Å, 10 Å].

    Notes:
        - Only the lattice is randomized/returned; atomic coordinates are unchanged.
        - Equality is detected with ``np.isclose`` using absolute tolerances below.

    Args:
        structure: Input ``Structure`` whose lattice constraints are to be respected.

    Returns:
        new_lat: A new ``Lattice`` sampled under the above constraints.
    """
    # --- Tunables (kept inside the function to preserve single-argument signature) ---
    LEN_MIN, LEN_MAX = 2.0, 10.0  # Angstrom
    ANG_MIN, ANG_MAX = 40.0, 140.0  # degrees (safe range away from degeneracy)
    TOL_LEN = 1e-3  # Angstrom; equality tolerance for a,b,c
    TOL_ANG = 1e-3  # degrees; equality & 90/120° pin tolerance

    lat = structure.lattice
    len_vals = np.array([lat.a, lat.b, lat.c], dtype=float)
    ang_vals = np.array([lat.alpha, lat.beta, lat.gamma], dtype=float)

    rng = np.random.default_rng()

    # --- Preserve equality among a, b, c: assign a single uniform random length per group ---
    len_groups = group_equal(len_vals, TOL_LEN)
    new_len = len_vals.copy()
    for g in len_groups:
        L = float(rng.uniform(LEN_MIN, LEN_MAX))
        for i in g:
            new_len[i] = L

    # --- Preserve equality among alpha, beta, gamma and pin 90°/120° groups if applicable ---
    ang_groups = group_equal(ang_vals, TOL_ANG)
    new_ang = ang_vals.copy()
    for g in ang_groups:
        # If any member is pinned, snap the whole group to that pinned value (90 or 120).
        pin = None
        for i in g:
            pv = pinned_angle_value(ang_vals[i], TOL_ANG)
            if pv is not None:
                pin = pv
                break
        if pin is not None:
            for i in g:
                new_ang[i] = pin
        else:
            # No pinning in the group: sample one uniform angle for the whole group.
            val = float(rng.uniform(ANG_MIN, ANG_MAX))
            for i in g:
                new_ang[i] = val

    # --- Build and return the new lattice ---
    new_lat = Lattice.from_parameters(
        a=float(new_len[0]),
        b=float(new_len[1]),
        c=float(new_len[2]),
        alpha=float(new_ang[0]),
        beta=float(new_ang[1]),
        gamma=float(new_ang[2]),
    )
    return new_lat


def perturb_lattice_lengths_keep_angles(
    structure: Structure,
    scale_min: float = 0.9,
    scale_max: float = 1.1,
) -> Lattice:
    """Perturb lattice lengths while preserving equality relations; keep angles fixed.

    If any of a/b/c are approximately equal, the same scale factor is applied
    to that whole length group so that metric constraints (e.g. a=b) remain
    satisfied after perturbation.
    """
    lat = structure.lattice
    len_vals = np.array([lat.a, lat.b, lat.c], dtype=float)
    len_groups = group_equal(len_vals, atol=1e-3)
    rng = np.random.default_rng()

    scales = np.ones(3, dtype=float)
    for g in len_groups:
        s = float(rng.uniform(scale_min, scale_max))
        for i in g:
            scales[i] = s

    return Lattice.from_parameters(
        a=float(len_vals[0] * scales[0]),
        b=float(len_vals[1] * scales[1]),
        c=float(len_vals[2] * scales[2]),
        alpha=float(lat.alpha),
        beta=float(lat.beta),
        gamma=float(lat.gamma),
    )


def align_wp_coords_with_axis_permutations(
    wp_coords: torch.Tensor,
    wp_atom_dist_vec: torch.Tensor,
    site_ids: torch.Tensor,
    structure: Structure,
    wyckoff_expander: WyckoffExpander,
) -> tuple[torch.Tensor, Structure]:
    """Align representative WP coordinates to the Wyckoff table axis setting.

    MP conventional cells and the internal Wyckoff table can use different axis
    conventions/origins for the same space group. This helper tries
    axis permutations and origin shifts, and returns the first WP
    representative coordinates whose expanded structure is:
    1) space-group consistent
    2) symmetry-equivalent to the MP conventional reference structure
    """
    matcher = StructureMatcher()
    lattice_mat = np.asarray(structure.lattice.matrix, dtype=float)
    origin_shift_values = (0.0, 0.5)

    for perm in itertools.permutations((0, 1, 2)):
        permuted_wp_coords = wp_coords[:, list(perm)]
        permuted_lattice = Lattice(lattice_mat[list(perm), :])
        for shift in itertools.product(origin_shift_values, repeat=3):
            shift_tensor = torch.tensor(
                shift, dtype=wp_coords.dtype, device=wp_coords.device
            )
            shifted_wp_coords = (permuted_wp_coords + shift_tensor) % 1.0
            coords, atom_vec = wyckoff_expander.wyckoff_expand(
                shifted_wp_coords, wp_atom_dist_vec, site_ids
            )
            species = [
                pd_elements[int(i) + 1].symbol for i in torch.argmax(atom_vec, 1).cpu()
            ]
            struct = Structure(permuted_lattice, species, coords.cpu().numpy())

            try:
                spg_number = SpacegroupAnalyzer(struct).get_space_group_number()
            except SymmetryUndeterminedError:
                continue

            if spg_number != wyckoff_expander.spg_number:
                continue
            if matcher.fit(structure, struct):
                return shifted_wp_coords, struct

    raise MaterialsError(
        "Failed to align WP coordinates to the expected Wyckoff setting."
    )


def _prepare_mp_wyckoff_structure_payload(
    doc: dict,
    wyckoff_expander: WyckoffExpander,
    material_id: str,
    nWp_site: int,
) -> dict[str, Any] | None:
    """Build the aligned MP conventional structure and metadata used by stable/perturbed modes."""
    s: Structure = doc["structure"]
    spa = SpacegroupAnalyzer(s)
    if spa.get_space_group_number() != wyckoff_expander.spg_number:
        return None

    conv_s = spa.get_conventional_standard_structure()
    conv_spa = SpacegroupAnalyzer(conv_s)
    symm_conv = conv_spa.get_symmetrized_structure()

    site_ids: list[int] = []
    wp_atom_dist_vec: list[np.ndarray] = []
    wp_coords: list[np.ndarray] = []
    for wp, eq_sites in zip(symm_conv.wyckoff_symbols, symm_conv.equivalent_sites):
        assert wp in wyckoff_expander.wyckoff_site, (
            f"Unexpected Wyckoff label: {wp} in {doc['material_id']}. "
            f"Valid labels: {wyckoff_expander.wyckoff_site}"
        )

        vec = np.zeros(98, dtype=float)
        atomic_number = pd_elements.symbol(eq_sites[0].species_string).number
        vec[atomic_number - 1] = 1.0
        wp_atom_dist_vec.append(vec)
        site_ids.append(wyckoff_letter_index(wp))
        wp_coords.append(eq_sites[0].frac_coords)

    site_ids_tensor = torch.tensor(site_ids, dtype=torch.long)
    wp_atom_dist_vec_tensor = torch.tensor(wp_atom_dist_vec, dtype=torch.float32)
    wp_coords_tensor = torch.tensor(
        np.asarray(wp_coords, dtype=float), dtype=torch.float32
    )

    num_atom_at_WP = np.array([len(eq) for eq in symm_conv.equivalent_sites])
    if len(num_atom_at_WP) != nWp_site:
        print(
            "(num_atom_at_WP) length mismatch: "
            f"expected {nWp_site}, got {len(num_atom_at_WP)}."
        )
        return None

    site_ids_np = site_ids_tensor.numpy()
    is_all_metal = is_all_metal_wyckoff_sites(symm_conv)
    mp_wp_oxidation_pattern = None
    if not is_all_metal:
        mp_wp_oxidation_pattern = infer_neutral_wp_oxidation_pattern(
            structure=conv_s,
            symm_conv=symm_conv,
            site_ids=site_ids_np,
            num_atom_at_WP=num_atom_at_WP,
        )
        if mp_wp_oxidation_pattern is None:
            print(
                "Failed to infer neutral WP oxidation pattern from MP structure; "
                f"skip material_id={material_id}."
            )
            return None

    try:
        aligned_wp_coords, expanded_structure = align_wp_coords_with_axis_permutations(
            wp_coords=wp_coords_tensor,
            wp_atom_dist_vec=wp_atom_dist_vec_tensor,
            site_ids=site_ids_tensor,
            structure=conv_s,
            wyckoff_expander=wyckoff_expander,
        )
    except MaterialsError:
        print(
            "Failed to align representative WP coordinates with MP conventional setting; "
            f"skip material_id={material_id}."
        )
        return None

    try:
        generated_oxidation_table = generate_sorted_oxidation_states(
            num_atom_at_WP, site_ids_np
        )
    except ValueError as exc:
        print(f"Failed to generate oxidation states: {exc}")
        return None

    generated_oxidation_table = np.asarray(generated_oxidation_table, dtype=np.int32)
    if generated_oxidation_table.size == 0:
        print(f"No oxidation patterns generated; skip material_id={material_id}.")
        return None
    if generated_oxidation_table.ndim == 1:
        generated_oxidation_table = generated_oxidation_table[None, :]

    if is_all_metal:
        base_pattern = generated_oxidation_table[0]
    else:
        base_pattern = mp_wp_oxidation_pattern.astype(np.int32)

    oxidation_table: list[np.ndarray] = [base_pattern]
    for ox_pattern in generated_oxidation_table:
        if not np.array_equal(ox_pattern, base_pattern):
            oxidation_table.append(ox_pattern)

    expected_num_sites = [
        wyckoff_expander.num_equivalent_sites[int(site_id)] for site_id in site_ids_np
    ]
    assert (
        expanded_structure.num_sites
        == np.sum(num_atom_at_WP)
        == np.sum(expected_num_sites)
    )

    return {
        "structure": expanded_structure,
        "site_ids": site_ids_tensor,
        "wp_coords": aligned_wp_coords,
        "num_atom_at_WP": num_atom_at_WP,
        "oxidation_table": oxidation_table,
    }


def generate_MP_Wyckoff_Position_stable_structure_data(
    doc: dict,
    wyckoff_expander: WyckoffExpander,
    sample_id: int | str,
    material_id: str,
    ox_mask_path: str,
    max_ox_patterns: int,
    nWp_site: int,
    saved_dir: str,
) -> dict | None:
    """Generate an MP-based structure without adding any lattice or coordinate noise."""
    payload = _prepare_mp_wyckoff_structure_payload(
        doc=doc,
        wyckoff_expander=wyckoff_expander,
        material_id=material_id,
        nWp_site=nWp_site,
    )
    if payload is None:
        return None

    return prepare_data(
        structure=payload["structure"],
        sample_id=f"MPStable-{sample_id}",
        material_id=material_id,
        oxidation_table=payload["oxidation_table"],
        ox_mask_path=ox_mask_path,
        num_atom_at_WP=payload["num_atom_at_WP"],
        max_ox_patterns=max_ox_patterns,
        site_ids=payload["site_ids"],
        wp_coords=payload["wp_coords"],
        saved_dir=saved_dir,
    )


def generate_MP_Wyckoff_Position_perturbed_lattice_structure_data(
    doc: dict,
    wyckoff_expander: WyckoffExpander,
    sample_id: int | str,
    material_id: str,
    ox_mask_path: str,
    max_ox_patterns: int,
    nWp_site: int,
    saved_dir: str,
    r_min: float = 1.0,
    r_max: Optional[float] = None,
    num_max_trials: int = 100,
    lattice_scale_min: float = 0.9,
    lattice_scale_max: float = 1.1,
) -> dict | None:
    """Generate MP-based structure with fixed WP/internal coords and perturbed lattice lengths."""
    payload = _prepare_mp_wyckoff_structure_payload(
        doc=doc,
        wyckoff_expander=wyckoff_expander,
        material_id=material_id,
        nWp_site=nWp_site,
    )
    if payload is None:
        return None

    # ----------------------------------------------------------------------
    # 6) Perturb only lattice lengths until distance/spg checks pass
    # ----------------------------------------------------------------------
    expanded_structure = payload["structure"]
    coords = np.asarray(expanded_structure.frac_coords, dtype=float)
    species = [site.species_string for site in expanded_structure]

    structure = None
    max_perturb_trials = max(1, min(int(num_max_trials), 100))
    for _ in tqdm(
        range(max_perturb_trials), desc="max_trials in MPPerturb", leave=False
    ):
        lattice = perturb_lattice_lengths_keep_angles(
            expanded_structure,
            scale_min=lattice_scale_min,
            scale_max=lattice_scale_max,
        )
        struct = Structure(lattice, species, coords)
        try:
            if structure_check(
                struct, wyckoff_expander, dist_min=r_min, dist_max=r_max
            ):
                structure = struct
                break
        except TypeError:
            continue

    if structure is None:
        print(
            f"Failed to generate a perturbed-lattice structure in {max_perturb_trials} "
            f"trials with mate-id: {material_id}"
        )
        return None

    # ----------------------------------------------------------------------
    # 7) Assemble output dictionary (existing output format)
    # ----------------------------------------------------------------------
    dict_data = prepare_data(
        structure=structure,
        sample_id=f"MPPerturb-{sample_id}",
        material_id=material_id,
        oxidation_table=payload["oxidation_table"],
        ox_mask_path=ox_mask_path,
        num_atom_at_WP=payload["num_atom_at_WP"],
        max_ox_patterns=max_ox_patterns,
        site_ids=payload["site_ids"],
        wp_coords=payload["wp_coords"],
        saved_dir=saved_dir,
    )
    return dict_data


def generate_MP_Wyckoff_Position_structure_data(
    doc: dict,
    wyckoff_expander: WyckoffExpander,
    sample_id: int | str,
    material_id: str,
    ox_mask_path: str,
    max_ox_patterns: int,
    nWp_site: int,
    saved_dir: str,
    r_min: float = 1.0,
    num_max_trials: int = 1000,
) -> dict | None:
    """
    Generate a new random structure (based on Wyckoff sites) and assemble
    the dataset dictionary required by downstream ``load_graph`` routines.

    Args:
        doc (dict):
            A Mongo/MP-API style document **containing a primitive cell**
            under the key ``"structure"``.
        wyckoff_expander (WyckoffExpander):
            Helper object that must expose
            ``wyckoff_site`` (list of valid labels) and
            ``wyckoff_expand`` (used downstream by
            :pyfunc:`generate_valid_structure`).
        sample_id (int | str):
            User-defined identifier that becomes part of the ``jid`` string.
        material_id (str):
            Materials Project ID or other unique formula/phase label.
        ox_mask_path (str):
            Path to ``*.npy`` file holding oxidation-state masks
            (``str(ox) → (N_wp, 98)`` array).
        max_ox_patterns (int):
            Depth of the oxidation-pattern dimension—extra patterns are
            dropped and missing ones are zero-padded.
        nWp_site (int):
            Expected number of Wyckoff representative sites; used for
            assertion against the multiplicity vector.
        saved_dir (str):
            Directory where the generated POSCAR will be written by
            :pyfunc:`prepare_data`.

    Returns:
        dict | None:
            * A dictionary ready for ``load_graph`` **if** structure
              generation succeeds.
            * ``None`` when no valid structure is found within the allowed
              trials (the failure is printed to stdout).



    """
    # ----------------------------------------------------------------------
    # 0) Start from the primitive cell stored in the document
    # ----------------------------------------------------------------------
    s: Structure = doc["structure"]  # primitive cell
    spa = SpacegroupAnalyzer(s)
    if not spa.get_space_group_number() == wyckoff_expander.spg_number:
        return None  # skip this entry if space group does not match

    # 1) Convert to an A-centered conventional standard cell
    conv_s = spa.get_conventional_standard_structure()

    # 2) Run symmetry analysis again on the conventional cell
    conv_spa = SpacegroupAnalyzer(conv_s)
    symm_conv = conv_spa.get_symmetrized_structure()

    # ----------------------------------------------------------------------
    # 3) Collect representative Wyckoff sites
    # ----------------------------------------------------------------------
    site_ids: list[int] = []
    wp_atom_dist_vec: list[np.ndarray] = []
    species: list[str] = []

    for wp, eq_sites in zip(symm_conv.wyckoff_symbols, symm_conv.equivalent_sites):
        # The Wyckoff label should be back to 2a, 2b, 4c, …
        assert (
            wp in wyckoff_expander.wyckoff_site
        ), f"Unexpected Wyckoff label: {wp} in {doc['material_id']}. Valid labels: {wyckoff_expander.wyckoff_site}"

        # One-hot vector for the element (length 98)
        vec = np.zeros(98, dtype=float)
        atomic_number = pd_elements.symbol(eq_sites[0].species_string).number
        vec[atomic_number - 1] = 1.0
        wp_atom_dist_vec.append(vec)
        species.append(eq_sites[0].species_string)

        # Map Wyckoff letter (a, b, …) to integer (0-based)
        site_ids.append(wyckoff_letter_index(wp))

    # ----------------------------------------------------------------------
    # 4) Convert lists to tensors
    # ----------------------------------------------------------------------
    site_ids = torch.tensor(site_ids, dtype=torch.long)
    wp_atom_dist_vec = torch.tensor(wp_atom_dist_vec, dtype=torch.float32)

    # ----------------------------------------------------------------------
    # 5) Randomly generate a new structure that passes the validity check
    # ----------------------------------------------------------------------
    try:
        structure, wp_coords, _ = generate_valid_structure(
            wp_atom_dist_vec,
            site_ids,
            wyckoff_expander,  # must provide .wyckoff_expand()
            max_trials=1000,
            structure=conv_s,
            r_min=r_min,
        )
    except MaterialsError:
        print(
            f"Failed to generate a valid structure in {num_max_trials} trials with mate-id: {material_id}"
        )
        return None  # skip this entry if generation fails

    # ----------------------------------------------------------------------
    # 6) Derive per-Wyckoff multiplicities and neutrality acceptance
    #    - all-metal: accepted without SMACT
    #    - otherwise: require SMACT neutral solution
    # ----------------------------------------------------------------------
    num_atom_at_WP = np.array([len(eq) for eq in symm_conv.equivalent_sites])
    if not len(num_atom_at_WP) == nWp_site:
        print(
            f"(num_atom_at_WP) length mismatch: expected {nWp_site}, got {len(num_atom_at_WP)}."
        )
        return None

    site_ids_np = site_ids.numpy()
    is_all_metal = is_all_metal_wyckoff_sites(symm_conv)
    mp_wp_oxidation_pattern = None
    if not is_all_metal:
        mp_wp_oxidation_pattern = infer_neutral_wp_oxidation_pattern(
            structure=conv_s,
            symm_conv=symm_conv,
            site_ids=site_ids_np,
            num_atom_at_WP=num_atom_at_WP,
        )
        if mp_wp_oxidation_pattern is None:
            print(
                "Failed to infer neutral WP oxidation pattern from MP structure; "
                f"skip material_id={material_id}."
            )
            return None

    try:
        generated_oxidation_table = generate_sorted_oxidation_states(
            num_atom_at_WP, site_ids_np
        )
    except ValueError as e:
        print(f"Failed to generate oxidation states: {e}")
        return None

    generated_oxidation_table = np.asarray(generated_oxidation_table, dtype=np.int32)
    if generated_oxidation_table.size == 0:
        print(f"No oxidation patterns generated; skip material_id={material_id}.")
        return None
    if generated_oxidation_table.ndim == 1:
        generated_oxidation_table = generated_oxidation_table[None, :]

    if is_all_metal:
        base_pattern = generated_oxidation_table[0]
    else:
        base_pattern = mp_wp_oxidation_pattern.astype(np.int32)

    oxidation_table: list[np.ndarray] = [base_pattern]
    for ox_pattern in generated_oxidation_table:
        if not np.array_equal(ox_pattern, base_pattern):
            oxidation_table.append(ox_pattern)

    num_atom_at_WP_2 = [
        wyckoff_expander.num_equivalent_sites[site_id] for site_id in site_ids
    ]

    assert structure.num_sites == np.sum(num_atom_at_WP) == np.sum(num_atom_at_WP_2)

    # ----------------------------------------------------------------------
    # 7) Assemble the final data dictionary (delegated helper)
    # ----------------------------------------------------------------------
    dict_data = prepare_data(
        structure=structure,
        sample_id=sample_id,
        material_id=material_id,
        oxidation_table=oxidation_table,
        ox_mask_path=ox_mask_path,
        num_atom_at_WP=num_atom_at_WP,
        max_ox_patterns=max_ox_patterns,
        site_ids=site_ids,
        wp_coords=wp_coords,
        saved_dir=saved_dir,
    )

    return dict_data


# In[48]:


def generate_random_Wyckoff_Position_structure_data(
    doc: dict,
    random_sample_id: int,
    wyckoff_expander: WyckoffExpander,
    ox_mask_path: str,
    max_ox_patterns: int,
    WYCKOFF_MAX: int,
    saved_dir: str,
    dist_min: float,
    dist_max: float,
    num_random_WP_trials: int = 50,
    max_trials_in_WP: int = 20,
    num_max_sites: Optional[int] = None,
) -> dict | None:
    """Generate a *fully random* Amm2 structure and data‐dict in one shot.

    The body of this function is a literal wrap of the supplied code block;
    no algorithmic modifications have been made.  Only the surrounding
    function, variable scoping, and docstring are new.

    Parameters
    ----------
    wyckoff_expander : WyckoffExpander
        Helper object that must expose
        ``wyckoff_site``, ``num_equivalent_sites`` and
        ``wyckoff_expand`` (used by :pyfunc:`generate_valid_structure`).
    ox_mask_path : str
        Path to the ``*.npy`` file containing oxidation masks
        (``str(ox) → (N_wp, 98) array``).
    max_ox_patterns : int
        Maximum depth of the oxidation‐pattern dimension for
        :pyfunc:`prepare_data`.
    WYCKOFF_MAX : int
        Upper bound on the number of Wyckoff representative sites to sample.
    saved_dir : str
        Directory where :pyfunc:`prepare_data` saves the generated POSCAR.
    dist_min : float
        Minimum nearest‐neighbor distance (Å) for
    dist_max : float
        Maximum nearest‐neighbor distance (Å) for
    num_random_WP_trials : int, default 10
        Number of attempts to generate a valid structure with random



    Returns
    -------
    dict | None
        A dictionary compatible with ``load_graph`` if structure generation
        succeeds; othKerwise **None** (and the failure reason is printed).

    Notes
    -----
    * All random numbers are drawn with NumPy’s global RNG—set the global
      seed in your calling code if you need reproducibility.
    * The structure is accepted as soon as
      :pyfunc:`generate_valid_structure` returns without error.
    """
    # ----------------------------------------------------------------------
    # 0) Start from the check symmetry of the primitive cell
    # ----------------------------------------------------------------------
    s: Structure = doc["structure"]  # primitive cell
    spa = SpacegroupAnalyzer(s)
    if not spa.get_space_group_number() == wyckoff_expander.spg_number:
        print(
            f"Space group number {spa.get_space_group_number()} does not match expected {wyckoff_expander.spg_number}."
        )
        return None  # skip this entry if space group does not match

    # ------------------------------------------------------------------
    # 1 ) Loop to generate random Wyckoff position specifications
    # ------------------------------------------------------------------
    structure = None
    for _ in tqdm(
        range(num_random_WP_trials), leave=False, desc="num_random_WP_trials"
    ):
        # ------------------------------------------------------------------
        # 1.1) Randomly choose the number of Wyckoff sites and their IDs
        # ------------------------------------------------------------------
        random_num_WP_sites = np.random.randint(2, WYCKOFF_MAX + 1)
        random_site_ids = np.random.choice(
            len(wyckoff_expander.wyckoff_site), size=WYCKOFF_MAX
        )
        random_site_ids = random_site_ids[:random_num_WP_sites]

        num_sites = sum(
            wyckoff_expander.num_equivalent_sites[site_id]
            for site_id in random_site_ids
        )
        if num_max_sites is not None and num_sites >= num_max_sites:
            continue

        # ------------------------------------------------------------------
        # 2) Build a random one‐hot element distribution per site
        # ------------------------------------------------------------------
        random_wp_atom_dist_vec = np.zeros((random_num_WP_sites, 98), dtype=float)
        for i in range(random_num_WP_sites):
            atomic_number = np.random.randint(1, 99)  # 1 to 98 inclusive
            random_wp_atom_dist_vec[i, atomic_number - 1] = 1.0

        # ------------------------------------------------------------------
        # 3) Try to construct a valid structure from the random specification
        # ------------------------------------------------------------------
        try:
            structure, random_wp_coords, _ = generate_valid_structure(
                torch.tensor(random_wp_atom_dist_vec, dtype=torch.float32),
                torch.tensor(random_site_ids, dtype=torch.long),
                wyckoff_expander,  # must provide .wyckoff_expand()
                max_trials=max_trials_in_WP,
                structure=spa.get_conventional_standard_structure(),
                r_min=dist_min,
                r_max=dist_max,
                random_abc=True,
            )
            break
        except MaterialsError:
            continue

    if structure is None:
        return None

    # ------------------------------------------------------------------
    # 4) Derive multiplicities and oxidation patterns
    # ------------------------------------------------------------------
    num_atom_at_WP = [
        wyckoff_expander.num_equivalent_sites[site_id] for site_id in random_site_ids
    ]
    try:
        oxidation_table = generate_sorted_oxidation_states(
            num_atom_at_WP, random_site_ids
        )
    except ValueError as e:
        print(f"Failed to generate oxidation states: {e}")
        return None

    assert structure.num_sites == np.sum(num_atom_at_WP)

    # ------------------------------------------------------------------
    # 5) Pack everything into the expected dictionary
    # ------------------------------------------------------------------
    data_dict = prepare_data(
        structure=structure,
        sample_id="Rand",
        material_id=random_sample_id,
        oxidation_table=oxidation_table,
        ox_mask_path=ox_mask_path,
        num_atom_at_WP=num_atom_at_WP,
        max_ox_patterns=max_ox_patterns,
        site_ids=torch.tensor(random_site_ids, dtype=torch.long),
        wp_coords=torch.tensor(random_wp_coords, dtype=torch.float32),
        saved_dir=saved_dir,
    )

    return data_dict


# In[49]:


def calculate_min_max_nn_distance_and_data_in_space_group(
    wyckoff_expander: WyckoffExpander,
    EHULL_MAX: float = 0.05,
    NSITES_MAX: int = 20,
    WYCKOFF_MAX: int = 6,
):
    """
    Calculate the minimum and maximum nearest-neighbor distance
    among structures in the Materials Project database that match
    the specified space group and filtering criteria.
    Args:
        wyckoff_expander (WyckoffExpander): Wyckoff expander object
            containing space group information.
        EHULL_MAX (float): Maximum energy above hull for filtering structures.
        NSITES_MAX (int): Maximum number of sites for filtering structures.
        WYCKOFF_MAX (int): Maximum number of Wyckoff positions for filtering structures.
    """
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        raise RuntimeError(f"Set {API_KEY_ENV} before fetching Materials Project data.")
    with MPRester(api_key) as mpr:

        docs = mpr.summary.search(
            spacegroup_number=wyckoff_expander.spg_number,  # Amm2
            nsites_max=NSITES_MAX,
            energy_above_hull_max=EHULL_MAX,
        )

    # Filter documents based on the number of Wyckoff positions
    docs = [d for d in docs if get_num_WP(d) <= WYCKOFF_MAX]

    docs = sorted(docs, key=lambda d: (d["nsites"], d["energy_above_hull"]))
    rows = [
        (
            d["material_id"],
            get_num_WP(d),
            d["nsites"],
            d["energy_above_hull"],
            d["volume"],
        )
        for d in docs
    ]
    all_data_rows = [
        (
            d["material_id"],
            get_num_WP(d),
            d["nsites"],
            d["energy_above_hull"],
            d["volume"],
            d,
        )
        for d in docs
    ]

    # material_id をリスト化
    df = pd.DataFrame(
        rows,
        columns=["material_id", "n_WP_sites", "nsites", "energy_above_hull", "volume"],
    )
    print("the number of documents:", len(df))
    display(df.head(10))
    # volume の統計量を表示
    df["density"] = df["nsites"] / df["volume"]
    display(df["density"].describe())

    dist_min, dist_max = compute_min_max_nn_distance(all_data_rows)
    return dist_min, dist_max, all_data_rows, docs


# ## main
#
# Wyckoff Pos: http://webbdcrista2.ehu.es/cryst/get_wp.html

# In[51]:


def initial_data_generation(
    used_space_group: str,
    ox_mask_path: str,
    clear_dir_before_generate: bool = True,
    EHULL_MAX: float = 0.05,
    NSITES_MAX: int = 20,
    WYCKOFF_MAX: int = 6,
    max_ox_patterns: int = 100,
    num_generations_MP_stable: int = 128,
    num_generations_MP_perturbed: int = 128,
    num_generations_MP: int = 128,
    num_generations_random: int = 128,
    output_root_dir: Union[str, Path, None] = None,
):
    """
    Main function to generate initial dataset based on Wyckoff positions
    for a specified space group. The function performs the following steps:
    1. Prepares the output directory.
    2. Initializes the WyckoffExpander for the specified space group.
    3. Calculates minimum and maximum nearest-neighbor distances
         among structures in the Materials Project database.
    4. Generates structures based on Wyckoff positions from the Materials Project.
    5. Generates random structures based on Wyckoff positions.

    Args:
    """

    # ----------------------------------------------------------
    # Define paths and prepare output directory
    # ----------------------------------------------------------
    if output_root_dir is None:
        output_root = (REPO_ROOT / "data" / "raw_data").resolve()
    else:
        output_root = Path(output_root_dir).expanduser().resolve()
    basename = f'initial_MP_based_{used_space_group.replace("/","")}'
    saved_dir = str((output_root / basename).resolve())
    info_csv_path = str((output_root / f"{basename}_info.pkl").resolve())
    if clear_dir_before_generate:
        shutil.rmtree(saved_dir, ignore_errors=True)
        os.makedirs(saved_dir, exist_ok=True)

    # ----------------------------------------------------------
    # Prepare WyckoffExpander for the used space group
    # ----------------------------------------------------------
    if used_space_group == "Amm2":
        sg_wyckoff = Amm2Wyckoff.build()
        spg_number = 38
    elif used_space_group == "P2m":
        sg_wyckoff = P2mWyckoff.build()
        spg_number = 10
    elif used_space_group == "Fbar43m":
        sg_wyckoff = Fbar43mWyckoff.build()
        spg_number = 216
    elif used_space_group == "R3m":
        sg_wyckoff = R3mWyckoff.build()
        spg_number = 160
    elif used_space_group == "P6_3/m":
        sg_wyckoff = P63mWyckoff.build()
        spg_number = 176
    elif used_space_group == "P321":
        sg_wyckoff = P321Wyckoff.build()
        spg_number = 150
    elif used_space_group == "P4/mmm":
        sg_wyckoff = P4mmmWyckoff.build()
        spg_number = 123
    else:
        raise ValueError(f"Unknown space group: {used_space_group}")

    wyckoff_expander = WyckoffExpander(
        WYCKOFF_TABLE=sg_wyckoff.wyckoff_table,
        dup_positions=sg_wyckoff.dup_pos,
        wyckoff_site=sg_wyckoff.wyckoff_site,
        spg_number=spg_number,
        spg_name=used_space_group,
        num_equivalent_sites=sg_wyckoff.num_equivalent_sites,
        device="cpu",
    )

    # ----------------------------------------------------------
    # Calculate min/max nearest-neighbor distances and collect data rows
    # ----------------------------------------------------------
    dist_min, dist_max, all_data_rows, docs = (
        calculate_min_max_nn_distance_and_data_in_space_group(
            wyckoff_expander,
            EHULL_MAX=EHULL_MAX,
            NSITES_MAX=NSITES_MAX,
            WYCKOFF_MAX=WYCKOFF_MAX,
        )
    )

    # ----------------------------------------------------------
    # Generate structures in this order:
    # 0) MP-stable, 1) MP-perturbed, 2) MP-based, 3) random
    # ----------------------------------------------------------
    data_dict_list = []

    # ----------------------------------------------------------
    # 0) Generate MP-based stable structures without any noise
    # ----------------------------------------------------------
    MP_stable_sample_id = 0
    material_ids = np.random.choice(
        len(all_data_rows), size=num_generations_MP_stable * 4
    )
    srt = time.time()
    for mate_id in tqdm(material_ids, desc="Generating MP-Stable structures"):
        row = all_data_rows[mate_id]
        material_id, nWp_site, nsites, ehull, volume, doc = row

        data_dict = generate_MP_Wyckoff_Position_stable_structure_data(
            doc=doc,
            wyckoff_expander=wyckoff_expander,
            sample_id=MP_stable_sample_id,
            material_id=material_id,
            ox_mask_path=ox_mask_path,
            max_ox_patterns=max_ox_patterns,
            nWp_site=nWp_site,
            saved_dir=saved_dir,
        )
        if data_dict is not None:
            MP_stable_sample_id += 1
            data_dict_list.append(data_dict)

        if MP_stable_sample_id % 25 == 0 and MP_stable_sample_id > 0:
            print(f"Generated MP-stable {MP_stable_sample_id} structures so far...")
            esp_time = time.time() - srt
            print(
                "estimated total (min):",
                esp_time / MP_stable_sample_id * num_generations_MP_stable / 60,
            )

        if MP_stable_sample_id >= num_generations_MP_stable:
            break

    clear_output()

    # ----------------------------------------------------------
    # 1) Generate MP-based perturbed-lattice structures
    # ----------------------------------------------------------
    MP_perturbed_sample_id = 0
    material_ids = np.random.choice(
        len(all_data_rows), size=num_generations_MP_perturbed * 4
    )
    srt = time.time()
    for mate_id in tqdm(material_ids, desc="Generating MP-Perturbed WP structures"):
        row = all_data_rows[mate_id]
        material_id, nWp_site, nsites, ehull, volume, doc = row

        data_dict = generate_MP_Wyckoff_Position_perturbed_lattice_structure_data(
            doc=doc,
            wyckoff_expander=wyckoff_expander,
            sample_id=MP_perturbed_sample_id,
            material_id=material_id,
            ox_mask_path=ox_mask_path,
            max_ox_patterns=max_ox_patterns,
            nWp_site=nWp_site,
            saved_dir=saved_dir,
            r_min=dist_min,
            r_max=dist_max,
        )
        if data_dict is not None:
            MP_perturbed_sample_id += 1
            data_dict_list.append(data_dict)

        if MP_perturbed_sample_id % 25 == 0 and MP_perturbed_sample_id > 0:
            print(
                f"Generated MP-perturbed {MP_perturbed_sample_id} structures so far..."
            )
            esp_time = time.time() - srt
            print(
                "estimated total (min):",
                esp_time / MP_perturbed_sample_id * num_generations_MP_perturbed / 60,
            )

        if MP_perturbed_sample_id >= num_generations_MP_perturbed:
            break

    clear_output()

    # ----------------------------------------------------------
    # 2) Generate MP-based Wyckoff position structures
    # ----------------------------------------------------------
    atomic_density_list = []
    material_ids = np.random.choice(len(all_data_rows), size=num_generations_MP * 4)
    MP_sample_id = 0
    srt = time.time()
    for mate_id in tqdm(material_ids, desc="Generating MP-Based WP structures"):
        row = all_data_rows[mate_id]
        material_id, nWp_site, nsites, ehull, volume, doc = row

        data_dict = generate_MP_Wyckoff_Position_structure_data(
            doc=doc,
            wyckoff_expander=wyckoff_expander,
            sample_id=MP_sample_id,
            material_id=material_id,
            ox_mask_path=ox_mask_path,
            max_ox_patterns=max_ox_patterns,
            nWp_site=nWp_site,
            saved_dir=saved_dir,
        )
        if data_dict is not None:
            MP_sample_id += 1
            atomic_density_list.append(data_dict["atomic_density"])
            data_dict_list.append(data_dict)

        if MP_sample_id % 25 == 0 and MP_sample_id > 0:
            print(f"Generated MP-based {MP_sample_id} structures so far...")
            esp_time = time.time() - srt
            print(
                "estimated total (min):",
                esp_time / MP_sample_id * num_generations_MP / 60,
            )

        if MP_sample_id >= num_generations_MP:
            break

    clear_output()
    display(pd.DataFrame(atomic_density_list, columns=["atomic_density"]).describe())

    # ----------------------------------------------------------
    # 3) Generate random Wyckoff position structures
    # ----------------------------------------------------------
    random_sample_id = 0
    srt = time.time()
    for _ in tqdm(
        range(num_generations_random * 4), desc="Generating random WP structures"
    ):
        doc = random.choice(docs)
        data_dict = generate_random_Wyckoff_Position_structure_data(
            doc=doc,
            random_sample_id=random_sample_id,
            wyckoff_expander=wyckoff_expander,
            ox_mask_path=ox_mask_path,
            max_ox_patterns=max_ox_patterns,
            WYCKOFF_MAX=WYCKOFF_MAX,
            saved_dir=saved_dir,
            dist_min=dist_min,
            dist_max=dist_max,
            num_random_WP_trials=50,
            max_trials_in_WP=20,
            num_max_sites=NSITES_MAX,
        )
        if data_dict is not None:
            random_sample_id += 1
            data_dict_list.append(data_dict)
        if random_sample_id % 10 == 0 and random_sample_id > 0:
            clear_output()
            print(f"Generated {random_sample_id} random structures so far...")
            esp_time = time.time() - srt
            print(
                "estimated total (min):",
                round(esp_time / random_sample_id * num_generations_random / 60, 2),
            )
            print(
                "remaining time (min):",
                round(
                    esp_time
                    / random_sample_id
                    * (num_generations_random - random_sample_id)
                    / 60,
                    3,
                ),
            )
        if random_sample_id >= num_generations_random:
            break

    clear_output()

    # ----------------------------------------------------------
    # Final checks and save info dataframe
    # ----------------------------------------------------------
    if (
        MP_stable_sample_id == num_generations_MP_stable
        and random_sample_id == num_generations_random
        and MP_sample_id == num_generations_MP
        and MP_perturbed_sample_id == num_generations_MP_perturbed
    ):
        print(
            f"Successfully generated total "
            f"{MP_stable_sample_id + random_sample_id + MP_sample_id + MP_perturbed_sample_id} samples "
            f"({MP_stable_sample_id} MP-stable, {MP_perturbed_sample_id} MP-perturbed, "
            f"{MP_sample_id} MP structures and "
            f"{random_sample_id} random structures)."
        )
    else:
        raise ValueError(
            f"Generation incomplete: {MP_stable_sample_id} MP-stable, "
            f"{MP_perturbed_sample_id} MP-perturbed and {MP_sample_id} MP structures and "
            f"{random_sample_id} random structures."
        )

    # check space group of generated structures
    import glob

    for pos_path in glob.glob(os.path.join(saved_dir, "*.vasp")):
        s = Structure.from_file(pos_path)
        spg_number = SpacegroupAnalyzer(s).get_space_group_number()
        assert (
            spg_number == wyckoff_expander.spg_number
        ), f"Space group number {spg_number} does not match expected {wyckoff_expander.spg_number}."

    df = pd.DataFrame(data_dict_list)
    df["nsites"] = df["elements"].apply(lambda x: len(x))

    # df.to_csv(info_csv_path, index=False)
    df.to_pickle(info_csv_path, protocol=4)


# In[52]:


target_space_groups = [
    "R3m",
    "Amm2",
    "P4/mmm",
    "P2m",
    "Amm2",
    "P6_3/m",
    "P321",
    "Fbar43m",
]
# target_space_groups = [ "P4/mmm", 'P6_3/m']
ox_mask_path = os.environ.get(
    "SMOACS_OX_MASK_PATH",
    str(REPO_ROOT / "data" / "intermidiate" / "mask_dict_super_common.npz"),
)


def main() -> None:
    for spg in target_space_groups:
        print(f"Starting data generation for space group: {spg}")
        initial_data_generation(
            used_space_group=spg,
            ox_mask_path=ox_mask_path,
        )
        print(f"Completed data generation for space group: {spg}\n")


if __name__ == "__main__":
    main()
    raise SystemExit(0)
