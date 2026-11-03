import os
from typing import Any, Dict, List, Optional, Tuple
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from jarvis.core.atoms import Atoms
from torch.utils.data import DataLoader
from periodictable import elements as pd_elements

from src.graph.graph_utils import get_torch_dataset, graph_to_tensors
from src.utils.oxidation import get_atom_masks_for_oxidation_states
from src.utils.scalers import ABC_Scaler, Angle_Scaler
from src.utils.structure_paths import get_structure_paths
from src.spg_wp.wyckoff_expander import WyckoffExpander
from src.utils.coord import compute_lattice_vectors
from src.preprocess.prepro_utils import compute_normalized_dist_and_ox_mask


GRAPH_GEOMETRY_KEYS = ("lattice_mat", "coords", "abc", "angles")


def _valid_graph_geometry(geometry):
    coords = np.asarray(geometry["coords"])
    if coords.ndim != 2 or coords.shape[1] != 3:
        return False
    if any(
        np.shape(geometry[key]) != shape
        for key, shape in (("lattice_mat", (3, 3)), ("abc", (3,)), ("angles", (3,)))
    ):
        return False
    if not all(np.isfinite(geometry[key]).all() for key in GRAPH_GEOMETRY_KEYS):
        return False
    determinant = np.linalg.det(np.asarray(geometry["lattice_mat"], dtype=np.float64))
    return np.isfinite(determinant) and abs(determinant) > 0


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Detach a tensor, move it to CPU, and convert it to a NumPy array.

    Args:
        tensor (torch.Tensor): Input tensor.

    Returns:
        np.ndarray: CPU-based NumPy array detached from the computation graph.
    """
    return tensor.detach().cpu().numpy()


def fill_nan_by_key(arr: np.ndarray, key: str) -> np.ndarray:
    """Fill NaN values in an array according to a predefined rule for *key*.

    Filling policy
    --------------
    * ``'batch_abc'``        → uniform random in [2, 10]
    * ``'batch_angle'``      → 90
    * ``'batch_dir_coords'`` → uniform random in [0, 1]
    * ``'normalized_dist'``  → 0
    * other keys             → raise ``ValueError`` if NaNs exist

    Args:
        arr (np.ndarray): Array that may contain NaNs.
        key (str): Identifier that determines the filling rule.

    Returns:
        np.ndarray: Array with NaNs filled.

    Raises:
        ValueError: If NaNs are present for an unsupported *key*.
    """
    if not np.isnan(arr).any():
        return arr

    if key == "batch_abc":
        fill = np.random.uniform(2.0, 10.0, size=arr.shape)
    elif key == "batch_angle":
        fill = 90.0
    elif key == "batch_dir_coords":
        fill = np.random.uniform(0.0, 1.0, size=arr.shape)
    elif key == "normalized_dist":
        fill = 0.0
    elif key == 'site_ids':
        return arr
    else:
        raise ValueError(f"NaN values detected in unsupported key '{key}'.")

    return np.where(np.isnan(arr), fill, arr)


def to_numpy_fill_nan(tensor_dict: Dict[str, torch.Tensor], key: str) -> np.ndarray:
    """Convert ``tensor_dict[key]`` to NumPy and fill NaNs based on *key*.

    Args:
        tensor_dict (dict[str, torch.Tensor]): Dictionary containing tensors.
        key (str): Target key whose tensor will be processed.

    Returns:
        np.ndarray: NumPy array with NaNs handled per the rule for *key*.

    Raises:
        KeyError: If *key* is absent in *tensor_dict*.
        ValueError: If NaNs remain for an unsupported *key*.
    """
    if key not in tensor_dict:
        raise KeyError(f"Key '{key}' not found in tensor_dict.")

    arr = tensor_to_numpy(tensor_dict[key])
    return fill_nan_by_key(arr, key)


def build_dataset_for_graph(
    optimized_mini_batch_inputs_dict: Dict[str, torch.Tensor],
    wyckoff_expander: Optional[WyckoffExpander],
    onehot_temperature: float = 1E-8,
) -> List[Dict[str, Any]]:
    """
    Construct a dataset for graph generation from the optimized mini-batch inputs.

    Args:
        optimized_mini_batch_inputs_dict (Dict[str, torch.Tensor]):
            Dictionary containing crystallographic and atomic data for multiple structures.
        onehot_temperature (float, optional): Temperature parameter for softmax normalization.
    Returns:
        List[Dict[str, Any]]: A list of graph input dictionaries for each structure.
    """
    # Extract arrays using a utility function
    if wyckoff_expander is None:
        # Keep invalid geometry visible so the entire candidate can be restored.
        wp_dir_coords = tensor_to_numpy(
            optimized_mini_batch_inputs_dict["batch_dir_coords"]
        )
        abc = tensor_to_numpy(optimized_mini_batch_inputs_dict["batch_abc"])
        angle = tensor_to_numpy(optimized_mini_batch_inputs_dict["batch_angle"])
    else:
        wp_dir_coords = to_numpy_fill_nan(
            optimized_mini_batch_inputs_dict, "batch_dir_coords"
        )
        abc = to_numpy_fill_nan(optimized_mini_batch_inputs_dict, "batch_abc")
        angle = to_numpy_fill_nan(optimized_mini_batch_inputs_dict, "batch_angle")
    lattice_vectors = compute_lattice_vectors(torch.tensor(abc), torch.tensor(angle))
    lattice_vectors = lattice_vectors.detach().cpu().numpy()
    size = to_numpy_fill_nan(optimized_mini_batch_inputs_dict, 'size')
    atomic_mask = to_numpy_fill_nan(optimized_mini_batch_inputs_dict, 'atomic_mask')
    radii_tensor = to_numpy_fill_nan(optimized_mini_batch_inputs_dict, 'radii_tensor')
    ox_states_used_mask = to_numpy_fill_nan(optimized_mini_batch_inputs_dict, 'ox_states_used_mask')
    site_ids = to_numpy_fill_nan(optimized_mini_batch_inputs_dict, 'site_ids')


    # Compute normalized atomic distribution and sharpened oxidation state mask.
    wp_normalized_dist, _ = compute_normalized_dist_and_ox_mask(
        atomic_distribution=optimized_mini_batch_inputs_dict['atomic_distribution'],
        atomic_mask=optimized_mini_batch_inputs_dict['atomic_mask'],
        ox_states_used_mask=optimized_mini_batch_inputs_dict['ox_states_used_mask'],
        temperature=onehot_temperature,
        size=optimized_mini_batch_inputs_dict['wp_size'],
        ox_mask_learnable_tensor_per_crystal=optimized_mini_batch_inputs_dict['ox_mask_learnable_tensor_per_crystal'],  # can be None
    )
    if wyckoff_expander is not None:
        expanded_batch_dir_coords, expanded_normalized_dist = wyckoff_expander.wyckoff_expand(
            wp_coords=torch.tensor(wp_dir_coords).to(wyckoff_expander.device),
            wp_atom_dist=torch.tensor(wp_normalized_dist).to(wyckoff_expander.device),
            site_ids=torch.tensor(site_ids).to(wyckoff_expander.device),
        )
        expanded_batch_dir_coords = expanded_batch_dir_coords.detach().cpu().numpy()
        print("DEBUG!!!")
        print(f"wp_coords = torch.{torch.tensor(wp_dir_coords)}")
        print(f"wp_atom_dist = torch.{torch.tensor(wp_normalized_dist)}")
        print(f"site_ids = torch.{torch.tensor(site_ids)}")


    else:
        expanded_batch_dir_coords = wp_dir_coords
        expanded_normalized_dist = wp_normalized_dist


    optimized_mini_batch_inputs_dict['normalized_dist'] = expanded_normalized_dist
    expanded_normalizesd_dist = to_numpy_fill_nan(optimized_mini_batch_inputs_dict, 'normalized_dist')
    elements_list = np.array([pd_elements[idx+1].symbol for idx in np.argmax(expanded_normalizesd_dist,axis=1)])

    # crystal_ids for each atom in crystal
    atom_crystal_ids = np.concatenate([
        np.full(cry_s, cry_i, dtype=int) for cry_i, cry_s in enumerate(size)
    ])
    # crystal_ids in the Wyckoff positions
    if wyckoff_expander is not None:
        wp_size = optimized_mini_batch_inputs_dict['wp_size'].detach().cpu().numpy()
        wp_atom_crystal_ids = np.concatenate([
            np.full(wp_s, cry_i, dtype=int) for cry_i, wp_s in enumerate(wp_size)
        ])
    else:
        wp_size = size
        wp_atom_crystal_ids = atom_crystal_ids
        
    print("[DEBUG] sum(size)=", int(size.sum()),
        "sum(wp_size)=", int(wp_size.sum()),
        "expanded_atoms=", int(expanded_batch_dir_coords.shape[0]),
        "expanded_dist=", int(expanded_normalized_dist.shape[0]))
    

    dataset_for_graph = []

    for cry_i in range(len(size)):

        cry_bool = atom_crystal_ids == cry_i
        wp_cry_bool = wp_atom_crystal_ids == cry_i
        num_atoms = size[cry_i]
        num_wp_atoms = wp_size[cry_i] 
        dummy_elements = elements_list[cry_bool].tolist()
        coords = expanded_batch_dir_coords[cry_bool]

        atom_mask = atomic_mask[wp_cry_bool]
        radii = radii_tensor[wp_cry_bool]
        site_id_tensor = torch.tensor(site_ids[wp_cry_bool])
        ox_mask_tensor = torch.tensor(ox_states_used_mask[wp_cry_bool][0])


        # Asserts with explanation
        assert len(atom_mask) == num_wp_atoms, f"atom_mask size mismatch for crystal {cry_i}"
        assert len(radii) == num_wp_atoms, f"radii size mismatch for crystal {cry_i}"
        assert len(site_id_tensor) == num_wp_atoms, f"site_ids size mismatch for crystal {cry_i}"
        assert len(dummy_elements) == num_atoms, f"elements size mismatch for crystal {cry_i}"


        data_dict = {
            'atoms': {
                'lattice_mat': lattice_vectors[cry_i],
                'abc': abc[cry_i],
                'angles': angle[cry_i],
                'coords': coords,
                'elements': dummy_elements,
                'props': [''] * num_atoms,
                'cartesian': False
            },
            'dummy_target': 0,
            'jid': optimized_mini_batch_inputs_dict['fnames'][cry_i],
            'atom_mask_for_all_ox_states': atom_mask,
            'radii_for_all_ox_states': radii,
            'elements': dummy_elements,
            'site_ids': site_id_tensor.detach().cpu().numpy(),
            'ox_states_used_mask': ox_mask_tensor.detach().cpu().numpy(),
        }

        if wyckoff_expander is None and not _valid_graph_geometry(data_dict["atoms"]):
            refs = optimized_mini_batch_inputs_dict.get("graph_fallback_geometry")
            jid = data_dict["jid"]
            if refs is None or len(refs) != len(size):
                raise ValueError(
                    f"{jid}: missing or incomplete graph fallback geometry"
                )
            ref = refs[cry_i]
            if not _valid_graph_geometry(ref) or np.shape(ref["coords"]) != (
                int(num_atoms),
                3,
            ):
                raise ValueError(
                    f"{jid}: invalid graph fallback geometry: expected_atoms={num_atoms}, "
                    f"reference_coords_shape={np.shape(ref['coords'])}"
                )
            for key in GRAPH_GEOMETRY_KEYS:
                data_dict["atoms"][key] = np.array(ref[key], copy=True)
            print(
                f"graph geometry restored: {jid}, source={ref['source_jid']}, "
                f"atoms={num_atoms}, reason=nonfinite_or_degenerate_geometry"
            )

        if wyckoff_expander is not None:
            data_dict['wp_coords'] = wp_dir_coords[wp_cry_bool]
            data_dict['test_coords'] = coords # debug
            ex_coords, _ = wyckoff_expander.wyckoff_expand( # debug
                wp_coords=torch.tensor(wp_dir_coords[wp_cry_bool]).to(wyckoff_expander.device),
                wp_atom_dist=torch.tensor(wp_normalized_dist[wp_cry_bool]).to(wyckoff_expander.device),
                site_ids=torch.tensor(site_ids[wp_cry_bool]).to(wyckoff_expander.device),
            )
            data_dict['ex_coords'] = ex_coords # debug

            if ex_coords.shape[0] != size[cry_i]:
                print(f"DEBUG [MISMATCH] cry={cry_i} size={size[cry_i]} expanded={ex_coords.shape[0]} wp_size={wp_size[cry_i]}")

        dataset_for_graph.append(data_dict)

    return dataset_for_graph


def create_ALIGNN_init_structure_from_directory(
    dir_path:str,
    intermid_dir_path:str,
    max_ox_states:int,
    mask_method :str,
    mask_dict_path: Optional[str] = None,
    crystal_system:Optional[str] = None,
    num_candidate:Optional[int] = None,
    graph_update_mode:bool = False,
    ):

    print("<info> load initial structures from", dir_path)
    if crystal_system is None:
        print("<info> crystal_system is None")
        if mask_dict_path is not None:
            mask_d = dict(np.load(mask_dict_path, allow_pickle=True))
        else:
            mask_d = dict(
                np.load(
                    os.path.join(intermid_dir_path, f"mask_dict_{mask_method}.npz"),
                    allow_pickle=True,
                )
            )
        radii_d = None
    elif crystal_system == "perovskite":
        if mask_dict_path is not None:
            raise ValueError(
                "mask_dict_path is not supported for perovskite mode. "
                "Use the default ionic masks/radii instead."
            )
        print("<info> crystal_system is perovskite")
        mask_d = dict(np.load(os.path.join(intermid_dir_path, f"ionic_mask_dict_{mask_method}.npz"), allow_pickle=True))
        radii_d = dict(np.load(os.path.join(intermid_dir_path, f"ionic_radii_dict_{mask_method}.npz"), allow_pickle=True))

    dataset_optimize = []
    num_total_crystals = 0
    file_paths = get_structure_paths(dir_path)
    for path in file_paths:
        atoms = Atoms.from_poscar(path)

        atom_mask_for_all_ox_states, radii_for_all_ox_states, ox_states_used_mask, site_ids = get_atom_masks_for_oxidation_states(
            atoms=atoms, 
            mask_d=mask_d, 
            radii_d=radii_d,
            max_ox_states=max_ox_states,
            crystal_system=crystal_system,
            graph_update_mode=graph_update_mode,
            mask_name=Path(mask_dict_path).name if mask_dict_path else None,
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

    if num_candidate is not None:
        dataset_optimize = dataset_optimize[:num_candidate]

    num_total_crystals = len(dataset_optimize)
    return dataset_optimize, num_total_crystals


def _is_pkl_path(path: str) -> bool:
    return Path(path).is_file() and Path(path).suffix == ".pkl"


def create_ALIGNN_init_structure_from_pkl(
    pkl_path: str,
    num_candidate: Optional[int] = None,
) -> List[Dict[str, Any]]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected list in pkl file: {pkl_path}")

    dataset_optimize = []
    required_keys = {
        "coords_frac",
        "lattice_matrix",
        "abc",
        "angles",
        "elements",
        "atom_mask_for_all_ox_states",
        "radii_for_all_ox_states",
        "ox_states_used_mask",
    }
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Expected dict item in pkl at index {idx}: {pkl_path}")
        missing = required_keys - set(item.keys())
        if missing:
            raise ValueError(
                f"Missing keys in pkl item {idx} ({pkl_path}): {sorted(missing)}"
            )
        elements = item["elements"]
        num_atoms = len(elements)
        site_ids = np.array([np.nan] * num_atoms)
        jid = item.get("jid") or item.get("material_id") or f"sample_{idx}"

        dataset_optimize.append(
            {
                "jid": jid,
                "atoms": {
                    "lattice_mat": np.asarray(item["lattice_matrix"]),
                    "coords": np.asarray(item["coords_frac"]),
                    "elements": elements,
                    "abc": np.asarray(item["abc"]),
                    "angles": np.asarray(item["angles"]),
                    "props": [""] * num_atoms,
                    "cartesian": False,
                },
                "dummy_target": 0,
                "elements": elements,
                "atom_mask_for_all_ox_states": np.asarray(
                    item["atom_mask_for_all_ox_states"]
                ),
                "radii_for_all_ox_states": np.asarray(
                    item["radii_for_all_ox_states"]
                ),
                "ox_states_used_mask": np.asarray(item["ox_states_used_mask"]),
                "site_ids": site_ids,
            }
        )

    if num_candidate is not None:
        dataset_optimize = dataset_optimize[:num_candidate]

    return dataset_optimize


def load_ALIGNN_initial_data(
        settings_dict:Dict[str,Any],
        dir_path:str,
        intermid_dir_path:str,
        mask_dict_path: Optional[str],
        batch_size:int,
        use_atomic_mask:bool,
        num_workers:int,
        max_ox_states:int,
        mask_method:str,
        crystal_system:Optional[str],
        angle_range:Tuple[float,float],
        abc_range:Tuple[float,float],
        wyckoff_expander:Optional[WyckoffExpander],
        device: str = "cuda",
        num_candidate:Optional[int] = None,
        max_neighbors:int = 12,
        cutoff:float = 8.0,
        cutoff_extra:float = 3.0,
        base_lattice_index:int = 0,
        test_mode:bool = True,
        graph_update_mode:bool = False,
        input_dataframe:Optional[pd.DataFrame] = None,
        intermid_data_dict:Optional[Dict[str, Any]] = None,
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
        num_workers (int): Number of worker processes for DataLoader.
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
        intermid_data_dict (Optional[Dict[str, Any]]): Intermediate data dictionary for precomputed values (if available).

    Returns:
        List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]: A list of ALIGNN-compatible batches,
        each consisting of (input_tensor_dict, scaler_dict).
    """

    if intermid_data_dict is None and input_dataframe is None:
        if _is_pkl_path(dir_path):
            dataset_optimize = create_ALIGNN_init_structure_from_pkl(
                pkl_path=dir_path,
                num_candidate=num_candidate,
            )
        else:
            dataset_optimize, _ = create_ALIGNN_init_structure_from_directory(
                dir_path=dir_path,
                crystal_system=crystal_system,
                intermid_dir_path=intermid_dir_path,
                max_ox_states=max_ox_states,
                num_candidate=num_candidate,
                mask_method=mask_method,
                mask_dict_path=mask_dict_path,
                graph_update_mode = graph_update_mode,
            )
    elif intermid_data_dict is None and input_dataframe is not None:
        dataset_optimize = (
            input_dataframe[:num_candidate] if num_candidate is not None else input_dataframe
        )
    elif (intermid_data_dict is not None) and graph_update_mode:
        dataset_optimize = build_dataset_for_graph(
            intermid_data_dict,
            wyckoff_expander,
        )
    else:
        raise Exception("intermid_data_dict is not None, but graph_update_mode is False. This is not supported yet.")

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

    num_workers = int(os.getenv("SMOACS_DATALOADER_WORKERS", "4"))
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=train_data.collate_line_graph,
        drop_last=False,
        num_workers=num_workers,
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
            'atom_features':atom_features,
            'bondlength':bondlength,
            'angle_features':angle_features,
        }

        alignn_minibatch_dict = graph_to_tensors(
            g=g, 
            lg=lg, 
            train_df=train_df.loc[i*batch_size:(i+1)*batch_size-1], 
            #data_dict_for_test=data_dict_for_test, 
            use_atomic_mask=use_atomic_mask,
            #test_mode=test_mode
            wyckoff_expander = wyckoff_expander
        )

        if not graph_update_mode and wyckoff_expander is None:
            batch_df = train_df.iloc[i * batch_size : (i + 1) * batch_size]
            refs = []
            for crystal_i, (_, row) in enumerate(batch_df.iterrows()):
                ref = {
                    "source_jid": row["jid"],
                    **{
                        key: np.array(row["atoms"][key], copy=True)
                        for key in GRAPH_GEOMETRY_KEYS
                    },
                }
                # Atom masks use the original site order; normalize only the coordinate basis.
                ref["coords"] = np.array(
                    Atoms.from_dict(row["atoms"]).frac_coords, copy=True
                )
                num_atoms = int(alignn_minibatch_dict["size"][crystal_i])
                if not _valid_graph_geometry(ref) or ref["coords"].shape != (
                    num_atoms,
                    3,
                ):
                    raise ValueError(
                        f"{row['jid']}: invalid initial graph geometry: expected_atoms={num_atoms}, "
                        f"reference_coords_shape={ref['coords'].shape}"
                    )
                refs.append(ref)
            alignn_minibatch_dict["graph_fallback_geometry"] = refs

        scalers = {
            "abc_scaler": ABC_Scaler(
                init_batch_abc=alignn_minibatch_dict["batch_abc"],
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
        # if i*batch_size>=num_total:
        #    break

    return ALIGNN_all_inputs
