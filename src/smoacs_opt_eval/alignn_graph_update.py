from typing import List, Dict, Any, Union
import numpy as npde
import torch
from periodictable import elements as pd_elements
import dgl


from typing import Dict, Optional, Tuple, List, Any, Union
import numpy as np
import torch
from src.spg_wp.wyckoff_expander import WyckoffExpander

from src.utils.coord import compute_lattice_vectors
from src.data_loader.alignn_loader import load_ALIGNN_initial_data
from src.preprocess.alignn_geometry import prepare_alignn_geometry


def calculate_update_steps(num_steps: int, num_g_upd: int) -> List[int]:
    """Calculates the step indices where the sine values first exceed the specified thresholds.

    This function computes the sine value for a sequence of steps and finds the first step
    where the sine value exceeds each of a set of decreasing thresholds.

    Args:
        num_steps (int): The total number of steps to compute sine values for. Default is 200.
        num_g_upd (int): The number of threshold values to compute, which are evenly spaced and decreasing. Default is 10.

    Returns:
        List[int]: A list of indices indicating the first step where the sine value exceeds each threshold.

    Raises:
        ValueError: If `num_g_upd` is zero or negative, which would lead to division by zero.
    """
    if num_g_upd <= 0:
        raise ValueError("num_g_upd must be greater than 1 to avoid division by zero.")
        
    # Generate the step numbers
    steps = np.arange(num_steps)
    
    # Calculate the sine values
    sin_val_steps = np.sin(np.pi / 2 * steps/num_steps)
    
    # Calculate update thresholds
    update_thres = np.linspace(0, 1, num_g_upd+1, endpoint=False)[1:]
    
    # Find the first step that exceeds each threshold
    update_steps = []
    for thres in update_thres:
        idx = np.argmax(sin_val_steps > thres)
        update_steps.append(idx if sin_val_steps[idx] > thres else -1)  # -1 if no step exceeds the threshold
    
    return update_steps


def matching_by_hash_table(
    a:np.ndarray, 
    b:np.ndarray
)->List[int]:

    """
    Find matching row indices between two 2D arrays.

    Args:
        a (np.ndarray): Reference array of shape (M, D), where each row is a key.
        b (np.ndarray): Query array of shape (N, D) to match against `a`.

    Returns:
        List[int]: Indices in `a` corresponding to each row in `b`.

    Raises:
        KeyError: If any row in `b` is not found in `a`.
    """
    # ハッシュテーブルを作成
    hash_map = {tuple(row): idx for idx, row in enumerate(a)}

    # bの各行に対して対応するaのインデックスを見つける
    matching_indices = [hash_map[tuple(row)] for row in b]
    return matching_indices


def graph_update(
        intermid_data_dict: Dict, 
        num_batch_crystal: int,
        old_g: dgl.DGLGraph,
        old_lg: dgl.DGLGraph,
        wyckoff_expander: Optional[WyckoffExpander],
        num_workers: int = 4,
        device: Optional[Union[str, torch.device]] = None,
        ) -> None:
    """Update graph and lattice tensors in `intermid_data_dict` using ALIGNN preprocessing.

    This function calls `load_ALIGNN_initial_data` in `graph_update_mode` to refresh
    graph information (`g`, `lg`) while checking that the core structure-related
    tensors (`batch_dir_coords`, `batch_abc`, `batch_angle`) remain consistent
    within a numerical tolerance.

    Args:
        intermid_data_dict (dict): Dictionary containing intermediate tensor data,
            which will be updated in-place with new graph info.
        num_batch_crystal (int): Batch size for the input data.
        old_g (dgl.DGLGraph): Original graph object representing the crystal structure.
        old_lg (dgl.DGLGraph): Original line graph object representing the crystal structure.
        device (Optional[Union[str, torch.device]]): Device for preprocessing and scalers.

    Raises:
        AssertionError: If any structural tensor deviates from the original
            beyond a specified tolerance (1e-4).
    """

    if device is None:
        device = "cuda"
    device_str = str(device)

    ALIGNN_input_data = load_ALIGNN_initial_data(
        settings_dict={},  # dummy
        dir_path='dummy',  # dummy
        intermid_dir_path='dummy',  # dummy
        mask_dict_path=None,  # dummy
        batch_size=num_batch_crystal,
        use_atomic_mask=True,
        num_workers=num_workers,
        max_ox_states=-1,  # dummy
        mask_method='dummy',  # dummy
        crystal_system='dummy',  # dummy
        angle_range=(np.nan, np.nan),  # dummy
        abc_range=(np.nan, np.nan),  # dummy
        graph_update_mode=True,
        intermid_data_dict=intermid_data_dict,
        wyckoff_expander=wyckoff_expander,
        device=device_str,
    )
    assert len(ALIGNN_input_data) == 1

    new_g = ALIGNN_input_data[0][0]['g']
    new_lg = ALIGNN_input_data[0][0]['lg']

    
    # Get old graph info
    old_num_nodes = old_g.num_nodes()
    old_num_edges = old_g.num_edges()
    old_lg_edges = old_lg.num_edges()
    new_num_nodes = new_g.num_nodes()
    new_num_edges = new_g.num_edges()
    new_lg_edges = new_lg.num_edges()
    print(f"[Graph Update] Nodes: {old_num_nodes} → {new_num_nodes}")
    print(f"[Graph Update] Edges (g): {old_num_edges} → {new_num_edges}")
    print(f"[Graph Update] Edges (lg): {old_lg_edges} → {new_lg_edges}")


    # Replace old graph objects
    intermid_data_dict['g'] = new_g
    intermid_data_dict['lg'] = new_lg

    for key in [
            'batch_displace',
            'batch_dst_ids',
            'batch_src_ids',
            'z_dst_ids',
            'z_src_ids',
            'num_edges',
        ]:
        data = ALIGNN_input_data[0][0][key]
        intermid_data_dict[key] = data.detach().to(data.device)

    return intermid_data_dict


def graph_parameter_update(
    optimization_target_names: List[str],
    optimization_targets: List[torch.Tensor],
    mini_batch_inputs_dict: Dict[str, Union[torch.Tensor, str]],
    output_dict: Dict[str, torch.Tensor],
    num_batch_crystal: int,
    wyckoff_expander: Optional[WyckoffExpander],
    num_workers: int = 4,
    *,
    scalers: Optional[Dict[str, Any]] = None,
    limit_coords_displacement: Optional[float] = None,
) -> Dict[str, Union[torch.Tensor, str]]:
    """
    Update graph-related parameters in the mini-batch dictionary after preparing intermediate tensors.

    This function prepares intermediate tensors for optimization, updates the graph using `graph_update`,
    and refreshes the graph-related entries in `mini_batch_inputs_dict`.

    Args:
        optimization_target_names (List[str]): Names of optimization targets.
        optimization_targets (List[torch.Tensor]): Tensors to be optimized.
        mini_batch_inputs_dict (Dict[str, Union[torch.Tensor, str]]): Dictionary containing input data.
        output_dict (Dict[str, torch.Tensor]): Output dictionary from the model containing new geometry.
        num_batch_crystal (int): Number of crystals in the batch.
        graph_update (Callable): Function used to update graph-related data.

    Returns:
        Dict[str, Union[torch.Tensor, str]]: Updated mini_batch_inputs_dict with new graph parameters.
    """

    intermid_data_dict = {}
    device = optimization_targets[0].device
    # Copy optimization targets to device
    for target_name, target_tensor in zip(
        optimization_target_names, optimization_targets
    ):
        intermid_data_dict[target_name] = target_tensor.detach().clone().to(device)

    intermid_data_dict["atomic_mask"] = mini_batch_inputs_dict["atomic_mask"].to(device)
    intermid_data_dict['size'] = mini_batch_inputs_dict["size"].to(device)
    intermid_data_dict['radii_tensor'] = mini_batch_inputs_dict["radii_tensor"].to(device)
    intermid_data_dict['site_ids'] = mini_batch_inputs_dict["site_ids"].to(device)
    intermid_data_dict['ox_states_used_mask'] = mini_batch_inputs_dict["ox_states_used_mask"].to(device)
    intermid_data_dict['fnames'] = mini_batch_inputs_dict["fnames"]
    if wyckoff_expander is None:
        if scalers is None:
            raise ValueError("Non-Wyckoff graph updates require the current scalers")
        batch_abc, batch_angle, _, coords_geometry, _ = prepare_alignn_geometry(
            intermid_data_dict["scaled_batch_abc"],
            intermid_data_dict["scaled_batch_angle"],
            intermid_data_dict["batch_dir_coords"],
            abc_scaler=scalers["abc_scaler"],
            angle_scaler=scalers["angle_scaler"],
            init_coords=mini_batch_inputs_dict["init_coords"].to(device),
            limit_coords_displacement=limit_coords_displacement,
        )
        intermid_data_dict["batch_abc"] = batch_abc.detach().clone()
        intermid_data_dict["batch_angle"] = batch_angle.detach().clone()
        intermid_data_dict["batch_dir_coords"] = coords_geometry.detach().clone()
    else:
        intermid_data_dict["batch_abc"] = (
            output_dict["batch_abc"].detach().clone().to(device)
        )
        intermid_data_dict["batch_angle"] = (
            output_dict["batch_angle"].detach().clone().to(device)
        )
    intermid_data_dict['wp_size'] = mini_batch_inputs_dict['wp_size'].to(device)
    intermid_data_dict["graph_fallback_geometry"] = mini_batch_inputs_dict.get(
        "graph_fallback_geometry"
    )

    # Update graph using external function
    intermid_data_dict = graph_update(
        intermid_data_dict = intermid_data_dict,
        num_batch_crystal = num_batch_crystal,
        old_g = mini_batch_inputs_dict['g'],
        old_lg = mini_batch_inputs_dict['lg'],
        wyckoff_expander = wyckoff_expander,
        num_workers = num_workers,
        device = device,

    )

    # Update graph-related keys in the original input dict
    for key in [
        'g',
        'lg',
        'batch_displace',
        'batch_dst_ids',
        'batch_src_ids',
        'z_dst_ids',
        'z_src_ids',
        'num_edges',
        
    ]:
        # update graph information
        if not (key in ['g','lg']):
            print(f"<info> update {key} in intermid_data_dict. shape: {mini_batch_inputs_dict[key].shape} ->", end="")
        mini_batch_inputs_dict.pop(key, None)  # remove old key if exists
        mini_batch_inputs_dict[key] = intermid_data_dict[key]
        if not (key in ['g','lg']):
            print(f"{mini_batch_inputs_dict[key].shape}")

    return mini_batch_inputs_dict
