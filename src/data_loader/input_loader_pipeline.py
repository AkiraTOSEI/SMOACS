from typing import Any, Dict, List, Optional, Tuple

import torch

from src.data_loader.alignn_loader import load_ALIGNN_initial_data
from src.data_loader.crystalformer_loader import load_crystalformer_initial_data


def load_initial_data(
    settings_dict: Dict[str, Any],
    model_name: str,
    dir_path: str,
    intermid_dir_path: str,
    use_atomic_mask: bool,
    max_ox_states: int,
    mask_method: str,
    test_mode: bool,
    crystal_system: Optional[str],
    initial_dataset: str,
    num_candidate: Optional[int],
    batch_size: int,
    angle_range: Tuple[float, float],
    abc_range: Tuple[float, float],
    graph_update_mode: bool = False,
) -> List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]:
    """
    Unified interface for loading initial data for different structure optimization models.

    This function loads crystal structure data in batches depending on the specified model type
    (e.g., 'Crystalformer' or 'ALIGNN'), and returns a list of minibatches for use in optimization.

    Args:
        settings_dict (Dict[str, Any]): Optional settings (may be used internally by loaders).
        model_name (str): Name of the model to load data for. Must be either 'Crystalformer' or 'ALIGNN'.
        dir_path (str): Directory containing crystal structure files (.vasp).
        intermid_dir_path (str): Directory containing oxidation masks and radii .npz files.
        use_atomic_mask (bool): Whether to include oxidation-state-based atomic masks.
        max_ox_states (int): Maximum number of oxidation states to consider.
        mask_method (str): Key used for selecting oxidation mask and radii.
        test_mode (bool): Whether to enable testing assertions inside data loaders.
        crystal_system (Optional[str]): Target crystal system (e.g., 'perovskite'), or None.
        num_candidate (Optional[int]): Maximum number of candidate structures to include.
        batch_size (int): Number of structures per batch.
        angle_range (Tuple[float, float]): Min and max bond angles for normalization.
        abc_range (Tuple[float, float]): Min and max lattice lengths for normalization.
        graph_update_mode (bool, optional): Whether to enable graph update mode (ALIGNN only). Default is False.

    Returns:
        List[Tuple[Dict[str, torch.Tensor], Dict[str, Any]]]: A list of minibatch data,
        each containing a structure dictionary and associated scalers.
    """
    if model_name == "Crystalformer":
        # load all initial data
        minibatch_datas = load_crystalformer_initial_data(
            settings_dict=settings_dict,
            dir_path=dir_path,
            intermid_dir_path=intermid_dir_path,
            batch_size=batch_size,
            max_ox_states=max_ox_states,
            num_candidate=num_candidate,
            use_atomic_mask=use_atomic_mask,
            crystal_system=crystal_system,
            initial_dataset=initial_dataset,
            mask_method=mask_method,
            angle_range=angle_range,
            abc_range=abc_range,
            test_mode=test_mode,
        )
    elif model_name == "ALIGNN":
        minibatch_datas = load_ALIGNN_initial_data(
            settings_dict=settings_dict,
            dir_path=dir_path,
            intermid_dir_path=intermid_dir_path,
            batch_size=batch_size,
            max_ox_states=max_ox_states,
            num_candidate=num_candidate,
            use_atomic_mask=use_atomic_mask,
            crystal_system=crystal_system,
            initial_dataset=initial_dataset,
            mask_method=mask_method,
            angle_range=angle_range,
            abc_range=abc_range,
            graph_update_mode=graph_update_mode,
            test_mode=test_mode,
        )
    else:
        raise NotImplementedError(f"model_name, {model_name}, is not implemented yet.")

    return minibatch_datas
