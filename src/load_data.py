from typing import Any, Dict, Optional, Tuple

from .load_data_for_alignn import load_ALIGNN_initial_data
from .load_data_for_crystalformer import load_crystalformer_initial_data


def load_initial_data(
    settings_dict:Dict[str,Any],
    model_name:str,
    dir_path:str,
    mask_data_path:str,
    radii_data_path:Optional[str],
    use_atomic_mask:bool,
    max_ox_states:int,
    test_mode:bool,
    device:str,
    num_candidate:Optional[int],
    batch_size:int,
    angle_range:Tuple[float,float],
    abc_range:Tuple[float,float],
    graph_update_mode:bool = False,
):
    if model_name == "Crystalformer":
        # load all initial data
        minibatch_datas = load_crystalformer_initial_data(
            settings_dict=settings_dict,
            dir_path=dir_path,
            mask_data_path=mask_data_path,
            radii_data_path=radii_data_path,
            batch_size=batch_size,
            max_ox_states=max_ox_states,
            num_candidate=num_candidate,
            use_atomic_mask=use_atomic_mask,
            angle_range = angle_range,
            abc_range = abc_range,
            test_mode=test_mode,
            device=device

        )
    elif model_name == 'ALIGNN':
        minibatch_datas = load_ALIGNN_initial_data(
            settings_dict=settings_dict,
            dir_path=dir_path,
            mask_data_path=mask_data_path,
            radii_data_path=radii_data_path,
            batch_size=batch_size,
            max_ox_states=max_ox_states,
            num_candidate=num_candidate,
            use_atomic_mask=use_atomic_mask,
            angle_range = angle_range,
            abc_range = abc_range,
            graph_update_mode = graph_update_mode,
            test_mode=test_mode,
            device=device,
        )
    else:
        raise NotImplementedError(f"model_name, {model_name}, is not implemented yet.")

    return minibatch_datas