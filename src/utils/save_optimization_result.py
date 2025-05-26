import os
from typing import Dict, List, Optional

import numpy as np
import torch


def save_optimization_results(
    npz_path: str,
    optimized_dict_list: List[Dict[str, torch.Tensor]],
    model_name: str,
    crystal_system: Optional[str],
) -> None:
    """
    Save optimization results into a .npz file.

    This function aggregates optimized structure and property information from multiple
    samples, formats it as NumPy arrays, and saves it to a `.npz` file for future analysis
    and visualization.

    Args:
        npz_path (str): Path to save the `.npz` file.
        optimized_dict_list (List[Dict[str, torch.Tensor]]): List of dictionaries containing
            optimization results for each mini-batch.
        model_name (str): Name of the model used (e.g., 'ALIGNN', 'Crystalformer', or 'Both').
        crystal_system (Optional[str]): Type of crystal system (e.g., 'perovskite'); affects validation.

    Returns:
        None
    """
    common_data_dict = {
        "lattice_vectors": torch.cat(
            [opt_dict["lattice_vectors"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "dir_coords": torch.cat(
            [
                opt_dict["normed_batch_dir_coords"].squeeze()
                for opt_dict in optimized_dict_list
            ]
        )
        .detach()
        .cpu()
        .numpy(),
        "init_coords": torch.cat(
            [opt_dict["init_coords"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "scaled_batch_abc": torch.cat(
            [opt_dict["scaled_batch_abc"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "scaled_batch_angle": torch.cat(
            [
                opt_dict["scaled_batch_angle"].squeeze()
                for opt_dict in optimized_dict_list
            ]
        )
        .detach()
        .cpu()
        .numpy(),
        "ox_states_used_mask": torch.cat(
            [
                opt_dict["ox_states_used_mask"].squeeze()
                for opt_dict in optimized_dict_list
            ]
        )
        .detach()
        .cpu()
        .numpy(),
        "atomic_mask": torch.cat(
            [opt_dict["atomic_mask"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "sharpened_ox_mask": torch.cat(
            [
                opt_dict["sharpened_ox_mask"].squeeze()
                for opt_dict in optimized_dict_list
            ]
        )
        .detach()
        .cpu()
        .numpy(),
        "atomic_distribution": torch.cat(
            [
                opt_dict["atomic_distribution"].squeeze()
                for opt_dict in optimized_dict_list
            ]
        )
        .detach()
        .cpu()
        .numpy(),
        "ox_mask_learnable_tensor_per_crystal": torch.cat(
            [
                opt_dict["ox_mask_learnable_tensor_per_crystal"].squeeze()
                for opt_dict in optimized_dict_list
            ]
        )
        .detach()
        .cpu()
        .numpy(),
        "normalized_dist": torch.cat(
            [opt_dict["normalized_dist"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "onehot_x": torch.cat(
            [opt_dict["onehot_x"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "batch_abc": torch.cat(
            [opt_dict["batch_abc"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "batch_angle": torch.cat(
            [opt_dict["batch_angle"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "init_abc": torch.cat(
            [opt_dict["init_abc"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "init_angles": torch.cat(
            [opt_dict["init_angles"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "size": torch.cat(
            [opt_dict["size"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "site_ids": torch.cat(
            [opt_dict["site_ids"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "num_atoms": torch.cat(
            [opt_dict["size"].squeeze() for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "original_fnames": np.concatenate(
            [opt_dict["fnames"] for opt_dict in optimized_dict_list]
        ),
    }
    if model_name == "Both":
        model_suffixes = ["_crystalformer", "_alignn"]
    else:
        model_suffixes = [""]

    for model_sfx in model_suffixes:
        # loss etc.
        for key in [
            "bandgap_onehot",
            "bandgap_dist",
            "ef_onehot",
            "ef_dist",
            "loss_onehot",
            "tolerance",
            "tolerance_loss",
        ]:
            common_data_dict[key + model_sfx] = (
                torch.cat(
                    [
                        opt_dict[key + model_sfx].squeeze()
                        for opt_dict in optimized_dict_list
                    ]
                )
                .detach()
                .cpu()
                .numpy()
            )

        # History
        for key in ["gap_history", "ef_history", "t_history"]:
            common_data_dict[key + model_sfx] = np.concatenate(
                [
                    opt_dict[key + model_sfx].squeeze()
                    for opt_dict in optimized_dict_list
                ]
            ).T

    # important results
    if model_name == "Both":
        common_data_dict["loss_onehot"] = (
            common_data_dict["loss_onehot_alignn"]
            + common_data_dict["loss_onehot_crystalformer"]
        )
        common_data_dict["bandgap_onehot"] = (
            common_data_dict["bandgap_onehot_alignn"]
            + common_data_dict["bandgap_onehot_crystalformer"]
        ) / 2
        common_data_dict["ef_onehot"] = (
            common_data_dict["ef_onehot_alignn"]
            + common_data_dict["ef_onehot_crystalformer"]
        ) / 2
        if crystal_system == "perovskite":
            torch.testing.assert_close(
                common_data_dict["tolerance_alignn"],
                common_data_dict["tolerance_crystalformer"],
            )
        common_data_dict["tolerance"] = common_data_dict["tolerance_alignn"]

    np.savez(npz_path, **common_data_dict)
