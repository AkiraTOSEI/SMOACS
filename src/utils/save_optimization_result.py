from typing import Dict, List, Optional

import numpy as np
import torch

from src.losses.force_pressure_loss import DIAGNOSTIC_KEYS, FINAL_KEYS


def _cat_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    normalized = []
    for tensor in tensors:
        if tensor.ndim == 0:
            tensor = tensor.unsqueeze(0)
        normalized.append(tensor)
    combined = torch.cat(normalized)
    # Keep the leading sample axis, including a batch containing one crystal.
    for dim in range(combined.ndim - 1, 0, -1):
        combined = combined.squeeze(dim)
    return combined


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
        "lattice_vectors": _cat_tensors(
            [opt_dict["lattice_vectors"] for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "dir_coords": _cat_tensors(
            [opt_dict["normed_batch_dir_coords"] for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "init_coords": _cat_tensors(
            [opt_dict["init_coords"] for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "scaled_batch_abc": _cat_tensors(
            [opt_dict["scaled_batch_abc"] for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "scaled_batch_angle": _cat_tensors(
            [opt_dict["scaled_batch_angle"] for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "ox_states_used_mask": _cat_tensors(
            [opt_dict["ox_states_used_mask"] for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "atomic_mask": _cat_tensors(
            [opt_dict["atomic_mask"] for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "sharpened_ox_mask": _cat_tensors(
            [opt_dict["sharpened_ox_mask"] for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "atomic_distribution": _cat_tensors(
            [opt_dict["atomic_distribution"] for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "ox_mask_learnable_tensor_per_crystal": _cat_tensors(
            [
                opt_dict["ox_mask_learnable_tensor_per_crystal"]
                for opt_dict in optimized_dict_list
            ]
        )
        .detach()
        .cpu()
        .numpy(),
        "normalized_dist": _cat_tensors(
            [opt_dict["normalized_dist"] for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "onehot_x": _cat_tensors(
            [opt_dict["onehot_x"] for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "batch_abc": _cat_tensors(
            [opt_dict["batch_abc"] for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "batch_angle": _cat_tensors(
            [opt_dict["batch_angle"] for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "init_abc": _cat_tensors(
            [opt_dict["init_abc"] for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "init_angles": _cat_tensors(
            [opt_dict["init_angles"] for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "size": _cat_tensors([opt_dict["size"] for opt_dict in optimized_dict_list])
        .detach()
        .cpu()
        .numpy(),
        "site_ids": _cat_tensors(
            [opt_dict["site_ids"] for opt_dict in optimized_dict_list]
        )
        .detach()
        .cpu()
        .numpy(),
        "num_atoms": _cat_tensors(
            [opt_dict["size"] for opt_dict in optimized_dict_list]
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

    optional_keys = [
        "pred_G",
        "pred_B",
        "k_ratio",
        "pred_magmom",
        "pred_energy",
        "loss_G",
        "loss_B",
        "loss_k",
        "loss_energy",
        "loss_total",
        "sa_step",
        "sa_temperature",
        "sa_loss",
        "sa_best_loss",
        "sa_accept_rate",
        "sa_accept_flags",
        "sa_sample_accept_flags",
        "sa_accepted",
        "sa_start_step",
        "grad_sa_hybrid_grad_steps",
        "grad_sa_hybrid_sa_steps",
    ]
    optional_keys.extend(FINAL_KEYS)
    history_keys = [f"{key}_history" for key in DIAGNOSTIC_KEYS]
    optional_keys.extend(history_keys)
    matrix_keys = {"sa_accept_flags", "sa_sample_accept_flags", *history_keys}
    for key in optional_keys:
        if all(key in opt_dict for opt_dict in optimized_dict_list):
            if key in matrix_keys:
                values = [opt_dict[key] for opt_dict in optimized_dict_list]
                if (
                    key == "sa_sample_accept_flags"
                    and len({value.shape[-1] for value in values}) > 1
                ):
                    # Partial batches have different sample axes; preserve each batch.
                    batches = np.empty(len(values), dtype=object)
                    for index, value in enumerate(values):
                        batches[index] = value.detach().cpu().numpy()
                    common_data_dict[key] = batches
                else:
                    common_data_dict[key] = (
                        torch.cat(values, dim=0).detach().cpu().numpy()
                    )
            else:
                common_data_dict[key] = (
                    _cat_tensors([opt_dict[key] for opt_dict in optimized_dict_list])
                    .detach()
                    .cpu()
                    .numpy()
                )

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
                _cat_tensors(
                    [opt_dict[key + model_sfx] for opt_dict in optimized_dict_list]
                )
                .detach()
                .cpu()
                .numpy()
            )

        # History
        for key in ["gap_history", "ef_history", "t_history"]:
            common_data_dict[key + model_sfx] = np.concatenate(
                [
                    np.asarray(opt_dict[key + model_sfx]).reshape(
                        len(opt_dict["size"]), -1
                    )
                    for opt_dict in optimized_dict_list
                ],
                axis=0,
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
