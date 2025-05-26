from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.losses.loss_utils import calculate_loss_from_output
from src.smoacs_opt_eval.alignn_forward import (
    ALIGNN_prediction,
    forward_propagation_ALIGNN,
)
from src.smoacs_opt_eval.alignn_graph_update import (
    update_graphs_and_trainable_parameters,
)
from src.utils.common import create_element_list
from src.utils.feature_utils import calculate_onehot


def evaluation_for_each_batch_ALIGNN(
    settings_dict: Dict[str, Any],
    optimized_mini_batch_inputs_dict: dict,
    scalers: dict,
    bandgap_model: torch.nn.Module,
    e_form_model: torch.nn.Module,
    gap_loss_func: torch.nn.Module,
    tolerance_loss_func: torch.nn.Module,
    crystal_system: Optional[str],
    limit_coords_displacement: Optional[float],
    e_form_min: Optional[float],
    ef_coef4eval: int,
    adding_noise_scale: Optional[float],
    onehot_temperature: float = 1e-8,
) -> Dict[str, Any]:
    """
    Evaluate optimized crystal structures using ALIGNN predictors.

    This function updates the graph representation of each crystal structure,
    performs predictions for bandgap and formation energy using both soft (distribution)
    and hard (one-hot) atomic features, and computes associated losses.

    Args:
        settings_dict (Dict[str, Any]): Evaluation configuration including `abc_range`, `angle_range`, etc.
        optimized_mini_batch_inputs_dict (Dict[str, Any]): Structure dictionary after optimization,
            containing variables like `atomic_distribution`, `scaled_batch_abc`, `batch_dir_coords`, etc.
        scalers (Dict[str, Any]): Dictionary containing 'abc_scaler' and 'angle_scaler' for rescaling.
        bandgap_model (torch.nn.Module): ALIGNN model for bandgap prediction.
        e_form_model (torch.nn.Module): ALIGNN model for formation energy prediction.
        gap_loss_func (torch.nn.Module): Loss function for bandgap.
        tolerance_loss_func (torch.nn.Module): Loss function for tolerance factor (used for perovskites).
        crystal_system (Optional[str]): If 'perovskite', tolerance loss is applied.
        limit_coords_displacement (Optional[float]): Maximum displacement allowed from initial coordinates.
        e_form_min (Optional[float]): Minimum target for formation energy (used in ef loss).
        ef_coef4eval (int): weighting the formation energy loss term.
        adding_noise_scale (Optional[float]): If set, applies Gaussian noise to coordinates.
        onehot_temperature (float, optional): Softmax temperature used for converting distributions to one-hot. Defaults to 1e-8.

    Returns:
        Dict[str, Any]: Updated input dictionary augmented with:
            - 'bandgap_dist', 'ef_dist', 'loss_dist'
            - 'bandgap_onehot', 'ef_onehot', 'loss_onehot'
            - 'onehot_x', 'onehot_ox_mask', 'tolerance', 'tolerance_loss'
    """
    print("<info> graph update at evaluation stage")
    updated_dict, scalers, optimiers_dict, learnable_parameters_dict = (
        update_graphs_and_trainable_parameters(
            optimized_mini_batch_inputs_dict=optimized_mini_batch_inputs_dict,
            settings_dict=settings_dict,
            tmp_poscar_dir="./tmp_poscars",
        )
    )
    optimized_mini_batch_inputs_dict.update(updated_dict)

    abc_scaler, angle_scaler = scalers["abc_scaler"], scalers["angle_scaler"]

    g = optimized_mini_batch_inputs_dict["g"].to("cuda")
    lg = optimized_mini_batch_inputs_dict["lg"].to("cuda")
    atom_feat_matrix = optimized_mini_batch_inputs_dict["atom_feat_matrix"].to("cuda")
    radii_tensor = optimized_mini_batch_inputs_dict["radii_tensor"].to("cuda")

    # print("site_ids in eval:", optimized_mini_batch_inputs_dict['site_ids'])
    abc_range = settings_dict["abc_range"]
    angle_range = settings_dict["angle_range"]

    with torch.no_grad():
        atomic_distribution = optimized_mini_batch_inputs_dict["atomic_distribution"]
        # scaled_batch_abc = optimized_mini_batch_inputs_dict['scaled_batch_abc']
        # scaled_batch_angle = optimized_mini_batch_inputs_dict['scaled_batch_angle']
        # batch_dir_coords = optimized_mini_batch_inputs_dict['batch_dir_coords'] #あとで消す??
        scaled_batch_abc = torch.round(
            optimized_mini_batch_inputs_dict["scaled_batch_abc"], decimals=4
        )
        scaled_batch_angle = torch.round(
            optimized_mini_batch_inputs_dict["scaled_batch_angle"], decimals=4
        )
        batch_dir_coords = torch.round(
            optimized_mini_batch_inputs_dict["batch_dir_coords"], decimals=4
        )
        ox_mask_learnable_tensor_per_crystal = optimized_mini_batch_inputs_dict[
            "ox_mask_learnable_tensor_per_crystal"
        ]
        optimization_targets = [
            scaled_batch_abc,
            scaled_batch_angle,
            batch_dir_coords,
            atomic_distribution,
            ox_mask_learnable_tensor_per_crystal,
        ]

        # calculation with atomic distribution
        bandgap_dist, ef_dist, output_dict = forward_propagation_ALIGNN(
            optimization_targets=optimization_targets,
            fixed_inputs=optimized_mini_batch_inputs_dict,
            scalers=scalers,
            temperature=onehot_temperature,
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            adding_noise_scale=adding_noise_scale,
            limit_coords_displacement=limit_coords_displacement,
            abc_range=abc_range,
            angle_range=angle_range,
        )
        _, _, loss_dist, _, _, _ = calculate_loss_from_output(
            bandgap_pred=bandgap_dist,
            ef_pred=ef_dist,
            sharpened_ox_mask=output_dict["sharpened_ox_mask"],
            normalized_dist=output_dict["normalized_dist"],
            site_ids=optimized_mini_batch_inputs_dict["site_ids"].to("cuda"),
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            ef_coef=ef_coef4eval,
            crystal_system=crystal_system,
            e_form_min=e_form_min,
            radii_tensor=radii_tensor,
        )

        # calculation with one-hot atomic distribution
        max_val, onehot_x, onehot_atom_feat, onehot_ox_mask = calculate_onehot(
            output_dict, atom_feat_matrix
        )
        bandgap_onehot, ef_onehot = ALIGNN_prediction(
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            g=g.to("cuda"),
            lg=lg.to("cuda"),
            atomic_features=onehot_atom_feat,  # one-hot atom features
            bondlength=output_dict["bondlength"],
            bond_angles=output_dict["bond_angles"],
        )
        _, _, loss_onehot, _, tolerance_loss, tolerance = calculate_loss_from_output(
            bandgap_pred=bandgap_onehot,
            ef_pred=ef_onehot,
            sharpened_ox_mask=onehot_ox_mask,
            normalized_dist=onehot_x,
            site_ids=optimized_mini_batch_inputs_dict["site_ids"].to("cuda"),
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            ef_coef=ef_coef4eval,
            e_form_min=e_form_min,
            crystal_system=crystal_system,
            radii_tensor=radii_tensor,
        )
    # あとで消す
    # print("onehot_atom_feat.shape", onehot_atom_feat.shape)
    # print("onehot_atom_feat", onehot_atom_feat)
    VALID_ELEMENTS98 = create_element_list()
    print(
        [
            VALID_ELEMENTS98[a_idx]
            for a_idx in np.argmax(onehot_x[-9:].detach().cpu().numpy(), axis=1)
        ]
    )
    print(batch_dir_coords[-9:].detach().cpu().numpy())

    # 結果の記録
    optimized_mini_batch_inputs_dict = optimized_mini_batch_inputs_dict | output_dict
    optimized_mini_batch_inputs_dict["max_val"] = max_val
    optimized_mini_batch_inputs_dict["bandgap_onehot"] = bandgap_onehot
    optimized_mini_batch_inputs_dict["bandgap_dist"] = bandgap_dist
    optimized_mini_batch_inputs_dict["ef_onehot"] = ef_onehot
    optimized_mini_batch_inputs_dict["ef_dist"] = ef_dist
    optimized_mini_batch_inputs_dict["loss_onehot"] = loss_onehot
    optimized_mini_batch_inputs_dict["loss_dist"] = loss_dist
    optimized_mini_batch_inputs_dict["onehot_x"] = onehot_x
    optimized_mini_batch_inputs_dict["tolerance"] = tolerance
    optimized_mini_batch_inputs_dict["tolerance_loss"] = tolerance_loss
    optimized_mini_batch_inputs_dict["onehot_ox_mask"] = onehot_ox_mask

    return optimized_mini_batch_inputs_dict
