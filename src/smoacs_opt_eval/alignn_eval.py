from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.losses.loss_utils import calculate_loss_from_output
from src.smoacs_opt_eval.alignn_forward import (
    ALIGNN_prediction,
    forward_propagation_ALIGNN,
    predict_all_alignn,
)
from src.smoacs_opt_eval.alignn_graph_update import graph_parameter_update
from src.utils.common import create_element_list
from src.utils.feature_utils import calculate_onehot
from src.utils.coord import direct_to_cartesian_batch
from src.spg_wp.wyckoff_expander import WyckoffExpander


def evaluation_for_each_batch_ALIGNN(
    settings_dict: Dict[str, Any],
    optimized_mini_batch_inputs_dict: dict,
    scalers: dict,
    bandgap_model: Optional[torch.nn.Module],
    e_form_model: Optional[torch.nn.Module],
    gap_loss_func: torch.nn.Module,
    tolerance_loss_func: torch.nn.Module,
    crystal_system: Optional[str] ,
    limit_coords_displacement: Optional[float],
    e_form_min: Optional[float],
    e_form_coef: List[float],
    adding_noise_scale: Optional[float],
    wyckoff_expander:Optional[WyckoffExpander],
    onehot_temperature:float = 1E-8,
    predictors: Optional[Dict[str, Any]] = None,
    loss_composer: Optional[Any] = None,
    primary_metrics: Optional[Dict[str, str]] = None,
    energy_module: Optional[Any] = None,
    energy_coef: float = 0.0,
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
        optimization_target_names (List[str]): Names of the optimization targets.
        optimization_targets (List[torch.Tensor]): List of tensors corresponding to the optimization targets.
        scalers (Dict[str, Any]): Dictionary containing 'abc_scaler' and 'angle_scaler' for rescaling.
        bandgap_model (torch.nn.Module): ALIGNN model for bandgap prediction.
        e_form_model (torch.nn.Module): ALIGNN model for formation energy prediction.
        gap_loss_func (torch.nn.Module): Loss function for bandgap.
        tolerance_loss_func (torch.nn.Module): Loss function for tolerance factor (used for perovskites).
        crystal_system (Optional[str]): If 'perovskite', tolerance loss is applied.
        limit_coords_displacement (Optional[float]): Maximum displacement allowed from initial coordinates.
        e_form_min (Optional[float]): Minimum target for formation energy (used in ef loss).
        Ef_coef_schedule (List[float]): Schedule for weighting the formation energy loss term.
        adding_noise_scale (Optional[float]): If set, applies Gaussian noise to coordinates.
        onehot_temperature (float, optional): Softmax temperature used for converting distributions to one-hot. Defaults to 1e-8.

    Returns:
        Dict[str, Any]: Updated input dictionary augmented with:
            - 'bandgap_dist', 'ef_dist', 'loss_dist'
            - 'bandgap_onehot', 'ef_onehot', 'loss_onehot'
            - 'onehot_x', 'onehot_ox_mask', 'tolerance', 'tolerance_loss'
    """
    print('<info> graph update at evaluation stage')
    optimized_mini_batch_inputs_dict = graph_parameter_update(
        optimization_target_names=optimized_mini_batch_inputs_dict['optimization_target_names'],
        optimization_targets=optimized_mini_batch_inputs_dict['optimization_targets'],
        mini_batch_inputs_dict=optimized_mini_batch_inputs_dict,
        output_dict=optimized_mini_batch_inputs_dict,
        num_batch_crystal=settings_dict['num_batch_crystal'],
        wyckoff_expander=wyckoff_expander,
    ) 

    abc_scaler, angle_scaler = scalers['abc_scaler'], scalers['angle_scaler']
    
    device = optimized_mini_batch_inputs_dict["scaled_batch_abc"].device
    g = optimized_mini_batch_inputs_dict['g'].to(device)
    lg = optimized_mini_batch_inputs_dict['lg'].to(device)
    atom_feat_matrix = optimized_mini_batch_inputs_dict['atom_feat_matrix'].to(device)
    radii_tensor = optimized_mini_batch_inputs_dict['radii_tensor'].to(device)
    
    #print("site_ids in eval:", optimized_mini_batch_inputs_dict['site_ids'])
    abc_range = settings_dict['abc_range']
    angle_range = settings_dict['angle_range']
    superhard_mode = settings_dict.get("superhard_cfg") is not None


    with torch.no_grad():
        atomic_distribution = optimized_mini_batch_inputs_dict['atomic_distribution']
        #scaled_batch_abc = optimized_mini_batch_inputs_dict['scaled_batch_abc']
        #scaled_batch_angle = optimized_mini_batch_inputs_dict['scaled_batch_angle']
        #batch_dir_coords = optimized_mini_batch_inputs_dict['batch_dir_coords'] #あとで消す??
        scaled_batch_abc = torch.round(optimized_mini_batch_inputs_dict['scaled_batch_abc'],decimals=4)
        scaled_batch_angle = torch.round(optimized_mini_batch_inputs_dict['scaled_batch_angle'],decimals=4)
        batch_dir_coords = torch.round(optimized_mini_batch_inputs_dict['batch_dir_coords'],decimals=4)
        ox_mask_learnable_tensor_per_crystal = optimized_mini_batch_inputs_dict['ox_mask_learnable_tensor_per_crystal']
        optimization_targets = [scaled_batch_abc, scaled_batch_angle, batch_dir_coords, atomic_distribution, ox_mask_learnable_tensor_per_crystal]

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
            wyckoff_expander=wyckoff_expander,
            predictors=predictors,
        )
        pred_dict = output_dict.get("pred_dict")
        if loss_composer is not None:
            if pred_dict is None:
                raise ValueError("pred_dict is required when loss_composer is set")
            if not primary_metrics:
                raise ValueError("primary_metrics is required when loss_composer is set")
            bandgap_key = primary_metrics["bandgap_onehot"]
            ef_key = primary_metrics["ef_onehot"]
            bandgap_dist = pred_dict[bandgap_key]
            ef_dist = pred_dict[ef_key]
            context = {
                "gap_loss": lambda x: gap_loss_func(x, reduction="none"),
                "ef_coef": e_form_coef,
            }
            if tolerance_loss_func is not None:
                context["tolerance_loss"] = tolerance_loss_func
            _, _, _ = loss_composer.compute(pred_dict, context)
            if superhard_mode:
                loss_dist = torch.zeros_like(bandgap_dist)
            else:
                _, _, loss_dist, _, _, _ = calculate_loss_from_output(
                    bandgap_pred=bandgap_dist,
                    ef_pred=ef_dist,
                    sharpened_ox_mask=output_dict['sharpened_ox_mask'],
                    normalized_dist=output_dict['normalized_dist'],
                    site_ids=optimized_mini_batch_inputs_dict['site_ids'].to(device),
                    gap_loss_func=gap_loss_func,
                    tolerance_loss_func=tolerance_loss_func,
                    ef_coef=e_form_coef,
                    crystal_system=crystal_system,
                    e_form_min=e_form_min,
                    radii_tensor=radii_tensor,
                )
        else:
            if superhard_mode:
                loss_dist = torch.zeros_like(bandgap_dist)
            else:
                _, _, loss_dist, _, _, _ = calculate_loss_from_output(
                    bandgap_pred=bandgap_dist,
                    ef_pred=ef_dist,
                    sharpened_ox_mask=output_dict['sharpened_ox_mask'],
                    normalized_dist=output_dict['normalized_dist'],
                    site_ids=optimized_mini_batch_inputs_dict['site_ids'].to(device),
                    gap_loss_func=gap_loss_func,
                    tolerance_loss_func=tolerance_loss_func,
                    ef_coef=e_form_coef,
                    crystal_system=crystal_system,
                    e_form_min=e_form_min,
                    radii_tensor=radii_tensor,
                )

        # calculation with one-hot atomic distribution
        decode_output_dict = output_dict | {
            "atomic_distribution": atomic_distribution.to(device),
            "atomic_mask": optimized_mini_batch_inputs_dict["atomic_mask"].to(device),
            "site_ids": optimized_mini_batch_inputs_dict["site_ids"].to(device),
            "masking_temperature": onehot_temperature,
            "wyckoff_expander": wyckoff_expander,
        }
        max_val, onehot_x, onehot_atom_feat, onehot_ox_mask = calculate_onehot(
            decode_output_dict, atom_feat_matrix
        )
        bandgap_onehot, ef_onehot = None, None
        if bandgap_model is not None and e_form_model is not None:
            bandgap_onehot, ef_onehot = ALIGNN_prediction(
                bandgap_model = bandgap_model,
                e_form_model = e_form_model,
                g = g.to(device),
                lg = lg.to(device),
                atomic_features = onehot_atom_feat, # one-hot atom features
                bondlength = output_dict['bondlength'],
                bond_angles = output_dict['bond_angles'],
            )
        if predictors is not None:
            pred_onehot = predict_all_alignn(
                predictors=predictors,
                g=g.to(device),
                lg=lg.to(device),
                atomic_features=onehot_atom_feat,
                bondlength=output_dict['bondlength'],
                bond_angles=output_dict['bond_angles'],
            )
            if primary_metrics:
                bandgap_onehot = pred_onehot[primary_metrics["bandgap_onehot"]]
                ef_onehot = pred_onehot[primary_metrics["ef_onehot"]]
            if "magmom" in pred_onehot:
                optimized_mini_batch_inputs_dict["pred_magmom"] = pred_onehot["magmom"]
        if superhard_mode:
            loss_onehot = torch.zeros_like(bandgap_onehot)
            tolerance_loss = torch.zeros_like(bandgap_onehot)
            tolerance = torch.zeros_like(bandgap_onehot)
        else:
            _, _, loss_onehot, _, tolerance_loss, tolerance = calculate_loss_from_output(
                bandgap_pred=bandgap_onehot,
                ef_pred=ef_onehot,
                sharpened_ox_mask=onehot_ox_mask,
                normalized_dist= onehot_x,
                site_ids=optimized_mini_batch_inputs_dict['site_ids'].to(device),
                gap_loss_func=gap_loss_func,
                tolerance_loss_func=tolerance_loss_func,
                ef_coef=e_form_coef,
                e_form_min=e_form_min,
                crystal_system=crystal_system,
                radii_tensor=radii_tensor,
            )
    # あとで消す
    #print("onehot_atom_feat.shape", onehot_atom_feat.shape)
    #print("onehot_atom_feat", onehot_atom_feat)
    VALID_ELEMENTS98 = create_element_list()
    print([VALID_ELEMENTS98[a_idx] for a_idx in np.argmax(onehot_x[-9:].detach().cpu().numpy(),axis=1)])
    print(batch_dir_coords[-9:].detach().cpu().numpy())


    pred_energy = None
    if energy_module is not None:
        size = optimized_mini_batch_inputs_dict["size"].to(device)
        batch_index = torch.repeat_interleave(
            torch.arange(len(size), device=device), size
        )
        lattice_per_atom = output_dict["lattice_vectors"][batch_index]
        positions = direct_to_cartesian_batch(
            output_dict["normed_batch_dir_coords"], lattice_per_atom
        )
        energy = energy_module.compute_energy(
            atom_types=onehot_x,
            positions=positions,
            lattice=output_dict["lattice_vectors"],
            sizes=size,
            batch_index=batch_index,
        )
        pred_energy = energy / size.to(energy.dtype)

    superhard_cfg = settings_dict.get("superhard_cfg")
    if superhard_cfg is not None:
        pred_G = bandgap_onehot
        pred_B = ef_onehot
        k_eps = float(superhard_cfg.get("k_eps", 1e-8))
        k_ratio = pred_G / (pred_B + k_eps)
        loss_G = -pred_G
        loss_B = -pred_B
        loss_k = (
            torch.relu(superhard_cfg["k_min"] - k_ratio) ** 2
            + torch.relu(k_ratio - superhard_cfg["k_max"]) ** 2
        )
        loss_energy = pred_energy if pred_energy is not None else torch.zeros_like(loss_G)
        loss_total = (
            superhard_cfg["G_coef"] * loss_G
            + superhard_cfg["B_coef"] * loss_B
            + superhard_cfg["k_coef"] * loss_k
            + energy_coef * loss_energy
        )

    # 結果の記録
    optimized_mini_batch_inputs_dict = optimized_mini_batch_inputs_dict | output_dict
    optimized_mini_batch_inputs_dict['max_val'] = max_val
    optimized_mini_batch_inputs_dict['bandgap_onehot'] = bandgap_onehot
    optimized_mini_batch_inputs_dict['bandgap_dist'] = bandgap_dist
    optimized_mini_batch_inputs_dict['ef_onehot'] = ef_onehot
    optimized_mini_batch_inputs_dict['ef_dist'] = ef_dist
    optimized_mini_batch_inputs_dict['loss_onehot'] = loss_onehot
    optimized_mini_batch_inputs_dict['loss_dist'] = loss_dist
    optimized_mini_batch_inputs_dict['onehot_x'] = onehot_x
    optimized_mini_batch_inputs_dict['tolerance'] = tolerance
    optimized_mini_batch_inputs_dict['tolerance_loss'] = tolerance_loss
    optimized_mini_batch_inputs_dict['onehot_ox_mask'] = onehot_ox_mask
    if pred_energy is not None:
        optimized_mini_batch_inputs_dict["pred_energy"] = pred_energy
    if superhard_cfg is not None:
        optimized_mini_batch_inputs_dict["pred_G"] = pred_G
        optimized_mini_batch_inputs_dict["pred_B"] = pred_B
        optimized_mini_batch_inputs_dict["k_ratio"] = k_ratio
        optimized_mini_batch_inputs_dict["loss_G"] = loss_G
        optimized_mini_batch_inputs_dict["loss_B"] = loss_B
        optimized_mini_batch_inputs_dict["loss_k"] = loss_k
        optimized_mini_batch_inputs_dict["loss_energy"] = loss_energy
        optimized_mini_batch_inputs_dict["loss_total"] = loss_total

    return optimized_mini_batch_inputs_dict
