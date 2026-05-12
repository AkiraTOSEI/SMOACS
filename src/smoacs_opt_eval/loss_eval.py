from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from src.losses.element_count_loss import element_count_l1_loss
from src.losses.loss_utils import calculate_loss_from_output
from src.smoacs_opt_eval.alignn_forward import forward_propagation_ALIGNN
from src.smoacs_opt_eval.crystalformer_forward import forward_propagation_Crystalformer
from src.utils.coord import direct_to_cartesian_batch


def evaluate_loss_alignn(
    optimization_targets,
    mini_batch_inputs_dict,
    scalers,
    temperature: float,
    gap_loss_func,
    tolerance_loss_func,
    e_form_coef: float,
    crystal_system: Optional[str],
    e_form_min: Optional[float],
    bandgap_model,
    e_form_model,
    predictors=None,
    loss_composer=None,
    primary_metrics=None,
    energy_module=None,
    energy_coef: float = 0.0,
    energy_every: int = 1,
    step_i: Optional[int] = None,
    limit_coords_displacement: Optional[float] = None,
    abc_range=(2.0, 10.0),
    angle_range=(30.0, 150.0),
    adding_noise_scale: Optional[float] = None,
    wyckoff_expander=None,
    use_element_count_l1_loss: bool = False,
    element_count_l1_weight: float = 1.0,
    grad_enabled: bool = True,
    superhard_mode: bool = False,
) -> Dict[str, Any]:
    with torch.set_grad_enabled(grad_enabled):
        bandgap_pred, ef_pred, output_dict = forward_propagation_ALIGNN(
            optimization_targets=optimization_targets,
            fixed_inputs=mini_batch_inputs_dict,
            scalers=scalers,
            temperature=temperature,
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
        energy_loss = None
        energy_per_atom = None
        if (
            energy_module is not None
            and energy_coef > 0.0
            and energy_every > 0
            and (step_i is None or step_i % energy_every == 0)
        ):
            device = optimization_targets[0].device
            size = mini_batch_inputs_dict["size"].to(device)
            batch_index = torch.repeat_interleave(
                torch.arange(len(size), device=device), size
            )
            lattice_per_atom = output_dict["lattice_vectors"][batch_index]
            positions = direct_to_cartesian_batch(
                output_dict["normed_batch_dir_coords"], lattice_per_atom
            )
            energy = energy_module.compute_energy(
                atom_types=output_dict["normalized_dist"],
                positions=positions,
                lattice=output_dict["lattice_vectors"],
                sizes=size,
                batch_index=batch_index,
            )
            energy_per_atom = energy / size.to(energy.dtype)
            energy_loss = energy_per_atom.mean() * energy_coef

        if loss_composer is not None:
            if pred_dict is None:
                raise ValueError("pred_dict is required when loss_composer is set")
            if not primary_metrics:
                raise ValueError("primary_metrics is required when loss_composer is set")
            bandgap_key = primary_metrics["bandgap_onehot"]
            ef_key = primary_metrics["ef_onehot"]
            bandgap_pred = pred_dict[bandgap_key]
            ef_pred = pred_dict[ef_key]
            context = {
                "gap_loss": lambda x: gap_loss_func(x, reduction="none"),
                "ef_coef": e_form_coef,
            }
            if tolerance_loss_func is not None:
                context["tolerance_loss"] = tolerance_loss_func
            total_loss, each_loss, _ = loss_composer.compute(pred_dict, context)
            if superhard_mode:
                gap_loss = torch.zeros_like(each_loss)
                ef_loss = torch.zeros_like(each_loss)
                tolerance_loss = torch.zeros_like(each_loss)
                tolerance = torch.zeros_like(each_loss)
            else:
                (
                    _,
                    _,
                    gap_loss,
                    ef_loss,
                    tolerance_loss,
                    tolerance,
                ) = calculate_loss_from_output(
                    bandgap_pred=bandgap_pred,
                    ef_pred=ef_pred,
                    sharpened_ox_mask=output_dict["sharpened_ox_mask"],
                    normalized_dist=output_dict["normalized_dist"],
                    site_ids=mini_batch_inputs_dict["site_ids"].to(
                        output_dict["normalized_dist"].device
                    ),
                    gap_loss_func=gap_loss_func,
                    tolerance_loss_func=tolerance_loss_func,
                    ef_coef=e_form_coef,
                    crystal_system=crystal_system,
                    e_form_min=e_form_min,
                    radii_tensor=mini_batch_inputs_dict["radii_tensor"].to(
                        output_dict["normalized_dist"].device
                    ),
                )
        else:
            total_loss, each_loss, gap_loss, ef_loss, tolerance_loss, tolerance = (
                calculate_loss_from_output(
                    bandgap_pred=bandgap_pred,
                    ef_pred=ef_pred,
                    sharpened_ox_mask=output_dict["sharpened_ox_mask"],
                    normalized_dist=output_dict["normalized_dist"],
                    site_ids=mini_batch_inputs_dict["site_ids"].to(
                        output_dict["normalized_dist"].device
                    ),
                    gap_loss_func=gap_loss_func,
                    tolerance_loss_func=tolerance_loss_func,
                    ef_coef=e_form_coef,
                    crystal_system=crystal_system,
                    e_form_min=e_form_min,
                    radii_tensor=mini_batch_inputs_dict["radii_tensor"].to(
                        output_dict["normalized_dist"].device
                    ),
                )
            )

        element_count_l1_value = torch.zeros((), device=total_loss.device)
        per_crystal_loss = None
        if use_element_count_l1_loss:
            size_for_loss = mini_batch_inputs_dict["size"].to(total_loss.device)
            current_atomic_distribution = optimization_targets[3]
            if int(size_for_loss.sum().item()) != current_atomic_distribution.shape[0]:
                if "wp_size" in mini_batch_inputs_dict:
                    size_for_loss = mini_batch_inputs_dict["wp_size"].to(
                        total_loss.device
                    )
            per_crystal_loss = element_count_l1_loss(
                F.softplus(current_atomic_distribution),
                size_for_loss,
                reduction="none",
            )
            element_count_l1_value = per_crystal_loss.mean()
            total_loss = total_loss + element_count_l1_weight * element_count_l1_value
            each_loss = each_loss + element_count_l1_weight * per_crystal_loss

        if energy_loss is not None:
            total_loss = total_loss + energy_loss
            each_loss = each_loss + (energy_per_atom * energy_coef)

        return {
            "total_loss": total_loss,
            "each_loss": each_loss,
            "gap_loss": gap_loss,
            "ef_loss": ef_loss,
            "tolerance_loss": tolerance_loss,
            "tolerance": tolerance,
            "bandgap_pred": bandgap_pred,
            "ef_pred": ef_pred,
            "energy_loss": energy_loss,
            "energy_per_atom": energy_per_atom,
            "element_count_l1_value": element_count_l1_value,
            "per_crystal_loss": per_crystal_loss,
            "output_dict": output_dict,
            "pred_dict": pred_dict,
        }


def evaluate_loss_crystalformer(
    optimization_targets,
    mini_batch_inputs_dict,
    scalers,
    temperature: float,
    gap_loss_func,
    tolerance_loss_func,
    e_form_coef: float,
    crystal_system: Optional[str],
    e_form_min: Optional[float],
    bandgap_model,
    e_form_model,
    predictors=None,
    loss_composer=None,
    primary_metrics=None,
    energy_module=None,
    energy_coef: float = 0.0,
    energy_every: int = 1,
    step_i: Optional[int] = None,
    limit_coords_displacement: Optional[float] = None,
    abc_range=(2.0, 10.0),
    angle_range=(30.0, 150.0),
    adding_noise_scale: Optional[float] = None,
    use_element_count_l1_loss: bool = False,
    element_count_l1_weight: float = 1.0,
    grad_enabled: bool = True,
) -> Dict[str, Any]:
    with torch.set_grad_enabled(grad_enabled):
        bandgap_pred, ef_pred, output_dict = forward_propagation_Crystalformer(
            optimization_targets=optimization_targets,
            fixed_inputs=mini_batch_inputs_dict,
            scalers=scalers,
            temperature=temperature,
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            adding_noise_scale=adding_noise_scale,
            limit_coords_displacement=limit_coords_displacement,
            abc_range=abc_range,
            angle_range=angle_range,
            predictors=predictors,
        )

        pred_dict = output_dict.get("pred_dict")
        energy_loss = None
        energy_per_atom = None
        if (
            energy_module is not None
            and energy_coef > 0.0
            and energy_every > 0
            and (step_i is None or step_i % energy_every == 0)
        ):
            device = optimization_targets[0].device
            size = mini_batch_inputs_dict["size"].to(device)
            energy = energy_module.compute_energy(
                atom_types=output_dict["normalized_dist"],
                positions=output_dict["pos"],
                lattice=output_dict["lattice_vectors"],
                sizes=size,
                batch_index=output_dict["batch"],
            )
            energy_per_atom = energy / size.to(energy.dtype)
            energy_loss = energy_per_atom.mean() * energy_coef

        if loss_composer is not None:
            if pred_dict is None:
                raise ValueError("pred_dict is required when loss_composer is set")
            if not primary_metrics:
                raise ValueError("primary_metrics is required when loss_composer is set")
            bandgap_key = primary_metrics["bandgap_onehot"]
            ef_key = primary_metrics["ef_onehot"]
            bandgap_pred = pred_dict[bandgap_key]
            ef_pred = pred_dict[ef_key]
            context = {
                "gap_loss": lambda x: gap_loss_func(x, reduction="none"),
                "ef_coef": e_form_coef,
            }
            if tolerance_loss_func is not None:
                context["tolerance_loss"] = tolerance_loss_func
            total_loss, each_loss, _ = loss_composer.compute(pred_dict, context)
            (
                _,
                _,
                gap_loss,
                ef_loss,
                tolerance_loss,
                tolerance,
            ) = calculate_loss_from_output(
                bandgap_pred=bandgap_pred,
                ef_pred=ef_pred,
                sharpened_ox_mask=output_dict["sharpened_ox_mask"],
                normalized_dist=output_dict["normalized_dist"],
                site_ids=mini_batch_inputs_dict["site_ids"].to(
                    output_dict["normalized_dist"].device
                ),
                gap_loss_func=gap_loss_func,
                tolerance_loss_func=tolerance_loss_func,
                ef_coef=e_form_coef,
                crystal_system=crystal_system,
                e_form_min=e_form_min,
                radii_tensor=mini_batch_inputs_dict["radii_tensor"].to(
                    output_dict["normalized_dist"].device
                ),
            )
        else:
            total_loss, each_loss, gap_loss, ef_loss, tolerance_loss, tolerance = (
                calculate_loss_from_output(
                    bandgap_pred=bandgap_pred,
                    ef_pred=ef_pred,
                    sharpened_ox_mask=output_dict["sharpened_ox_mask"],
                    normalized_dist=output_dict["normalized_dist"],
                    site_ids=mini_batch_inputs_dict["site_ids"].to(
                        output_dict["normalized_dist"].device
                    ),
                    gap_loss_func=gap_loss_func,
                    tolerance_loss_func=tolerance_loss_func,
                    ef_coef=e_form_coef,
                    crystal_system=crystal_system,
                    e_form_min=e_form_min,
                    radii_tensor=mini_batch_inputs_dict["radii_tensor"].to(
                        output_dict["normalized_dist"].device
                    ),
                )
            )

        element_count_l1_value = torch.zeros((), device=total_loss.device)
        per_crystal_loss = None
        if use_element_count_l1_loss:
            size_for_loss = mini_batch_inputs_dict["size"].to(total_loss.device)
            current_atomic_distribution = optimization_targets[3]
            per_crystal_loss = element_count_l1_loss(
                F.softplus(current_atomic_distribution),
                size_for_loss,
                reduction="none",
            )
            element_count_l1_value = per_crystal_loss.mean()
            total_loss = total_loss + element_count_l1_weight * element_count_l1_value
            each_loss = each_loss + element_count_l1_weight * per_crystal_loss

        if energy_loss is not None:
            total_loss = total_loss + energy_loss
            each_loss = each_loss + (energy_per_atom * energy_coef)

        return {
            "total_loss": total_loss,
            "each_loss": each_loss,
            "gap_loss": gap_loss,
            "ef_loss": ef_loss,
            "tolerance_loss": tolerance_loss,
            "tolerance": tolerance,
            "bandgap_pred": bandgap_pred,
            "ef_pred": ef_pred,
            "energy_loss": energy_loss,
            "energy_per_atom": energy_per_atom,
            "element_count_l1_value": element_count_l1_value,
            "per_crystal_loss": per_crystal_loss,
            "output_dict": output_dict,
            "pred_dict": pred_dict,
        }
