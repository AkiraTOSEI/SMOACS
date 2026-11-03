from typing import Any, Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.optimization.optimizer_utils import define_optimizer_and_scheduler
from src.optimizers.proposals import propose_state
from src.optimizers.simulated_annealing import (
    SimulatedAnnealingOptimizer,
    geometric_schedule_with_endpoints,
)
from src.smoacs_opt_eval.loss_eval import evaluate_loss_crystalformer
from src.smoacs_opt_eval.step_update_utils import update_parameters_step
from src.utils.common import create_element_list


def optimize_solution_Crystalformer(
    settings_dict: Dict[str, Any],
    mini_batch_inputs_dict: dict,
    learning_rates: List[float],
    learning_rate_cycle: List[int],
    num_steps: int,
    dist_temp_sche: List[float],
    e_form_coef: float,
    gap_loss_func: torch.nn.Module,
    tolerance_loss_func: Optional[torch.nn.Module],
    bandgap_model: Optional[torch.nn.Module],
    e_form_model: Optional[torch.nn.Module],
    scalers: dict,
    crystal_system: Optional[torch.nn.Module],
    limit_coords_displacement: Optional[float],
    e_form_min: Optional[float],
    adding_noise_scale: Optional[float],
    angle_optimization: bool = True,
    device: str = "cuda",
    time_series_save: bool = True,
    predictors: Optional[Dict[str, Any]] = None,
    loss_composer: Optional[Any] = None,
    primary_metrics: Optional[Dict[str, str]] = None,
    energy_module: Optional[Any] = None,
    energy_coef: float = 0.0,
    energy_every: int = 1,
    use_element_count_l1_loss: bool = False,
    element_count_l1_weight: float = 0.0,
    optimizer: str = "gradient",
):
    """
    Performs optimization of a batch of candidate crystal structures using Crystalformer.

    Args:
        settings_dict: Dictionary containing configuration settings.
        mini_batch_inputs_dict: Dictionary containing batch-specific model inputs.
        learning_rates: List of learning rates for lattice, atom, and coordinate optimization.
        learning_rate_cycle: List of cycle lengths for each optimizer.
        num_steps: Number of optimization steps.
        dist_temp_sche: Temperature schedule for softmax during optimization.
        e_form_coef: weight for  formation energy loss.
        gap_loss_func: Loss function module for bandgap.
        bandgap_model: ALIGNN model predicting bandgap.
        e_form_model: ALIGNN model predicting formation energy.
        scalers: Dictionary containing ABC and angle scalers.
        adding_noise_scale: Amount of noise to add to coordinates.
        e_form_min: Minimum formation energy threshold.
        limit_coords_displacement: Maximum displacement from initial coordinates.
        crystal_system: Optional crystal system type (e.g. "perovskite").
        angle_optimization: Whether to optimize lattice angles.
        tolerance_loss_func: Loss function module for tolerance.
        device: Computation device ('cuda' or 'cpu').
        time_series_save: Whether to store optimization history.

    Returns:
        Tuple containing:
            - optimized_mini_batch_inputs_dict: Dictionary of optimized inputs.
            - time_series: List of history arrays for structural variables and predictions.
            - scalers: Updated scalers.
    """
    abc_scaler = scalers["abc_scaler"]
    angle_scaler = scalers["angle_scaler"]
    if hasattr(abc_scaler, "base_length"):
        abc_scaler.device = device
        abc_scaler.base_length = abc_scaler.base_length.to(device)
        if hasattr(abc_scaler, "init_batch_abc") and torch.is_tensor(
            abc_scaler.init_batch_abc
        ):
            abc_scaler.init_batch_abc = abc_scaler.init_batch_abc.to(device)
    device = torch.device(device)
    requires_grad = optimizer == "gradient"
    atomic_distribution = torch.nn.Parameter(
        mini_batch_inputs_dict["atomic_distribution"].clone().to(device),
        requires_grad=requires_grad,
    )
    scaled_batch_abc = torch.nn.Parameter(
        abc_scaler.scale(mini_batch_inputs_dict["batch_abc"].clone().to(device)),
        requires_grad=requires_grad,
    )
    scaled_batch_angle = torch.nn.Parameter(
        angle_scaler.scale(mini_batch_inputs_dict["batch_angle"].clone().to(device)),
        requires_grad=requires_grad,
    )
    batch_dir_coords = torch.nn.Parameter(
        mini_batch_inputs_dict["batch_dir_coords"].clone().to(device),
        requires_grad=requires_grad,
    )
    ox_mask_learnable_tensor_per_crystal = torch.nn.Parameter(
        mini_batch_inputs_dict["ox_mask_learnable_tensor_per_crystal"]
        .clone()
        .to(device),
        requires_grad=requires_grad,
    )
    abc_range = settings_dict["abc_range"]
    angle_range = settings_dict["angle_range"]

    # optimizer and scheduler
    copy_mutation = settings_dict["copy_mutation"]
    (
        optimizer_lattice,
        optimizer_atom,
        optimizer_coords,
        scheduler_lattice,
        scheduler_atom,
        scheduler_coords,
    ) = define_optimizer_and_scheduler(
        lattice_lr=learning_rates[0],
        atom_lr=learning_rates[1],
        coords_lr=learning_rates[2],
        ox_mask_learnable_tensor_per_crystal=ox_mask_learnable_tensor_per_crystal,
        atomic_distribution=atomic_distribution,
        scaled_batch_abc=scaled_batch_abc,
        scaled_batch_angle=scaled_batch_angle,
        batch_dir_coords=batch_dir_coords,
        angle_optimization=angle_optimization,
        num_steps=num_steps,
        lattice_cycle=learning_rate_cycle[0],
        atom_cycle=learning_rate_cycle[1],
        coords_cycle=learning_rate_cycle[2],
    )

    # initialzation of history
    lattice_mat_history, coords_history, normed_coords_history, elememts_history = (
        [],
        [],
        [],
        [],
    )
    abc_history, angle_history = [], []
    gap_history, e_form_history, t_loss_history = [], [], []
    force_pressure_history = {}
    element_count_l1_history, element_count_l1_max_history = [], []
    VALID_ELEMENTS98 = create_element_list()

    optimization_targets = [
        scaled_batch_abc,
        scaled_batch_angle,
        batch_dir_coords,
        atomic_distribution,
        ox_mask_learnable_tensor_per_crystal,
    ]
    optimization_target_names = [
        "scaled_batch_abc",
        "scaled_batch_angle",
        "batch_dir_coords",
        "atomic_distribution",
        "ox_mask_learnable_tensor_per_crystal",
    ]

    if optimizer == "sa":
        sa_cfg = settings_dict.get("sa_config", {})
        schedule = geometric_schedule_with_endpoints(
            sa_cfg.get("t0", 1.0),
            sa_cfg.get("tmin", 1e-3),
            num_steps,
        )
        trials_per_step = int(sa_cfg.get("trials_per_step", 1))
        if angle_optimization:
            move_probs = {"coords": 1.0 / 3.0, "lattice": 1.0 / 3.0, "angle": 1.0 / 3.0}
        else:
            move_probs = {"coords": 0.5, "lattice": 0.5}
        move_scales = {
            "coords": float(sa_cfg.get("coords_sigma", 1e-3)),
            "lattice": float(sa_cfg.get("lattice_sigma", 1e-3)),
            "angle": float(sa_cfg.get("angle_sigma", 1e-3)),
        }
        sa_per_sample_accept = bool(sa_cfg.get("per_sample_accept", True))
        sample_sizes_by_state = [
            None,
            None,
            mini_batch_inputs_dict["size"].to(device=device, dtype=torch.long),
            mini_batch_inputs_dict["size"].to(device=device, dtype=torch.long),
            None,
        ]
        sa_accept_flags: List[List[int]] = []
        sa_sample_accept_flags: List[List[List[int]]] = []

        def evaluate_fn(state, step_i):
            eval_out = evaluate_loss_crystalformer(
                optimization_targets=state,
                mini_batch_inputs_dict=mini_batch_inputs_dict,
                scalers=scalers,
                temperature=dist_temp_sche[step_i],
                gap_loss_func=gap_loss_func,
                tolerance_loss_func=tolerance_loss_func,
                e_form_coef=e_form_coef,
                crystal_system=crystal_system,
                e_form_min=e_form_min,
                bandgap_model=bandgap_model,
                e_form_model=e_form_model,
                predictors=predictors,
                loss_composer=loss_composer,
                primary_metrics=primary_metrics,
                energy_module=energy_module,
                force_pressure_config=settings_dict.get("force_pressure_config"),
                energy_coef=energy_coef,
                energy_every=energy_every,
                step_i=step_i,
                limit_coords_displacement=limit_coords_displacement,
                abc_range=abc_range,
                angle_range=angle_range,
                adding_noise_scale=adding_noise_scale,
                use_element_count_l1_loss=use_element_count_l1_loss,
                element_count_l1_weight=element_count_l1_weight,
                grad_enabled=False,
            )
            return eval_out

        def proposal_fn(state, generator):
            return propose_state(
                state=state,
                generator=generator,
                move_probs=move_probs,
                move_scales=move_scales,
                init_coords=mini_batch_inputs_dict.get("init_coords", None),
                limit_coords_displacement=limit_coords_displacement,
                abc_scaler=abc_scaler,
                angle_scaler=angle_scaler,
            )

        sa = SimulatedAnnealingOptimizer(
            evaluate_fn=evaluate_fn,
            proposal_fn=proposal_fn,
            schedule=schedule,
            seed=sa_cfg.get("seed"),
            cache_size=sa_cfg.get("cache_size", 0),
        )

        def _log_step(step_i, _state, current_eval, _best_eval, _stats):
            for key, value in current_eval.get("force_pressure", {}).items():
                force_pressure_history.setdefault(key, []).append(
                    value.detach().cpu().reshape(-1)
                )
            gap_history.append(
                current_eval["bandgap_pred"].detach().cpu().numpy()
                if current_eval["bandgap_pred"] is not None
                else np.zeros((settings_dict["num_batch_crystal"],))
            )
            e_form_history.append(
                current_eval["ef_pred"].detach().cpu().numpy()
                if current_eval["ef_pred"] is not None
                else np.zeros((settings_dict["num_batch_crystal"],))
            )
            t_loss_history.append(
                current_eval["tolerance"].detach().cpu().numpy()
                if current_eval["tolerance"] is not None
                else np.zeros((settings_dict["num_batch_crystal"],))
            )
            element_count_l1_history.append(
                current_eval["element_count_l1_value"].detach().cpu().item()
            )
            if current_eval["per_crystal_loss"] is not None:
                element_count_l1_max_history.append(
                    current_eval["per_crystal_loss"].detach().cpu().max().item()
                )
            else:
                element_count_l1_max_history.append(0.0)

            if time_series_save:
                out = current_eval["output_dict"]
                lattice_mat_history.append(
                    out["lattice_vectors"].detach().cpu().numpy()
                )
                normed_coords_history.append(
                    out["normed_batch_dir_coords"].detach().cpu().numpy()
                )
                coords_history.append(out["batch_dir_coords"].detach().cpu().numpy())
                abc_history.append(out["batch_abc"].detach().cpu().numpy())
                angle_history.append(out["batch_angle"].detach().cpu().numpy())
                elements = [
                    VALID_ELEMENTS98[atom_idx]
                    for atom_idx in torch.argmax(out["normalized_dist"], dim=1)
                ]
                elememts_history.append(elements)
            sa_accept_flags.append(list(_stats.get("accept_flags", [])))
            sample_flags = _stats.get("sample_accept_flags", [])
            if sample_flags:
                sa_sample_accept_flags.append([list(row) for row in sample_flags])

        best_state, sa_history = sa.run(
            initial_state=optimization_targets,
            steps=num_steps,
            trials_per_step=trials_per_step,
            per_sample_accept=sa_per_sample_accept,
            sample_sizes_by_state=(
                sample_sizes_by_state if sa_per_sample_accept else None
            ),
            step_callback=_log_step,
        )
        optimization_targets = best_state
        sa_steps = [h.step for h in sa_history]
        sa_temps = [h.temperature for h in sa_history]
        sa_losses = [h.loss for h in sa_history]
        sa_best = [h.best_loss for h in sa_history]
        sa_accept_rate = [h.accept_rate for h in sa_history]
        if sa_accept_flags:
            sa_accept_flags_tensor = torch.tensor(sa_accept_flags, dtype=torch.int64)
        else:
            sa_accept_flags_tensor = torch.empty(
                (0, trials_per_step), dtype=torch.int64
            )
        if sa_sample_accept_flags:
            sa_sample_accept_flags_tensor = torch.tensor(
                sa_sample_accept_flags, dtype=torch.int64
            )
        else:
            sa_sample_accept_flags_tensor = torch.empty(
                (0, trials_per_step, int(mini_batch_inputs_dict["size"].numel())),
                dtype=torch.int64,
            )
    else:
        for step_i in tqdm(range(num_steps), desc="step", leave=False):
            eval_out = evaluate_loss_crystalformer(
                optimization_targets=optimization_targets,
                mini_batch_inputs_dict=mini_batch_inputs_dict,
                scalers=scalers,
                temperature=dist_temp_sche[step_i],
                gap_loss_func=gap_loss_func,
                tolerance_loss_func=tolerance_loss_func,
                e_form_coef=e_form_coef,
                crystal_system=crystal_system,
                e_form_min=e_form_min,
                bandgap_model=bandgap_model,
                e_form_model=e_form_model,
                predictors=predictors,
                loss_composer=loss_composer,
                primary_metrics=primary_metrics,
                energy_module=energy_module,
                force_pressure_config=settings_dict.get("force_pressure_config"),
                energy_coef=energy_coef,
                energy_every=energy_every,
                step_i=step_i,
                limit_coords_displacement=limit_coords_displacement,
                abc_range=abc_range,
                angle_range=angle_range,
                adding_noise_scale=adding_noise_scale,
                use_element_count_l1_loss=use_element_count_l1_loss,
                element_count_l1_weight=element_count_l1_weight,
                grad_enabled=True,
            )

            bandgap_pred = eval_out["bandgap_pred"]
            ef_pred = eval_out["ef_pred"]
            total_loss = eval_out["total_loss"]
            each_loss = eval_out["each_loss"]
            gap_loss = eval_out["gap_loss"]
            ef_loss = eval_out["ef_loss"]
            tolerance_loss = eval_out["tolerance_loss"]
            tolerance = eval_out["tolerance"]
            element_count_l1_value = eval_out["element_count_l1_value"]
            per_crystal_loss = eval_out["per_crystal_loss"]
            output_dict = eval_out["output_dict"]
            if (
                eval_out["energy_loss"] is not None
                and eval_out["energy_per_atom"] is not None
            ):
                print(
                    f"<info> step {step_i} total_loss={total_loss.item():.6f} "
                    f"energy_loss={eval_out['energy_loss'].item():.6f} "
                    f"energy_per_atom={eval_out['energy_per_atom'].mean().item():.6f}"
                )

            # Update parameters
            total_loss, optimization_targets, additional_info = update_parameters_step(
                optimization_targets=optimization_targets,
                optimization_target_names=optimization_target_names,
                step_i=step_i,
                total_loss=total_loss,
                gap_loss=gap_loss,
                ef_loss=ef_loss,
                tolerance_loss=tolerance_loss,
                each_loss=each_loss,
                total_steps=num_steps,
                learning_rates=learning_rates,
                cry_atom_data=output_dict["batch"].detach(),
                size=mini_batch_inputs_dict["size"].to(device),
                wp_size=mini_batch_inputs_dict["size"].to(device),
                atomic_mask=mini_batch_inputs_dict["atomic_mask"],
                ox_states_used_mask=mini_batch_inputs_dict["ox_states_used_mask"],
                site_ids=mini_batch_inputs_dict["site_ids"].to(device),
                radii_tensor=mini_batch_inputs_dict["radii_tensor"].to(device),
                init_coords=mini_batch_inputs_dict["init_coords"].to(device),
                copy_mutation=copy_mutation,
                angle_optimization=angle_optimization,
                optimizer_lattice=optimizer_lattice,
                optimizer_atom=optimizer_atom,
                optimizer_coords=optimizer_coords,
                scheduler_lattice=scheduler_lattice,
                scheduler_atom=scheduler_atom,
                scheduler_coords=scheduler_coords,
                fixed_inputs=mini_batch_inputs_dict,
                scalers=scalers,
                bandgap_model=bandgap_model,
                e_form_model=e_form_model,
                limit_coords_displacement=limit_coords_displacement,
                gap_loss_func=gap_loss_func,
                tolerance_loss_func=tolerance_loss_func,
                Ef_coef_value=e_form_coef,
                crystal_system=crystal_system,
                e_form_min=e_form_min,
            )
            # update optimizer
            optimizer_atom = additional_info["optimizer_atom"]
            optimizer_lattice = additional_info["optimizer_lattice"]
            optimizer_coords = additional_info["optimizer_coords"]
            scheduler_lattice = additional_info["scheduler_lattice"]
            scheduler_atom = additional_info["scheduler_atom"]
            scheduler_coords = additional_info["scheduler_coords"]

            mini_batch_inputs_dict["batch"] = additional_info["new_cry_atom_data"].to(
                device
            )
            mini_batch_inputs_dict["size"] = additional_info["new_size"].to(device)
            mini_batch_inputs_dict["atomic_mask"] = additional_info["atomic_mask"].to(
                device
            )
            mini_batch_inputs_dict["ox_states_used_mask"] = additional_info[
                "ox_states_used_mask"
            ].to(device)
            mini_batch_inputs_dict["site_ids"] = additional_info["site_ids"].to(device)
            mini_batch_inputs_dict["radii_tensor"] = additional_info["radii_tensor"].to(
                device
            )
            mini_batch_inputs_dict["init_coords"] = additional_info["init_coords"].to(
                device
            )

            scalers["abc_scaler"] = additional_info["abc_scaler"]
            abc_scaler = additional_info["abc_scaler"]

            # 損失の記録
            for key, value in eval_out.get("force_pressure", {}).items():
                force_pressure_history.setdefault(key, []).append(
                    value.detach().cpu().reshape(-1)
                )
            gap_history.append(bandgap_pred.detach().cpu().numpy())
            e_form_history.append(ef_pred.detach().cpu().numpy())
            t_loss_history.append(tolerance.detach().cpu().numpy())
            element_count_l1_history.append(
                element_count_l1_value.detach().cpu().item()
            )
            element_count_l1_max_history.append(
                per_crystal_loss.detach().cpu().max().item()
                if (use_element_count_l1_loss or element_count_l1_weight != 0.0)
                else 0.0
            )

            if time_series_save:
                lattice_mat_history.append(
                    output_dict["lattice_vectors"].detach().cpu().numpy()
                )
                normed_coords_history.append(
                    output_dict["normed_batch_dir_coords"].detach().cpu().numpy()
                )
                coords_history.append(
                    output_dict["batch_dir_coords"].detach().cpu().numpy()
                )
                abc_history.append(output_dict["batch_abc"].detach().cpu().numpy())
                angle_history.append(output_dict["batch_angle"].detach().cpu().numpy())
                elements = [
                    VALID_ELEMENTS98[atom_idx]
                    for atom_idx in torch.argmax(output_dict["normalized_dist"], dim=1)
                ]
                elememts_history.append(elements)

    if optimizer == "sa":
        eval_out = evaluate_loss_crystalformer(
            optimization_targets=optimization_targets,
            mini_batch_inputs_dict=mini_batch_inputs_dict,
            scalers=scalers,
            temperature=dist_temp_sche[-1],
            gap_loss_func=gap_loss_func,
            tolerance_loss_func=tolerance_loss_func,
            e_form_coef=e_form_coef,
            crystal_system=crystal_system,
            e_form_min=e_form_min,
            bandgap_model=bandgap_model,
            e_form_model=e_form_model,
            predictors=predictors,
            loss_composer=loss_composer,
            primary_metrics=primary_metrics,
            energy_module=energy_module,
            force_pressure_config=settings_dict.get("force_pressure_config"),
            energy_coef=energy_coef,
            energy_every=energy_every,
            step_i=num_steps - 1,
            limit_coords_displacement=limit_coords_displacement,
            abc_range=abc_range,
            angle_range=angle_range,
            adding_noise_scale=adding_noise_scale,
            use_element_count_l1_loss=use_element_count_l1_loss,
            element_count_l1_weight=element_count_l1_weight,
            grad_enabled=False,
        )
        output_dict = eval_out["output_dict"]

    # End of optimization
    (
        scaled_batch_abc,
        scaled_batch_angle,
        batch_dir_coords,
        atomic_distribution,
        ox_mask_learnable_tensor_per_crystal,
    ) = optimization_targets
    print(
        f"The number of invalid crystral structures : {(torch.isnan(bandgap_pred)).detach().cpu().numpy().sum()}/{len(bandgap_pred)}"
    )
    optimized_mini_batch_inputs_dict = mini_batch_inputs_dict | output_dict
    optimized_mini_batch_inputs_dict["atomic_distribution"] = atomic_distribution
    optimized_mini_batch_inputs_dict["scaled_batch_abc"] = scaled_batch_abc
    optimized_mini_batch_inputs_dict["scaled_batch_angle"] = scaled_batch_angle
    optimized_mini_batch_inputs_dict["ox_mask_learnable_tensor_per_crystal"] = (
        ox_mask_learnable_tensor_per_crystal
    )
    optimized_mini_batch_inputs_dict["batch_dir_coords"] = batch_dir_coords
    for key, values in force_pressure_history.items():
        optimized_mini_batch_inputs_dict[f"{key}_history"] = torch.stack(values, dim=1)
    optimized_mini_batch_inputs_dict["gap_history"] = np.transpose(
        np.array(gap_history).reshape(len(gap_history), -1), (1, 0)
    )
    optimized_mini_batch_inputs_dict["ef_history"] = np.transpose(
        np.array(e_form_history).reshape(len(e_form_history), -1), (1, 0)
    )
    optimized_mini_batch_inputs_dict["t_history"] = np.transpose(
        np.array(t_loss_history).reshape(len(t_loss_history), -1), (1, 0)
    )
    optimized_mini_batch_inputs_dict["element_count_l1_history"] = (
        element_count_l1_history
    )
    optimized_mini_batch_inputs_dict["element_count_l1_max_history"] = (
        element_count_l1_max_history
    )
    if optimizer == "sa":
        optimized_mini_batch_inputs_dict["sa_step"] = torch.tensor(sa_steps)
        optimized_mini_batch_inputs_dict["sa_temperature"] = torch.tensor(sa_temps)
        optimized_mini_batch_inputs_dict["sa_loss"] = torch.tensor(sa_losses)
        optimized_mini_batch_inputs_dict["sa_best_loss"] = torch.tensor(sa_best)
        optimized_mini_batch_inputs_dict["sa_accept_rate"] = torch.tensor(
            sa_accept_rate
        )
        optimized_mini_batch_inputs_dict["sa_accept_flags"] = sa_accept_flags_tensor
        if sa_per_sample_accept:
            optimized_mini_batch_inputs_dict["sa_sample_accept_flags"] = (
                sa_sample_accept_flags_tensor
            )
        # Backward-compatibility alias for existing downstream readers.
        optimized_mini_batch_inputs_dict["sa_accepted"] = (
            optimized_mini_batch_inputs_dict["sa_accept_rate"].clone()
        )
    time_series = [
        lattice_mat_history,
        coords_history,
        normed_coords_history,
        abc_history,
        angle_history,
        elememts_history,
    ]

    return (optimized_mini_batch_inputs_dict, time_series, scalers)
