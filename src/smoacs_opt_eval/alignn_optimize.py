from typing import Any, Dict, List, Optional, Tuple

import json
import time
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.optimization.optimizer_utils import define_optimizer_and_scheduler
from src.optimizers.proposals import propose_state
from src.optimizers.simulated_annealing import (
    SimulatedAnnealingOptimizer,
    geometric_schedule_with_endpoints,
)
from src.smoacs_opt_eval.alignn_graph_update import graph_parameter_update
from src.smoacs_opt_eval.copy_and_mutate import parse_evolution_parameters
from src.smoacs_opt_eval.loss_eval import evaluate_loss_alignn
from src.smoacs_opt_eval.step_update_utils import update_parameters_step
from src.utils.common import create_element_list
from src.utils.poscar_utils import save_poscar
from src.spg_wp.wyckoff_expander import WyckoffExpander
from src.spg_wp.lattice_params_control import (
    create_equal_length_copy_mask,
    create_special_angle_fix_mask,
)


def optimize_solution_ALIGNN(
    settings_dict: Dict[str, Any],
    mini_batch_inputs_dict: Dict[str, Any],
    learning_rates: List[float],
    learning_rate_cycle: List[int],
    num_steps: int,
    dist_temp_sche: List[float],
    e_form_coef: float,
    gap_loss_func: torch.nn.Module,
    bandgap_model: Optional[torch.nn.Module],
    e_form_model: Optional[torch.nn.Module],
    scalers: dict,
    adding_noise_scale: Optional[float],
    e_form_min: Optional[float],
    limit_coords_displacement: Optional[float],
    wyckoff_expander: Optional[WyckoffExpander],
    crystal_system: Optional[str] = None,
    angle_optimization: bool = True,
    tolerance_loss_func=Optional[torch.nn.Module],
    device: str = "cuda",
    time_series_save: bool = True,
    abc_equal_eps: float = 1e-5,
    angle_fix_eps: float = 1e-3,
    predictors: Optional[Dict[str, Any]] = None,
    loss_composer: Optional[Any] = None,
    primary_metrics: Optional[Dict[str, str]] = None,
    energy_module: Optional[Any] = None,
    energy_coef: float = 0.0,
    energy_every: int = 1,
    use_element_count_l1_loss: bool = False,
    element_count_l1_weight: float = 1.0,
    optimizer: str = "gradient",
    timing_log_path: Optional[str] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    Performs optimization of a batch of candidate crystal structures using ALIGNN.

    Args:
        settings_dict: Dictionary containing configuration settings.
        mini_batch_inputs_dict: Dictionary containing batch-specific model inputs.
        learning_rates: List of learning rates for lattice, atom, and coordinate optimization.
        learning_rate_cycle: List of cycle lengths for each optimizer.
        num_steps: Number of optimization steps.
        dist_temp_sche: Temperature schedule for softmax during optimization.
        Ef_coef_schedule: Schedule for scaling formation energy loss.
        step_of_x_opt_stop: Step at which optimization of atomic distribution stops.
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
    superhard_mode = settings_dict.get("superhard_cfg") is not None
    if hasattr(abc_scaler, "base_length"):
        abc_scaler.device = device
        abc_scaler.base_length = abc_scaler.base_length.to(device)
        if hasattr(abc_scaler, "init_batch_abc") and torch.is_tensor(
            abc_scaler.init_batch_abc
        ):
            abc_scaler.init_batch_abc = abc_scaler.init_batch_abc.to(device)
    device = torch.device(device)
    use_cuda = device.type == "cuda" and torch.cuda.is_available()
    if optimizer not in {"gradient", "sa", "grad_sa_hybrid"}:
        raise ValueError(f"Unknown optimizer: {optimizer}")
    if optimizer == "grad_sa_hybrid":
        grad_sa_hybrid_cfg = settings_dict.get("grad_sa_hybrid_config")
        if grad_sa_hybrid_cfg is None:
            raise ValueError(
                "grad_sa_hybrid optimizer requires settings_dict['grad_sa_hybrid_config']"
            )
        step_key_values = {
            key: int(grad_sa_hybrid_cfg[key])
            for key in (
                "grad_only_steps",
                "sa_start_step",
                "grad_sa_hybrid_grad_steps",
            )
            if key in grad_sa_hybrid_cfg
        }
        if step_key_values:
            if len(set(step_key_values.values())) != 1:
                details = ", ".join(
                    f"{key}={value}" for key, value in step_key_values.items()
                )
                raise ValueError(
                    "grad_sa_hybrid step keys must match when multiple aliases "
                    f"are provided: {details}"
                )
            grad_only_steps = next(iter(step_key_values.values()))
        else:
            grad_only_steps = num_steps // 2
        if grad_only_steps < 0:
            raise ValueError(
                f"grad_sa_hybrid grad_only_steps must be >= 0, got {grad_only_steps}"
            )
        if grad_only_steps >= num_steps:
            raise ValueError(
                f"grad_sa_hybrid grad_only_steps must be < num_steps, got "
                f"grad_only_steps={grad_only_steps}, num_steps={num_steps}"
            )
    elif optimizer == "gradient":
        grad_only_steps = num_steps
    else:
        grad_only_steps = 0

    run_gradient = optimizer in {"gradient", "grad_sa_hybrid"}
    sa_total_steps = num_steps if optimizer == "sa" else (num_steps - grad_only_steps)
    run_sa = sa_total_steps > 0

    requires_grad = optimizer != "sa"
    atomic_distribution = nn.Parameter(
        mini_batch_inputs_dict["atomic_distribution"].clone().to(device),
        requires_grad=requires_grad,
    )
    scaled_batch_abc = nn.Parameter(
        abc_scaler.scale(mini_batch_inputs_dict["batch_abc"].clone().to(device)),
        requires_grad=requires_grad,
    )

    scaled_batch_angle = nn.Parameter(
        angle_scaler.scale(mini_batch_inputs_dict["batch_angle"].clone().to(device)),
        requires_grad=requires_grad,
    )
    ox_mask_learnable_tensor_per_crystal = nn.Parameter(
        mini_batch_inputs_dict["ox_mask_learnable_tensor_per_crystal"]
        .to(device)
        .clone(),
        requires_grad=requires_grad,
    )
    batch_dir_coords = nn.Parameter(
        mini_batch_inputs_dict["batch_dir_coords"].to(device).clone(),
        requires_grad=requires_grad,
    )
    abc_range = settings_dict["abc_range"]
    angle_range = settings_dict["angle_range"]

    if run_gradient:
        # optimizer and scheduler (gradient phase only)
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
            num_steps=max(num_steps, 1),
            lattice_cycle=learning_rate_cycle[0],
            atom_cycle=learning_rate_cycle[1],
            coords_cycle=learning_rate_cycle[2],
        )
    else:
        optimizer_lattice = None
        optimizer_atom = None
        optimizer_coords = None
        scheduler_lattice = None
        scheduler_atom = None
        scheduler_coords = None

    # initialzation of history
    lattice_mat_history, coords_history, normed_coords_history, elememts_history = (
        [],
        [],
        [],
        [],
    )
    abc_history, angle_history = [], []
    gap_history, e_form_history, t_loss_history = [], [], []
    element_count_l1_history, element_count_l1_max_history = [], []
    VALID_ELEMENTS98 = create_element_list()
    lattice_lr_history, atom_lr_history, coords_lr_history = [], [], []
    step_times = []
    step_peak_bytes = []
    step_batch_atoms = []
    step_batch_wp_atoms = []
    step_graph_updated = []

    def _append_timing_event(payload: Dict[str, Any]) -> None:
        if not timing_log_path:
            return
        with open(timing_log_path, "a", encoding="ascii") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")

    # (English) After copy_and_mutate, it is necessary to perform the ALIGNN graph update_step.
    copy_mutation = settings_dict["copy_mutation"]
    parse_evolution_parameters_dict = parse_evolution_parameters(copy_mutation)
    if parse_evolution_parameters_dict["steps_cm"][0] != "None":
        additional_alignn_update_steps = [
            int(step) + 1 for step in parse_evolution_parameters_dict["steps_cm"]
        ]
        settings_dict["ALIGNN_update_step"].extend(additional_alignn_update_steps)

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

    # create masks for enforcing lattice constraints
    if wyckoff_expander is not None:
        abc_copy_mask = create_equal_length_copy_mask(
            scaled_batch_abc=scaled_batch_abc,
            eps=abc_equal_eps,
        )
        batch_angle = angle_scaler.rescale(scaled_batch_angle)
        angle_mask, fixed_angle_tensor = create_special_angle_fix_mask(
            batch_angle=batch_angle,
            eps=angle_fix_eps,
            special_angles=(60.0, 90.0, 120.0),
        )
        wyckoff_expander.abc_copy_mask = abc_copy_mask.to(wyckoff_expander.device)
        wyckoff_expander.angle_mask = angle_mask.to(wyckoff_expander.device)
        wyckoff_expander.fixed_angle_tensor = fixed_angle_tensor.to(
            wyckoff_expander.device
        )

    output_dict = None
    sa_step_values: List[int] = []
    sa_temps: List[float] = []
    sa_losses: List[float] = []
    sa_best: List[float] = []
    sa_accept_rate: List[float] = []
    sa_accept_flags_tensor = torch.empty((0, 0), dtype=torch.int64)
    sa_sample_accept_flags_tensor = torch.empty((0, 0, 0), dtype=torch.int64)
    sa_cfg = None
    schedule = None
    trials_per_step = 1
    sa_per_sample_accept = False
    move_probs: Optional[Dict[str, float]] = None
    move_scales: Optional[Dict[str, float]] = None
    if run_sa:
        sa_cfg = settings_dict.get("sa_config")
        if sa_cfg is None:
            raise ValueError("SA phase requires settings_dict['sa_config']")
        schedule = geometric_schedule_with_endpoints(
            sa_cfg.get("t0", 1.0),
            sa_cfg.get("tmin", 1e-3),
            sa_total_steps,
        )
        trials_per_step = int(sa_cfg.get("trials_per_step", 1))
        if angle_optimization:
            move_probs = {
                "coords": 1.0 / 3.0,
                "lattice": 1.0 / 3.0,
                "angle": 1.0 / 3.0,
            }
        else:
            move_probs = {"coords": 0.5, "lattice": 0.5}
        move_scales = {
            "coords": float(sa_cfg.get("coords_sigma", 1e-3)),
            "lattice": float(sa_cfg.get("lattice_sigma", 1e-3)),
            "angle": float(sa_cfg.get("angle_sigma", 1e-3)),
        }
        sa_accept_flags_tensor = torch.empty((0, trials_per_step), dtype=torch.int64)
        sa_per_sample_accept = bool(sa_cfg.get("per_sample_accept", True))
        sa_sample_accept_flags_tensor = torch.empty(
            (0, trials_per_step, int(mini_batch_inputs_dict["size"].numel())),
            dtype=torch.int64,
        )

    def _sa_sample_sizes_by_state() -> List[Optional[torch.Tensor]]:
        def _sum_or_missing(key: str) -> str:
            sizes = mini_batch_inputs_dict.get(key)
            if sizes is None:
                return "missing"
            return str(int(sizes.to(torch.long).sum().item()))

        def _sizes_for_rows(row_count: int) -> torch.Tensor:
            for key in ("wp_size", "size"):
                sizes = mini_batch_inputs_dict.get(key)
                if sizes is None:
                    continue
                sizes = sizes.to(device=device, dtype=torch.long)
                if int(sizes.sum().item()) == row_count:
                    return sizes
            raise ValueError(
                "Could not map SA state rows to samples: "
                f"row_count={row_count}, size_sum="
                f"{int(mini_batch_inputs_dict['size'].to(torch.long).sum().item())}, "
                f"wp_size_sum={_sum_or_missing('wp_size')}"
            )

        return [
            None,
            None,
            _sizes_for_rows(int(optimization_targets[2].shape[0])),
            _sizes_for_rows(int(optimization_targets[3].shape[0])),
            None,
        ]

    if run_gradient:
        for step_i in tqdm(range(num_steps), desc="Optimizing", leave=False):
            global_step_i = step_i
            step_start = time.perf_counter()
            if use_cuda:
                torch.cuda.reset_peak_memory_stats(device)
            ### Update graph and inputs
            graph_updated = False
            if (
                global_step_i in settings_dict["ALIGNN_update_step"]
                and global_step_i != 0
            ):
                print(f"<info> graph update at step {global_step_i}")
                mini_batch_inputs_dict = graph_parameter_update(
                    optimization_target_names=optimization_target_names,
                    optimization_targets=optimization_targets,
                    mini_batch_inputs_dict=mini_batch_inputs_dict,
                    output_dict=output_dict,
                    num_batch_crystal=settings_dict["num_batch_crystal"],
                    wyckoff_expander=wyckoff_expander,
                    num_workers=settings_dict.get("num_workers", 4),
                )
                graph_updated = True

            eval_out = evaluate_loss_alignn(
                optimization_targets=optimization_targets,
                mini_batch_inputs_dict=mini_batch_inputs_dict,
                scalers=scalers,
                temperature=dist_temp_sche[global_step_i],
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
                energy_coef=energy_coef,
                energy_every=energy_every,
                step_i=global_step_i,
                limit_coords_displacement=limit_coords_displacement,
                abc_range=abc_range,
                angle_range=angle_range,
                adding_noise_scale=adding_noise_scale,
                wyckoff_expander=wyckoff_expander,
                use_element_count_l1_loss=use_element_count_l1_loss,
                element_count_l1_weight=element_count_l1_weight,
                grad_enabled=True,
                superhard_mode=superhard_mode,
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
                    f"<info> step {global_step_i} total_loss={total_loss.item():.6f} "
                    f"energy_loss={eval_out['energy_loss'].item():.6f} "
                    f"energy_per_atom={eval_out['energy_per_atom'].mean().item():.6f}"
                )

            if global_step_i == 0:
                # save initial poscar
                save_poscar(
                    mini_batch_inputs_dict | output_dict,
                    "init_poscars",
                    wyckoff_expander=wyckoff_expander,
                )

            size = mini_batch_inputs_dict["size"].to(device)
            wp_size = mini_batch_inputs_dict["wp_size"].to(device)
            pre_batch_atoms = int(size.sum().item())
            pre_batch_wp_atoms = int(wp_size.to(torch.long).sum().item())
            if timing_log_path:
                step_batch_atoms.append(int(size.sum().item()))
                step_batch_wp_atoms.append(int(wp_size.sum().item()))
                step_graph_updated.append(bool(graph_updated))
                _append_timing_event(
                    {
                        "type": "opt_step_start",
                        "step_i": int(global_step_i),
                        "batch_crystals": int(len(size)),
                        "batch_atoms": int(size.sum().item()),
                        "batch_wp_atoms": int(wp_size.sum().item()),
                        "graph_updated": bool(graph_updated),
                    }
                )
            cry_atom_data = torch.repeat_interleave(
                torch.arange(len(size), device=size.device), size
            )

            total_loss, optimization_targets, additional_info = update_parameters_step(
                optimization_targets=optimization_targets,
                optimization_target_names=optimization_target_names,
                step_i=global_step_i,
                total_loss=total_loss,
                gap_loss=gap_loss,
                ef_loss=ef_loss,
                tolerance_loss=tolerance_loss,
                each_loss=each_loss,
                total_steps=num_steps,
                learning_rates=learning_rates,
                cry_atom_data=cry_atom_data,
                size=size,
                wp_size=wp_size,
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
            mini_batch_inputs_dict["wp_size"] = additional_info["new_wp_size"].to(
                device
            )
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
            record_eval = eval_out
            post_batch_atoms = int(
                mini_batch_inputs_dict["size"].to(torch.long).sum().item()
            )
            post_batch_wp_atoms = int(
                mini_batch_inputs_dict["wp_size"].to(torch.long).sum().item()
            )
            size_or_wp_changed = (
                post_batch_atoms != pre_batch_atoms
                or post_batch_wp_atoms != pre_batch_wp_atoms
            )

            def _sync_graph_immediately(reason: str) -> None:
                nonlocal mini_batch_inputs_dict, graph_updated
                if output_dict is None:
                    raise RuntimeError(
                        f"graph sync requested before output_dict is available "
                        f"(step={global_step_i}, reason={reason})"
                    )
                print(
                    f"<guard> immediate graph sync at step {global_step_i} "
                    f"(reason={reason})"
                )
                mini_batch_inputs_dict = graph_parameter_update(
                    optimization_target_names=optimization_target_names,
                    optimization_targets=optimization_targets,
                    mini_batch_inputs_dict=mini_batch_inputs_dict,
                    output_dict=output_dict,
                    num_batch_crystal=settings_dict["num_batch_crystal"],
                    wyckoff_expander=wyckoff_expander,
                    num_workers=settings_dict.get("num_workers", 4),
                )
                graph_updated = True
                if timing_log_path:
                    _append_timing_event(
                        {
                            "type": "graph_sync_immediate",
                            "step_i": int(global_step_i),
                            "reason": reason,
                            "batch_atoms": int(
                                mini_batch_inputs_dict["size"]
                                .to(torch.long)
                                .sum()
                                .item()
                            ),
                            "batch_wp_atoms": int(
                                mini_batch_inputs_dict["wp_size"]
                                .to(torch.long)
                                .sum()
                                .item()
                            ),
                            "graph_nodes": int(
                                mini_batch_inputs_dict["g"].number_of_nodes()
                            ),
                        }
                    )

            if optimizer == "grad_sa_hybrid" and global_step_i >= grad_only_steps:
                if size_or_wp_changed:
                    _sync_graph_immediately("copy_mutation_size_change")

                expected_nodes = int(
                    mini_batch_inputs_dict["size"].to(torch.long).sum().item()
                )
                graph_nodes = int(mini_batch_inputs_dict["g"].number_of_nodes())
                if graph_nodes != expected_nodes:
                    print(
                        f"<guard> node mismatch before SA at step {global_step_i}: "
                        f"graph_nodes={graph_nodes}, expected={expected_nodes}"
                    )
                    if timing_log_path:
                        _append_timing_event(
                            {
                                "type": "graph_guard_mismatch",
                                "step_i": int(global_step_i),
                                "graph_nodes": int(graph_nodes),
                                "expected_nodes": int(expected_nodes),
                            }
                        )
                    _sync_graph_immediately("pre_sa_guard_mismatch")
                    graph_nodes = int(mini_batch_inputs_dict["g"].number_of_nodes())
                    expected_nodes = int(
                        mini_batch_inputs_dict["size"].to(torch.long).sum().item()
                    )
                    if graph_nodes != expected_nodes:
                        raise RuntimeError(
                            "graph-size mismatch persists after forced sync: "
                            f"step={global_step_i}, graph_nodes={graph_nodes}, "
                            f"expected_nodes={expected_nodes}"
                        )

                sa_local_step = global_step_i - grad_only_steps
                assert schedule is not None
                assert move_probs is not None
                assert move_scales is not None
                sa_temperature = schedule(sa_local_step)
                sa_capture: Dict[str, Any] = {}

                def evaluate_fn(state, _sa_step_i):
                    return evaluate_loss_alignn(
                        optimization_targets=state,
                        mini_batch_inputs_dict=mini_batch_inputs_dict,
                        scalers=scalers,
                        temperature=dist_temp_sche[global_step_i],
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
                        energy_coef=energy_coef,
                        energy_every=energy_every,
                        step_i=global_step_i,
                        limit_coords_displacement=limit_coords_displacement,
                        abc_range=abc_range,
                        angle_range=angle_range,
                        adding_noise_scale=adding_noise_scale,
                        wyckoff_expander=wyckoff_expander,
                        use_element_count_l1_loss=use_element_count_l1_loss,
                        element_count_l1_weight=element_count_l1_weight,
                        grad_enabled=False,
                        superhard_mode=superhard_mode,
                    )

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
                    schedule=lambda _step: sa_temperature,
                    seed=sa_cfg.get("seed") if sa_cfg is not None else None,
                    cache_size=sa_cfg.get("cache_size", 0) if sa_cfg is not None else 0,
                )

                def _log_grad_sa_hybrid_sa(
                    _sa_step_i, state, current_eval, _best_eval, stats
                ):
                    sa_capture["state"] = [tensor.detach().clone() for tensor in state]
                    sa_capture["current_eval"] = current_eval
                    sa_capture["accept_flags"] = list(stats.get("accept_flags", []))
                    sa_capture["accept_rate"] = float(stats.get("accept_rate", 0.0))
                    sa_capture["sample_accept_flags"] = [
                        list(row) for row in stats.get("sample_accept_flags", [])
                    ]

                _best_state, sa_history = sa.run(
                    initial_state=optimization_targets,
                    steps=1,
                    trials_per_step=trials_per_step,
                    per_sample_accept=sa_per_sample_accept,
                    sample_sizes_by_state=(
                        _sa_sample_sizes_by_state() if sa_per_sample_accept else None
                    ),
                    step_callback=_log_grad_sa_hybrid_sa,
                )
                if "state" not in sa_capture or "current_eval" not in sa_capture:
                    raise RuntimeError(
                        "grad_sa_hybrid SA did not produce callback output."
                    )
                with torch.no_grad():
                    for param, sa_value in zip(
                        optimization_targets, sa_capture["state"]
                    ):
                        param.copy_(sa_value)

                output_dict = sa_capture["current_eval"]["output_dict"]
                record_eval = sa_capture["current_eval"]
                sa_step_values.append(global_step_i)
                sa_temps.append(sa_temperature)
                sa_losses.append(sa_history[0].loss)
                sa_best.append(sa_history[0].best_loss)
                sa_accept_rate.append(
                    float(sa_capture.get("accept_rate", sa_history[0].accept_rate))
                )
                sa_flags = sa_capture.get("accept_flags", [])
                if not sa_flags:
                    sa_flags = [0] * trials_per_step
                sa_accept_flags_tensor = torch.cat(
                    [
                        sa_accept_flags_tensor,
                        torch.tensor([sa_flags], dtype=torch.int64),
                    ],
                    dim=0,
                )
                sa_sample_flags = sa_capture.get("sample_accept_flags", [])
                if sa_sample_flags:
                    sa_sample_accept_flags_tensor = torch.cat(
                        [
                            sa_sample_accept_flags_tensor,
                            torch.tensor([sa_sample_flags], dtype=torch.int64),
                        ],
                        dim=0,
                    )

            # 損失の記録
            bandgap_pred = record_eval["bandgap_pred"]
            ef_pred = record_eval["ef_pred"]
            tolerance = record_eval["tolerance"]
            element_count_l1_value = record_eval["element_count_l1_value"]
            per_crystal_loss = record_eval["per_crystal_loss"]
            gap_history.append(
                bandgap_pred.detach().cpu().numpy()
                if bandgap_pred is not None
                else np.zeros((settings_dict["num_batch_crystal"],))
            )
            e_form_history.append(
                ef_pred.detach().cpu().numpy()
                if ef_pred is not None
                else np.zeros((settings_dict["num_batch_crystal"],))
            )
            t_loss_history.append(
                tolerance.detach().cpu().numpy()
                if tolerance is not None
                else np.zeros((settings_dict["num_batch_crystal"],))
            )
            element_count_l1_history.append(
                element_count_l1_value.detach().cpu().item()
            )
            element_count_l1_max_history.append(
                per_crystal_loss.detach().cpu().max().item()
                if (
                    per_crystal_loss is not None
                    and use_element_count_l1_loss
                )
                else 0.0
            )

            # learning rateの記録
            lattice_lr_history.append(optimizer_lattice.param_groups[0]["lr"])
            atom_lr_history.append(optimizer_atom.param_groups[0]["lr"])
            coords_lr_history.append(optimizer_coords.param_groups[0]["lr"])

            if time_series_save:
                output_dict = record_eval["output_dict"]
                lattice_mat_history.append(
                    output_dict["lattice_vectors"].detach().cpu().numpy()
                )
                normed_coords_history.append(
                    output_dict["normed_batch_dir_coords"].detach().cpu().numpy()
                )
                coords_history.append(optimization_targets[2].detach().cpu().numpy())
                abc_history.append(output_dict["batch_abc"].detach().cpu().numpy())
                angle_history.append(output_dict["batch_angle"].detach().cpu().numpy())
                elements = [
                    VALID_ELEMENTS98[atom_idx]
                    for atom_idx in torch.argmax(output_dict["normalized_dist"], dim=1)
                ]
                elememts_history.append(elements)
            step_times.append(time.perf_counter() - step_start)
            if use_cuda:
                step_peak_bytes.append(torch.cuda.max_memory_allocated(device))
            if timing_log_path:
                _append_timing_event(
                    {
                        "type": "opt_step_end",
                        "step_i": int(global_step_i),
                        "step_time_sec": float(step_times[-1]),
                        "gpu_peak_bytes": (
                            int(step_peak_bytes[-1]) if step_peak_bytes else 0
                        ),
                    }
                )

    if optimizer == "sa":
        assert schedule is not None
        assert move_probs is not None
        assert move_scales is not None
        assert sa_cfg is not None
        last_output = {"output_dict": output_dict}
        sa_accept_flags: List[List[int]] = []
        sa_sample_accept_flags: List[List[List[int]]] = []

        def evaluate_fn(state, sa_step_i):
            global_step_i = sa_step_i
            if (
                global_step_i in settings_dict["ALIGNN_update_step"]
                and global_step_i != 0
                and last_output["output_dict"] is not None
            ):
                print(f"<info> graph update at step {global_step_i}")
                mini_batch_inputs_dict.update(
                    graph_parameter_update(
                        optimization_target_names=optimization_target_names,
                        optimization_targets=state,
                        mini_batch_inputs_dict=mini_batch_inputs_dict,
                        output_dict=last_output["output_dict"],
                        num_batch_crystal=settings_dict["num_batch_crystal"],
                        wyckoff_expander=wyckoff_expander,
                        num_workers=settings_dict.get("num_workers", 4),
                    )
                )
            eval_out = evaluate_loss_alignn(
                optimization_targets=state,
                mini_batch_inputs_dict=mini_batch_inputs_dict,
                scalers=scalers,
                temperature=dist_temp_sche[global_step_i],
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
                energy_coef=energy_coef,
                energy_every=energy_every,
                step_i=global_step_i,
                limit_coords_displacement=limit_coords_displacement,
                abc_range=abc_range,
                angle_range=angle_range,
                adding_noise_scale=adding_noise_scale,
                wyckoff_expander=wyckoff_expander,
                use_element_count_l1_loss=use_element_count_l1_loss,
                element_count_l1_weight=element_count_l1_weight,
                grad_enabled=False,
                superhard_mode=superhard_mode,
            )
            last_output["output_dict"] = eval_out["output_dict"]
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

        def _log_step(sa_step_i, state, current_eval, _best_eval, stats):
            step_start = time.perf_counter()
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

            lattice_lr_history.append(learning_rates[0])
            atom_lr_history.append(learning_rates[1])
            coords_lr_history.append(learning_rates[2])

            if time_series_save:
                out = current_eval["output_dict"]
                lattice_mat_history.append(
                    out["lattice_vectors"].detach().cpu().numpy()
                )
                normed_coords_history.append(
                    out["normed_batch_dir_coords"].detach().cpu().numpy()
                )
                coords_history.append(state[2].detach().cpu().numpy())
                abc_history.append(out["batch_abc"].detach().cpu().numpy())
                angle_history.append(out["batch_angle"].detach().cpu().numpy())
                elements = [
                    VALID_ELEMENTS98[atom_idx]
                    for atom_idx in torch.argmax(out["normalized_dist"], dim=1)
                ]
                elememts_history.append(elements)
            sa_accept_flags.append(list(stats.get("accept_flags", [])))
            sample_flags = stats.get("sample_accept_flags", [])
            if sample_flags:
                sa_sample_accept_flags.append([list(row) for row in sample_flags])
            step_times.append(time.perf_counter() - step_start)
            if use_cuda:
                step_peak_bytes.append(torch.cuda.max_memory_allocated(device))

        best_state, sa_history = sa.run(
            initial_state=optimization_targets,
            steps=sa_total_steps,
            trials_per_step=trials_per_step,
            per_sample_accept=sa_per_sample_accept,
            sample_sizes_by_state=(
                _sa_sample_sizes_by_state() if sa_per_sample_accept else None
            ),
            step_callback=_log_step,
        )
        optimization_targets = best_state
        sa_step_values = [h.step for h in sa_history]
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

    if optimizer == "sa":
        final_step_i = num_steps - 1
        eval_out = evaluate_loss_alignn(
            optimization_targets=optimization_targets,
            mini_batch_inputs_dict=mini_batch_inputs_dict,
            scalers=scalers,
            temperature=dist_temp_sche[final_step_i],
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
            energy_coef=energy_coef,
            energy_every=energy_every,
            step_i=final_step_i,
            limit_coords_displacement=limit_coords_displacement,
            abc_range=abc_range,
            angle_range=angle_range,
            adding_noise_scale=adding_noise_scale,
            wyckoff_expander=wyckoff_expander,
            use_element_count_l1_loss=use_element_count_l1_loss,
            element_count_l1_weight=element_count_l1_weight,
            grad_enabled=False,
            superhard_mode=superhard_mode,
        )
        output_dict = eval_out["output_dict"]

    # End of optimization
    [
        scaled_batch_abc,
        scaled_batch_angle,
        batch_dir_coords,
        atomic_distribution,
        ox_mask_learnable_tensor_per_crystal,
    ] = optimization_targets

    optimized_mini_batch_inputs_dict = mini_batch_inputs_dict | output_dict
    optimized_mini_batch_inputs_dict["atomic_distribution"] = atomic_distribution
    optimized_mini_batch_inputs_dict["scaled_batch_abc"] = scaled_batch_abc
    optimized_mini_batch_inputs_dict["scaled_batch_angle"] = scaled_batch_angle
    optimized_mini_batch_inputs_dict["ox_mask_learnable_tensor_per_crystal"] = (
        ox_mask_learnable_tensor_per_crystal
    )
    optimized_mini_batch_inputs_dict["batch_dir_coords"] = batch_dir_coords
    gap_arr = np.squeeze(np.array(gap_history))
    ef_arr = np.squeeze(np.array(e_form_history))
    t_arr = np.squeeze(np.array(t_loss_history))
    if gap_arr.ndim == 1:
        gap_arr = gap_arr[None, :]
    else:
        gap_arr = np.transpose(gap_arr, (1, 0))
    if ef_arr.ndim == 1:
        ef_arr = ef_arr[None, :]
    else:
        ef_arr = np.transpose(ef_arr, (1, 0))
    if t_arr.ndim == 1:
        t_arr = t_arr[None, :]
    else:
        t_arr = np.transpose(t_arr, (1, 0))
    optimized_mini_batch_inputs_dict["gap_history"] = gap_arr
    optimized_mini_batch_inputs_dict["ef_history"] = ef_arr
    optimized_mini_batch_inputs_dict["t_history"] = t_arr
    optimized_mini_batch_inputs_dict["element_count_l1_history"] = (
        element_count_l1_history
    )
    optimized_mini_batch_inputs_dict["element_count_l1_max_history"] = (
        element_count_l1_max_history
    )
    optimized_mini_batch_inputs_dict["lattice_lr_history"] = lattice_lr_history
    optimized_mini_batch_inputs_dict["atom_lr_history"] = atom_lr_history
    optimized_mini_batch_inputs_dict["coords_lr_history"] = coords_lr_history

    # for graph update
    optimized_mini_batch_inputs_dict["optimization_targets"] = optimization_targets
    optimized_mini_batch_inputs_dict["optimization_target_names"] = (
        optimization_target_names
    )
    if run_sa:
        optimized_mini_batch_inputs_dict["sa_step"] = torch.tensor(sa_step_values)
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
    if optimizer == "grad_sa_hybrid":
        optimized_mini_batch_inputs_dict["sa_start_step"] = torch.tensor(
            [grad_only_steps], dtype=torch.int64
        )
        optimized_mini_batch_inputs_dict["grad_sa_hybrid_grad_steps"] = torch.tensor(
            [grad_only_steps], dtype=torch.int64
        )
        optimized_mini_batch_inputs_dict["grad_sa_hybrid_sa_steps"] = torch.tensor(
            [num_steps - grad_only_steps], dtype=torch.int64
        )

    time_series = [
        lattice_mat_history,
        coords_history,
        normed_coords_history,
        abc_history,
        angle_history,
        elememts_history,
    ]

    if timing_log_path:
        step_time_arr = np.array(step_times, dtype=float)
        summary = {
            "type": "opt_timing",
            "optimizer": optimizer,
            "num_steps": int(num_steps),
            "num_batch_crystal": int(settings_dict.get("num_batch_crystal", 0)),
            "energy_coef": float(energy_coef),
            "energy_every": int(energy_every),
            "step_time_mean": (
                float(step_time_arr.mean()) if step_time_arr.size else 0.0
            ),
            "step_time_median": (
                float(np.median(step_time_arr)) if step_time_arr.size else 0.0
            ),
            "step_time_p90": (
                float(np.percentile(step_time_arr, 90)) if step_time_arr.size else 0.0
            ),
            "step_time_p95": (
                float(np.percentile(step_time_arr, 95)) if step_time_arr.size else 0.0
            ),
            "step_time_min": float(step_time_arr.min()) if step_time_arr.size else 0.0,
            "step_time_max": float(step_time_arr.max()) if step_time_arr.size else 0.0,
            "step_times": step_times,
            "gpu_peak_bytes_max": int(max(step_peak_bytes)) if step_peak_bytes else 0,
            "gpu_peak_bytes_by_step": step_peak_bytes,
            "step_batch_atoms": step_batch_atoms,
            "step_batch_wp_atoms": step_batch_wp_atoms,
            "step_graph_updated": step_graph_updated,
        }
        with open(timing_log_path, "a", encoding="ascii") as f:
            f.write(json.dumps(summary, ensure_ascii=True) + "\n")

    return (optimized_mini_batch_inputs_dict, time_series, scalers)
