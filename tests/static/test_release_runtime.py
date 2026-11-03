"""Regression tests for the interfaces needed by the release commands."""

import io
from pathlib import Path

import numpy as np
import pytest
import torch

from src.utils.oxidation import elec_neutral_all_common, elec_neutral_all_common_loose


@pytest.mark.parametrize(
    "sampler,cation",
    [(elec_neutral_all_common, "Na"), (elec_neutral_all_common_loose, "Li")],
)
@pytest.mark.parametrize("limit", [1, 100])
def test_neutral_sampler_keeps_site_order_and_caps_unique_samples(
    sampler, cation, limit
):
    # SMACT also allows Na(-1); LiCl has one solution even in loose mode.
    states, elements = sampler([cation, "Cl"], [[1], [1]], max_ox_states=limit)
    assert elements == [cation, "Cl"]
    assert states == [(1, -1)]
    states, elements = sampler(["Cl", cation], [[1], [1]], max_ox_states=limit)
    assert elements == ["Cl", cation]
    assert states == [(-1, 1)]


@pytest.mark.parametrize(
    "sampler", [elec_neutral_all_common, elec_neutral_all_common_loose]
)
def test_neutral_sampler_returns_no_solution_without_hiding_valid_limits(sampler):
    assert sampler(["Na"], [[1]], max_ox_states=100) == ([], ["Na"])
    assert sampler(["Na", "Cl"], [[1], [1]], max_ox_states=100, timeout_sec=0) == (
        [],
        ["Na", "Cl"],
    )


def _saved_batch(count, offset):
    data = {
        key: torch.ones(count, 3)
        for key in (
            "normed_batch_dir_coords",
            "init_coords",
            "batch_abc",
            "batch_angle",
            "init_abc",
            "init_angles",
            "scaled_batch_abc",
            "scaled_batch_angle",
        )
    }
    data.update(
        {
            key: torch.ones(count, 98)
            for key in (
                "atomic_distribution",
                "normalized_dist",
                "onehot_x",
            )
        }
    )
    data["atomic_mask"] = torch.ones(count, 98, 1)
    data.update(
        {
            key: torch.ones(count, 1)
            for key in (
                "sharpened_ox_mask",
                "ox_mask_learnable_tensor_per_crystal",
                "ox_states_used_mask",
            )
        }
    )
    data.update(
        {
            key: torch.zeros(count)
            for key in (
                "site_ids",
                "bandgap_onehot",
                "bandgap_dist",
                "ef_onehot",
                "ef_dist",
                "loss_onehot",
                "tolerance",
                "tolerance_loss",
            )
        }
    )
    data.update(
        size=torch.ones(count, dtype=torch.long),
        lattice_vectors=torch.eye(3).repeat(count, 1, 1),
        fnames=np.array([f"sample-{offset+i}" for i in range(count)]),
        gap_history=np.full((count, 3), offset),
        ef_history=np.full((count, 3), offset),
        t_history=np.full((count, 3), offset),
        loss_total_onehot=torch.arange(count, dtype=torch.float32) + offset,
        force_loss_raw_onehot=torch.full((count,), float(offset)),
        pressure_loss_raw_onehot=torch.full((count,), float(offset + 1)),
        force_loss_raw_history=torch.full((count, 3), float(offset)),
        pressure_loss_raw_history=torch.full((count, 3), float(offset + 1)),
        sa_accept_flags=torch.ones(3, 1, dtype=torch.bool),
        sa_sample_accept_flags=torch.ones(3, 1, count, dtype=torch.bool),
    )
    return data


def test_save_partial_batch_preserves_sample_axes_and_physical_diagnostics():
    from src.utils.save_optimization_result import save_optimization_results

    buffer = io.BytesIO()
    save_optimization_results(
        buffer, [_saved_batch(2, 0), _saved_batch(1, 2)], "ALIGNN", None
    )
    buffer.seek(0)
    with np.load(buffer, allow_pickle=True) as saved:
        assert saved["atomic_mask"].shape == (3, 98)
        assert saved["normalized_dist"].shape == (3, 98)
        assert saved["ox_mask_learnable_tensor_per_crystal"].shape == (3,)
        assert saved["ox_states_used_mask"].shape == (3,)
        assert saved["sharpened_ox_mask"].shape == (3,)
        assert saved["gap_history"].shape == (3, 3)
        np.testing.assert_array_equal(saved["gap_history"][:, 2], [2, 2, 2])
        np.testing.assert_array_equal(saved["loss_total_onehot"], [0, 1, 2])
        assert saved["force_loss_raw_onehot"].shape == (3,)
        assert saved["force_loss_raw_history"].shape == (3, 3)
        assert saved["pressure_loss_raw_history"].shape == (3, 3)
        np.testing.assert_array_equal(saved["pressure_loss_raw_onehot"], [1, 1, 3])
        assert saved["sa_accept_flags"].shape == (6, 1)
        flags = saved["sa_sample_accept_flags"]
        assert flags.shape == (2,) and flags.dtype == object
        assert [batch.shape for batch in flags] == [(3, 1, 2), (3, 1, 1)]
        np.testing.assert_array_equal(
            saved["original_fnames"], ["sample-0", "sample-1", "sample-2"]
        )


class _DifferentiablePredictor(torch.nn.Module):
    """Small analytic predictor; all preprocessing and optimization remain real."""

    def forward(self, inputs):
        distribution, positions, batch, lattice, sizes = inputs
        predictions = []
        for index in range(len(sizes)):
            selected = batch == index
            predictions.append(
                positions[selected].square().mean()
                + lattice[index].square().mean() * 0.01
                + distribution[selected].square().mean()
            )
        return torch.stack(predictions).unsqueeze(-1)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("e_form_min", [None, -0.5])
def test_loss_preserves_sample_axis_for_mutation(batch_size, e_form_min):
    from src.losses.bandgap_loss import GapLoss
    from src.losses.loss_utils import calculate_loss_from_output

    bandgap = torch.tensor([[2.5], [3.0]])[:batch_size]
    energy = torch.tensor([[-1.0], [-0.25]])[:batch_size]
    outputs = calculate_loss_from_output(
        bandgap_pred=bandgap,
        ef_pred=energy,
        sharpened_ox_mask=torch.ones(batch_size, 1),
        normalized_dist=torch.ones(batch_size, 98) / 98,
        site_ids=torch.zeros(batch_size),
        gap_loss_func=GapLoss(gap_min=1.0, gap_max=2.0),
        tolerance_loss_func=None,
        ef_coef=0.5,
        e_form_min=e_form_min,
        crystal_system=None,
        radii_tensor=torch.zeros(batch_size, 98, 1),
    )
    total, each, gap, formation, tolerance_loss, tolerance = outputs
    assert total.ndim == 0
    assert all(value.shape == (batch_size,) for value in outputs[1:])
    expected_energy = energy.flatten()
    if e_form_min is not None:
        expected_energy = (expected_energy - e_form_min).clamp_min(0)
    torch.testing.assert_close(gap, bandgap.flatten() - 2)
    torch.testing.assert_close(formation, expected_energy * 0.5)
    torch.testing.assert_close(each, gap + formation)
    assert torch.isnan(tolerance_loss).all() and torch.isnan(tolerance).all()


def test_crystalformer_directory_loader_and_gradient_step_with_partial_batch(
    monkeypatch,
):
    from src.data_loader.crystalformer_loader import load_crystalformer_initial_data
    from src.losses.bandgap_loss import GapLoss
    from src.smoacs_opt_eval.crystalformer_optimize import (
        optimize_solution_Crystalformer,
    )

    # The scientific mask file is replaced by memory data, while POSCAR loading,
    # neutral-state sampling, batching, preprocessing and Adam are exercised.
    monkeypatch.setattr(
        np, "load", lambda *args, **kwargs: {"1": np.ones(98), "-1": np.ones(98)}
    )
    settings = {
        "abc_range": (2, 20),
        "angle_range": (30, 150),
        "copy_mutation": "copy_mutation___steps_cm-50",
    }
    batches = load_crystalformer_initial_data(
        settings_dict=settings,
        dir_path=str(Path(__file__).parents[1] / "fixtures/crystalformer"),
        intermid_dir_path="unused-mask-path",
        mask_dict_path=None,
        crystal_system=None,
        batch_size=2,
        use_atomic_mask=True,
        max_ox_states=1,
        mask_method="test",
        angle_range=settings["angle_range"],
        abc_range=settings["abc_range"],
        test_mode=False,
        num_candidate=3,
        device="cpu",
    )
    assert [batch[0]["size"].tolist() for batch in batches] == [[4, 2], [6]]
    assert [name for batch, _ in batches for name in batch["fnames"]] == [
        "NaCl_4.vasp",
        "NaCl_2.vasp",
        "NaCl_6.vasp",
    ]
    for inputs, scalers in batches:
        initial_coords = inputs["batch_dir_coords"].clone()
        batch_settings = settings.copy()
        if len(inputs["size"]) == 1:
            # A final single-candidate batch must also reach the scheduled
            # copy/mutate step, even though there is no second candidate to copy.
            batch_settings["copy_mutation"] = "copy_mutation___steps_cm-0"
        optimized, _, _ = optimize_solution_Crystalformer(
            settings_dict=batch_settings,
            mini_batch_inputs_dict=inputs,
            learning_rates=[0.001] * 3,
            learning_rate_cycle=[1] * 3,
            num_steps=1,
            dist_temp_sche=[1.0],
            e_form_coef=1.0,
            gap_loss_func=GapLoss(mode="minimization"),
            tolerance_loss_func=None,
            bandgap_model=_DifferentiablePredictor(),
            e_form_model=_DifferentiablePredictor(),
            scalers=scalers,
            crystal_system=None,
            limit_coords_displacement=None,
            e_form_min=None,
            adding_noise_scale=None,
            device="cpu",
            time_series_save=False,
        )
        assert optimized["gap_history"].shape == (len(inputs["size"]), 1)
        assert torch.isfinite(optimized["batch_dir_coords"]).all()
        assert not torch.equal(optimized["batch_dir_coords"], initial_coords)
