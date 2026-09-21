import math

import pytest
import torch

from src.losses.force_pressure_loss import (
    active_force_pressure,
    force_pressure_losses,
    transform_loss,
    weighted_force_pressure,
)


def config(**kwargs):
    return (
        dict(
            use_force_loss=True,
            use_pressure_loss=True,
            force_loss_mode="raw",
            pressure_loss_mode="raw",
            F_ref=1.0,
            P_ref=1.0,
            force_loss_clip=1e-7,
            pressure_loss_clip=1e-7,
        )
        | kwargs
    )


def test_atom_mean_structure_weight_and_pressure_sign():
    forces = torch.tensor([[1.0, 2.0, 2.0], [0.0, 0.0, 0.0], [0.0, 0.0, 6.0]])
    stresses = torch.stack([-2 * torch.eye(3), 2 * torch.eye(3)])
    result = force_pressure_losses(
        dict(forces=forces, stresses=stresses), torch.tensor([1, 2]), config()
    )
    torch.testing.assert_close(result["force_loss_raw"], torch.tensor([9.0, 18.0]))
    torch.testing.assert_close(result["pressure_GPa"], torch.tensor([2.0, -2.0]))
    torch.testing.assert_close(result["pressure_loss_raw"], torch.tensor([4.0, 4.0]))
    assert weighted_force_pressure(result).mean() == 17.5


@pytest.mark.parametrize("mode", ["log", "raw"])
def test_independent_clips_and_weight_outside_log(mode):
    properties = dict(
        forces=torch.tensor([[0.01, 0.0, 0.0]]), stresses=-0.01 * torch.eye(3)[None]
    )
    base = config(force_loss_mode=mode, pressure_loss_mode=mode, F_ref=2, P_ref=3)
    a = force_pressure_losses(properties, torch.tensor([1]), base)
    b = force_pressure_losses(
        properties, torch.tensor([1]), base | {"force_loss_clip": 1.0}
    )
    c = force_pressure_losses(
        properties, torch.tensor([1]), base | {"pressure_loss_clip": 1.0}
    )
    torch.testing.assert_close(a["pressure_loss_weighted"], b["pressure_loss_weighted"])
    torch.testing.assert_close(a["force_loss_weighted"], c["force_loss_weighted"])
    assert a["force_loss_weighted"].item() == pytest.approx(
        -8 if mode == "log" else 0.0002
    )
    if mode == "raw":
        torch.testing.assert_close(a["force_loss_weighted"], b["force_loss_weighted"])
        torch.testing.assert_close(
            a["pressure_loss_weighted"], c["pressure_loss_weighted"]
        )


def test_zero_deviatoric_stress_and_disabled_terms():
    stress = torch.diag(torch.tensor([1.0, -1.0, 0.0]))[None]
    result = force_pressure_losses(
        {"stresses": stress}, torch.tensor([1]), config(use_force_loss=False)
    )
    assert result["pressure_loss_raw"].item() == 0
    assert result["deviatoric_stress_norm_GPa"].item() > 0
    assert "force_loss_raw" not in result
    assert not any(active_force_pressure(config(F_ref=0, P_ref=0)))
    assert force_pressure_losses({}, torch.tensor([1]), config(F_ref=0, P_ref=0)) == {}


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_loss_names_structure(bad):
    with pytest.raises(ValueError, match="sample-b"):
        transform_loss(torch.tensor([0.0, bad]), "log", 1e-7, ["sample-a", "sample-b"])


def test_nonfinite_shear_stress_fails_with_structure_id():
    stress = torch.eye(3)[None]
    stress[0, 0, 1] = float("nan")
    with pytest.raises(ValueError, match="shear-sample"):
        force_pressure_losses(
            {"stresses": stress},
            torch.tensor([1]),
            config(use_force_loss=False),
            ["shear-sample"],
        )


@pytest.mark.parametrize("mode", ["log", "raw"])
def test_analytic_derivative_and_gradcheck(mode):
    raw = torch.tensor([0.0, 0.001, 2.0], dtype=torch.float64, requires_grad=True)
    value = 2 * transform_loss(raw, mode, 0.01)
    (grad,) = torch.autograd.grad(value.sum(), raw)
    expected = [0.0, 0.0, 2 / (2 * math.log(10))] if mode == "log" else [2.0, 2.0, 2.0]
    torch.testing.assert_close(grad, torch.tensor(expected, dtype=torch.float64))
    positive = torch.tensor([0.001, 2.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda x: 2 * transform_loss(x, mode, 0.01), (positive,)
    )


def test_replication_and_batch_independence():
    force = torch.tensor([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]])
    a = force_pressure_losses(
        {"forces": force}, torch.tensor([2]), config(use_pressure_loss=False)
    )
    b = force_pressure_losses(
        {"forces": force.repeat(3, 1)},
        torch.tensor([2, 4]),
        config(use_pressure_loss=False),
    )
    torch.testing.assert_close(b["force_loss_raw"], a["force_loss_raw"].expand(2))


class _PhysicalModel:
    def __init__(self):
        self.calls = []

    def compute_properties(self, **kwargs):
        self.calls.append(kwargs)
        positions = kwargs["positions"]
        lattice = kwargs["lattice"]
        result = {"total_energy": lattice.square().sum((1, 2))}
        if kwargs["include_forces"]:
            result["forces"] = positions + kwargs["atom_types"][:, :3]
        if kwargs["include_stresses"]:
            result["stresses"] = lattice
        if not kwargs["create_graph"]:
            result = {k: v.detach() for k, v in result.items()}
        return result

    def compute_energy(self, **kwargs):
        self.calls.append(kwargs)
        return kwargs["lattice"].square().sum((1, 2))


@pytest.mark.parametrize("model", ["alignn", "crystalformer"])
@pytest.mark.parametrize(
    "enabled", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("grad_enabled", [True, False])
def test_loss_evaluation_routes(monkeypatch, model, enabled, grad_enabled):
    from src.smoacs_opt_eval import loss_eval

    lattice = torch.eye(3).repeat(2, 1, 1).requires_grad_()
    fractional = torch.tensor(
        [[0.2, 0.3, 0.4], [0.3, 0.4, 0.5], [0.4, 0.5, 0.6]], requires_grad=True
    )
    distribution = torch.full((3, 98), 1 / 98, requires_grad=True)
    sizes = torch.tensor([1, 2])
    batch = torch.tensor([0, 1, 1])

    def forward(**kwargs):
        output = dict(
            normalized_dist=distribution,
            lattice_vectors=lattice,
            normed_batch_dir_coords=fractional,
            sharpened_ox_mask=torch.ones(3, 1),
            pos=torch.einsum("ni,nij->nj", fractional, lattice[batch]),
            batch=batch,
        )
        return torch.zeros(2), torch.zeros(2), output

    monkeypatch.setattr(
        loss_eval,
        (
            "forward_propagation_ALIGNN"
            if model == "alignn"
            else "forward_propagation_Crystalformer"
        ),
        forward,
    )

    def legacy(**kwargs):
        each = lattice.square().sum((1, 2)) * 0.1
        zeros = torch.zeros(2)
        return each.mean(), each, zeros, zeros, zeros, zeros

    monkeypatch.setattr(loss_eval, "calculate_loss_from_output", legacy)
    physical = _PhysicalModel()
    settings = config(use_force_loss=enabled[0], use_pressure_loss=enabled[1])
    evaluate = getattr(loss_eval, f"evaluate_loss_{model}")
    kwargs = dict(
        optimization_targets=[lattice],
        mini_batch_inputs_dict=dict(
            size=sizes,
            site_ids=torch.zeros(3),
            radii_tensor=torch.ones(3, 98, 1),
            fnames=["a", "b"],
        ),
        scalers={},
        temperature=1.0,
        gap_loss_func=None,
        tolerance_loss_func=None,
        e_form_coef=1.0,
        crystal_system=None,
        e_form_min=None,
        bandgap_model=None,
        e_form_model=None,
        energy_module=physical,
        energy_coef=0.2,
        energy_every=2,
        force_pressure_config=settings,
        grad_enabled=grad_enabled,
    )
    first = evaluate(**kwargs, step_i=0)
    assert len(physical.calls) == 1
    contribution = weighted_force_pressure(first["force_pressure"])
    expected = torch.tensor([0.9, 0.6]) + contribution
    torch.testing.assert_close(first["each_loss"], expected)
    torch.testing.assert_close(first["total_loss"], expected.mean())
    if grad_enabled:
        first["total_loss"].backward()
        assert torch.isfinite(lattice.grad).all()
        if enabled[0]:
            assert fractional.grad is not None and distribution.grad is not None
    else:
        assert first["total_loss"].grad_fn is None
    second = evaluate(**kwargs, step_i=1)
    assert len(physical.calls) == (2 if any(enabled) else 1)
    torch.testing.assert_close(
        second["each_loss"],
        torch.full((2,), 0.3) + weighted_force_pressure(second["force_pressure"]),
    )


def test_final_totals_keep_soft_and_onehot_separate():
    from src.losses.force_pressure_loss import final_force_pressure

    physical = _PhysicalModel()
    actual = final_force_pressure(
        physical,
        config(),
        torch.full((1, 98), 0.1),
        torch.zeros(1, 98),
        torch.ones(1, 3),
        torch.eye(3)[None],
        torch.tensor([1]),
        torch.tensor([10.0]),
        torch.tensor([20.0]),
        energy_coef=2.0,
    )
    assert len(physical.calls) == 2
    assert actual["force_loss_raw_soft"] > actual["force_loss_raw_onehot"]
    torch.testing.assert_close(actual["loss_total_onehot"], torch.tensor([30.0]))
    torch.testing.assert_close(actual["loss_total"], actual["loss_total_onehot"])
    assert all(value.grad_fn is None for value in actual.values())


def test_zero_coefficients_make_no_physical_requests(monkeypatch):
    from src.smoacs_opt_eval import loss_eval

    def forward(**kwargs):
        return (
            torch.ones(1),
            torch.ones(1),
            dict(sharpened_ox_mask=torch.ones(1, 1), normalized_dist=torch.ones(1, 98)),
        )

    monkeypatch.setattr(loss_eval, "forward_propagation_Crystalformer", forward)
    monkeypatch.setattr(
        loss_eval,
        "calculate_loss_from_output",
        lambda **kwargs: (torch.tensor(2.0), *[torch.tensor([2.0]) for _ in range(5)]),
    )
    result = loss_eval.evaluate_loss_crystalformer(
        [torch.ones(1)],
        dict(site_ids=torch.ones(1), radii_tensor=torch.ones(1)),
        {},
        1.0,
        None,
        None,
        1.0,
        None,
        None,
        None,
        None,
        force_pressure_config=config(F_ref=0, P_ref=0),
    )
    assert result["total_loss"] == 2 and result["force_pressure"] == {}


@pytest.mark.parametrize("model", ["alignn", "crystalformer"])
@pytest.mark.parametrize("composed", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_final_evaluators_preserve_success_and_true_total(
    monkeypatch, model, composed, enabled, superhard=False
):
    from src.smoacs_opt_eval import alignn_eval, crystalformer_eval

    target = alignn_eval if model == "alignn" else crystalformer_eval

    class Graph:
        def to(self, device):
            return self

    physical = _PhysicalModel()
    soft = torch.full((1, 98), 1 / 98)
    hard = torch.zeros(1, 98)
    hard[0, 0] = 1.0
    data = dict(
        scaled_batch_abc=torch.ones(1, 3),
        scaled_batch_angle=torch.ones(1, 3),
        batch_dir_coords=torch.ones(1, 3),
        atomic_distribution=soft,
        ox_mask_learnable_tensor_per_crystal=torch.ones(1, 1),
        radii_tensor=torch.ones(1, 98, 1),
        site_ids=torch.zeros(1),
        size=torch.tensor([1]),
        batch=torch.tensor([0]),
        g=Graph(),
        lg=Graph(),
        atom_feat_matrix=torch.eye(98),
        atomic_mask=torch.ones(1, 98, 1),
        optimization_target_names=[],
        optimization_targets=[],
        sharpened_ox_mask=torch.ones(1, 1),
        normalized_dist=torch.full_like(soft, -1),
        fnames=["fixed-sample"],
    )
    soft_predictions = dict(bg=torch.tensor([2.0]), ef=torch.tensor([3.0]))
    hard_predictions = dict(bg=torch.tensor([4.0]), ef=torch.tensor([5.0]))

    def forward(**kwargs):
        return (
            soft_predictions["bg"],
            soft_predictions["ef"],
            dict(
                normalized_dist=soft,
                sharpened_ox_mask=torch.ones(1, 1),
                normed_batch_dir_coords=torch.ones(1, 3),
                lattice_vectors=torch.eye(3)[None],
                pos=torch.ones(1, 3),
                bondlength=torch.ones(1),
                bond_angles=torch.ones(1),
                pred_dict=soft_predictions,
            ),
        )

    monkeypatch.setattr(
        target,
        (
            "forward_propagation_ALIGNN"
            if model == "alignn"
            else "forward_propagation_Crystalformer"
        ),
        forward,
    )
    monkeypatch.setattr(
        target, f"predict_all_{model}", lambda **kwargs: hard_predictions
    )
    monkeypatch.setattr(
        target,
        "calculate_onehot",
        lambda *args, **kwargs: (torch.ones(1), hard, hard, torch.ones(1, 1)),
    )
    if model == "alignn":
        monkeypatch.setattr(
            target,
            "graph_parameter_update",
            lambda **kwargs: kwargs["mini_batch_inputs_dict"],
        )

    def legacy(**kwargs):
        if enabled and model == "crystalformer":
            assert torch.all(kwargs["normalized_dist"] >= 0)
        each = 2 * kwargs["bandgap_pred"] + 3 * kwargs["ef_pred"]
        zeros = torch.zeros(1)
        return each.mean(), each, zeros, zeros, zeros, zeros

    monkeypatch.setattr(target, "calculate_loss_from_output", legacy)

    class Composer:
        def compute(self, predictions, context):
            each = predictions["bg"] * 7 + predictions["ef"] * 11
            return each.mean(), each, {}

    settings = dict(
        abc_range=(2.0, 10.0),
        angle_range=(30.0, 150.0),
        num_batch_crystal=1,
        force_pressure_config=config(use_force_loss=enabled, use_pressure_loss=enabled),
    )
    if superhard:
        settings["superhard_cfg"] = dict(
            G_coef=2.0, B_coef=3.0, k_coef=0.0, k_min=0.0, k_max=2.0
        )
    kwargs = dict(
        settings_dict=settings,
        optimized_mini_batch_inputs_dict=data,
        scalers=dict(abc_scaler=None, angle_scaler=None),
        bandgap_model=None,
        e_form_model=None,
        gap_loss_func=None,
        tolerance_loss_func=None,
        crystal_system=None,
        limit_coords_displacement=None,
        e_form_min=None,
        adding_noise_scale=None,
        predictors={"test": None},
        primary_metrics=dict(bandgap_onehot="bg", ef_onehot="ef"),
        loss_composer=Composer() if composed else None,
        energy_module=physical if enabled else None,
        energy_coef=0.5,
    )
    if model == "alignn":
        kwargs.update(e_form_coef=1.0, wyckoff_expander=None)
        result = target.evaluation_for_each_batch_ALIGNN(**kwargs)
    else:
        kwargs.update(ef_coef4eval=1.0)
        result = target.evaluation_for_each_batch_Crystalformer(**kwargs)
    assert result["loss_onehot"].item() == 0
    if enabled:
        base = -23.0 if superhard else (83.0 if composed else 23.0)
        torch.testing.assert_close(result["loss_total"], torch.tensor([base + 7 + 1.5]))
        assert result["loss_total_soft"] != result["loss_total_onehot"]
        assert len(physical.calls) == 2
    else:
        assert "loss_total" not in result and not physical.calls


def test_superhard_energy_and_physical_terms_are_added_once(monkeypatch):
    test_final_evaluators_preserve_success_and_true_total(
        monkeypatch, "alignn", False, True, superhard=True
    )


@pytest.mark.parametrize("batch_count", [1, 2])
def test_npz_csv_shapes_and_ranking_without_files(batch_count, monkeypatch):
    import io
    import numpy as np
    import pandas as pd
    from src.utils.save_optimization_result import save_optimization_results
    from src.evaluation.evaluation import load_and_prepare_dataframe

    batches = []
    for index in range(batch_count):
        data = {
            key: torch.ones(1, 3)
            for key in (
                "normed_batch_dir_coords",
                "batch_dir_coords",
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
                key: torch.ones(1, 98)
                for key in (
                    "atomic_mask",
                    "atomic_distribution",
                    "normalized_dist",
                    "onehot_x",
                )
            }
        )
        data.update(
            {
                key: torch.ones(1, 1)
                for key in (
                    "sharpened_ox_mask",
                    "ox_mask_learnable_tensor_per_crystal",
                    "ox_states_used_mask",
                )
            }
        )
        data.update(
            {
                key: torch.zeros(1)
                for key in (
                    "site_ids",
                    "bandgap_onehot",
                    "bandgap_dist",
                    "loss_onehot",
                    "tolerance",
                    "tolerance_loss",
                )
            }
        )
        data.update(
            size=torch.tensor([1]),
            lattice_vectors=torch.eye(3)[None],
            fnames=np.array([f"sample-{index}"]),
            ef_onehot=torch.tensor([-2.0]),
            ef_dist=torch.tensor([-2.0]),
            gap_history=np.zeros((1, 3)),
            ef_history=np.zeros((1, 3)),
            t_history=np.zeros((1, 3)),
            loss_total=torch.tensor([10.0 - index]),
            loss_total_onehot=torch.tensor([10.0 - index]),
            loss_total_soft=torch.tensor([20.0 - index]),
            force_loss_raw_onehot=torch.tensor([1.0 + index]),
            force_loss_raw_history=torch.full((1, 3), float(index)),
            pred_energy=torch.tensor([100.0 + index]),
        )
        batches.append(data)
    buffer = io.BytesIO()
    save_optimization_results(buffer, batches, "ALIGNN", None)
    buffer.seek(0)
    with np.load(buffer) as saved:
        assert saved["force_loss_raw_history"].shape == (batch_count, 3)
        assert saved["force_loss_raw_onehot"].shape == (batch_count,)
        assert saved["gap_history"].shape == (3, batch_count)
    buffer.seek(0)
    csv = io.StringIO()
    df = load_and_prepare_dataframe(buffer, batch_count, -0.5, csv)
    assert df.index.tolist() == list(reversed(range(batch_count)))
    assert df["bandgap_success"].all()
    for index in range(batch_count):
        assert df.loc[index, "pred_energy"] == 100 + index
        assert df.loc[index, "force_loss_raw_onehot"] == 1 + index
    assert "loss_total_onehot" in pd.read_csv(io.StringIO(csv.getvalue())).columns
