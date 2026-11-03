"""CPU checks for copy-and-mutate graph fallback, without filesystem fixtures."""

from copy import deepcopy
from types import SimpleNamespace

import dgl
import numpy as np
import pandas as pd
import pytest
import torch

from src.data_loader import alignn_loader as loader
from src.graph import graph_utils as graphs
from src.optimization.optimizer_utils import define_optimizer_and_scheduler
from src.smoacs_opt_eval import alignn_graph_update as updates
from src.smoacs_opt_eval import copy_and_mutate as mutation
from src.smoacs_opt_eval.step_update_utils import update_parameters_step
from src.utils.feature_utils import calculate_onehot
from src.utils.scalers import ABC_Scaler, Angle_Scaler


TARGET_NAMES = [
    "scaled_batch_abc",
    "scaled_batch_angle",
    "batch_dir_coords",
    "atomic_distribution",
    "ox_mask_learnable_tensor_per_crystal",
]


def make_inputs(sizes):
    refs = []
    for i, count in enumerate(sizes):
        x = np.arange(count, dtype=np.float32) / count
        refs.append(
            {
                "source_jid": f"candidate-{i}",
                "lattice_mat": np.eye(3, dtype=np.float32) * (8 + i),
                "abc": np.full(3, 8 + i, dtype=np.float32),
                "angles": np.full(3, 90, dtype=np.float32),
                "coords": np.column_stack((x, (x * 2.1) % 1, (x * 3.3) % 1)),
            }
        )
    coords = torch.tensor(np.concatenate([r["coords"] for r in refs]))
    n_atoms = sum(sizes)
    mask = torch.zeros(n_atoms, 98, 2)
    mask[:, 1, :] = 1
    mask[:, 5, :] = 1
    return {
        "batch_dir_coords": coords,
        "init_coords": coords.clone(),
        "batch_abc": torch.tensor(np.stack([r["abc"] for r in refs])),
        "batch_angle": torch.full((len(sizes), 3), 90.0),
        "size": torch.tensor(sizes),
        "wp_size": torch.tensor(sizes, dtype=torch.int32),
        "atomic_distribution": torch.ones(n_atoms, 98),
        "ox_mask_learnable_tensor_per_crystal": torch.ones(len(sizes), 2),
        "atomic_mask": mask,
        "radii_tensor": torch.ones(n_atoms, 98, 2),
        "ox_states_used_mask": torch.ones(n_atoms, 2),
        "site_ids": torch.arange(n_atoms, dtype=torch.int32),
        "fnames": [r["source_jid"] for r in refs],
        "graph_fallback_geometry": refs,
    }


def mutate(inputs, monkeypatch, donor=0, target=2, skip=False, step=150):
    """Exercise the real update_parameters_step -> copy_and_mutate handoff."""
    abc_scaler = ABC_Scaler(inputs["batch_abc"].clone(), 2, 20, device="cpu")
    angle_scaler = Angle_Scaler(30, 150)
    targets = [
        torch.nn.Parameter(abc_scaler.scale(inputs["batch_abc"]).clone()),
        torch.nn.Parameter(angle_scaler.scale(inputs["batch_angle"]).clone()),
        torch.nn.Parameter(inputs["batch_dir_coords"].clone()),
        torch.nn.Parameter(inputs["atomic_distribution"].clone()),
        torch.nn.Parameter(inputs["ox_mask_learnable_tensor_per_crystal"].clone()),
    ]
    opts = define_optimizer_and_scheduler(
        0.001,
        0.001,
        0.001,
        targets[4],
        targets[3],
        targets[0],
        targets[1],
        targets[2],
        True,
        600,
        1,
        1,
        1,
    )
    gap_loss = torch.ones(3)
    ef_loss = torch.zeros(3)
    each_loss = torch.ones(3)
    gap_loss[donor] = 0
    ef_loss[donor] = -1
    each_loss[target] = 100
    monkeypatch.setattr(mutation.random, "choice", lambda choices: donor)
    monkeypatch.setattr(mutation.random, "random", lambda: 0.0 if skip else 0.5)
    _, targets, extra = update_parameters_step(
        optimization_targets=targets,
        optimization_target_names=TARGET_NAMES,
        step_i=step,
        total_loss=sum(p.sum() * 0 for p in targets),
        each_loss=each_loss,
        gap_loss=gap_loss,
        ef_loss=ef_loss,
        tolerance_loss=torch.zeros(3),
        total_steps=600,
        learning_rates=[0.001] * 3,
        copy_mutation="copy_mutation___steps_cm-150__mutation_noise-0.0__top_ratio-0.2__group_C_use_rate-0.02__atom_dist_init-False",
        cry_atom_data=torch.repeat_interleave(torch.arange(3), inputs["size"]),
        size=inputs["size"],
        wp_size=inputs["wp_size"],
        atomic_mask=inputs["atomic_mask"],
        ox_states_used_mask=inputs["ox_states_used_mask"],
        site_ids=inputs["site_ids"],
        radii_tensor=inputs["radii_tensor"],
        init_coords=inputs["init_coords"],
        angle_optimization=True,
        optimizer_lattice=opts[0],
        optimizer_atom=opts[1],
        optimizer_coords=opts[2],
        scheduler_lattice=opts[3],
        scheduler_atom=opts[4],
        scheduler_coords=opts[5],
        fixed_inputs=inputs,
        scalers={"abc_scaler": abc_scaler, "angle_scaler": angle_scaler},
        bandgap_model=None,
        e_form_model=None,
        limit_coords_displacement=None,
        gap_loss_func=None,
        tolerance_loss_func=None,
        Ef_coef_value=1,
        crystal_system=None,
        e_form_min=None,
    )
    for key in (
        "atomic_mask",
        "ox_states_used_mask",
        "site_ids",
        "radii_tensor",
        "init_coords",
    ):
        inputs[key] = extra[key]
    inputs["size"] = extra["new_size"]
    inputs["wp_size"] = extra["new_wp_size"]
    inputs.update(dict(zip(TARGET_NAMES, targets)))
    inputs["batch_abc"] = abc_scaler.rescale(targets[0])
    inputs["batch_angle"] = angle_scaler.rescale(targets[1])
    inputs["_test_scalers"] = {"abc_scaler": abc_scaler, "angle_scaler": angle_scaler}
    return targets


def memory_loader(monkeypatch):
    """Replace only loader file output and multiprocessing, retaining graph code."""

    def dataset_in_memory(dataset, **kwargs):
        df = pd.DataFrame(dataset)
        gs = graphs.load_graphs(df, use_canonize=True)
        lgs = [g.line_graph(shared=True) for g in gs]
        for lg in lgs:
            lg.apply_edges(graphs.compute_bond_cosines_for_line_graph)
        data = SimpleNamespace(
            collate_line_graph=None, batch=(dgl.batch(gs), dgl.batch(lgs))
        )
        return data, df

    monkeypatch.setattr(loader, "get_torch_dataset", dataset_in_memory)
    monkeypatch.setattr(loader, "DataLoader", lambda data, **kwargs: [data.batch])


@pytest.mark.parametrize(
    "old_count,new_count", [(6, 3), (11, 4), (10, 3), (3, 6), (4, 4)]
)
def test_mutation_restores_matching_geometry_and_edges(
    monkeypatch, old_count, new_count
):
    inputs = make_inputs([new_count, 4, old_count])
    original_names = inputs["fnames"].copy()
    donor_mask = inputs["atomic_mask"][:new_count].clone()
    donor_sites = inputs["site_ids"][:new_count].clone()
    mutate(inputs, monkeypatch)
    inputs["batch_angle"] = inputs["batch_angle"].detach().clone()
    inputs["batch_angle"][2] = torch.tensor([30.0, 30.0, 90.0])
    dataset = loader.build_dataset_for_graph(inputs, None)
    restored = dataset[2]
    assert inputs["fnames"] == original_names
    assert restored["jid"] == "candidate-2"
    assert inputs["graph_fallback_geometry"][2]["source_jid"] == "candidate-0"
    np.testing.assert_array_equal(
        restored["atoms"]["coords"], inputs["graph_fallback_geometry"][0]["coords"]
    )
    np.testing.assert_array_equal(
        restored["atom_mask_for_all_ox_states"], donor_mask.numpy()
    )
    np.testing.assert_array_equal(restored["site_ids"], donor_sites.numpy())
    df = pd.DataFrame(dataset)
    gs = graphs.load_graphs(df, use_canonize=True)
    tensors = graphs.get_all_atoms_coords(df, None)
    assert [g.num_nodes() for g in gs] == [new_count, 4, new_count]
    assert tensors[5] == [new_count, 4, new_count]
    assert [g.num_edges() for g in gs] == tensors[6]
    assert torch.isfinite(tensors[3]).all()
    assert int(tensors[1].max()) < sum(tensors[5])
    assert int(tensors[2].max()) < sum(tensors[5])
    # Compare the actual ordered edge vectors of both generation paths.
    coords, dst, src, displacement, lattices = tensors[:5]
    edge_lattices = torch.repeat_interleave(lattices, torch.tensor(tensors[6]), dim=0)
    vectors, _ = graphs.compute_bondlength(
        coords, dst, src, displacement, edge_lattices
    )
    torch.testing.assert_close(
        vectors, torch.cat([g.edata["r"] for g in gs]), atol=1e-5, rtol=1e-5
    )
    # Restoration is for graph data only; the invalid optimization angle remains.
    torch.testing.assert_close(
        inputs["batch_angle"][2], torch.tensor([30.0, 30.0, 90.0])
    )


def test_chained_copy_keeps_lineage_without_shared_arrays(monkeypatch):
    inputs = make_inputs([3, 5, 8])
    mutate(inputs, monkeypatch, donor=0, target=2)
    mutate(inputs, monkeypatch, donor=2, target=1)
    refs = inputs["graph_fallback_geometry"]
    assert [ref["source_jid"] for ref in refs] == ["candidate-0"] * 3
    assert inputs["size"].tolist() == [3, 3, 3]
    original = refs[0]["coords"].copy()
    refs[1]["coords"][0, 0] += 0.1
    np.testing.assert_array_equal(refs[0]["coords"], original)
    np.testing.assert_array_equal(refs[2]["coords"], original)


@pytest.mark.parametrize("skip,step", [(True, 150), (False, 149)])
def test_unmutated_candidates_keep_reference(monkeypatch, skip, step):
    inputs = make_inputs([3, 4, 6])
    ref = inputs["graph_fallback_geometry"][2]
    mutate(inputs, monkeypatch, skip=skip, step=step)
    assert inputs["graph_fallback_geometry"][2] is ref
    assert inputs["size"].tolist() == [3, 4, 6]


@pytest.mark.parametrize("bad_key", ["batch_dir_coords", "batch_abc", "batch_angle"])
def test_raw_nonfinite_geometry_restored_before_random_fill(bad_key):
    inputs = make_inputs([3, 4, 6])
    inputs[bad_key][0, 0] = float("nan")
    row = loader.build_dataset_for_graph(inputs, None)[0]
    for key in loader.GRAPH_GEOMETRY_KEYS:
        np.testing.assert_array_equal(
            row["atoms"][key], inputs["graph_fallback_geometry"][0][key]
        )


@pytest.mark.parametrize("invalid", ["missing", "count", "nan", "singular"])
def test_invalid_reference_fails_with_candidate_context(invalid):
    inputs = make_inputs([3, 4, 6])
    inputs["batch_angle"][2] = torch.tensor([30.0, 30.0, 90.0])
    if invalid == "missing":
        del inputs["graph_fallback_geometry"]
    elif invalid == "count":
        inputs["graph_fallback_geometry"][2]["coords"] = np.zeros((2, 3))
    elif invalid == "nan":
        inputs["graph_fallback_geometry"][2]["coords"][0, 0] = np.nan
    else:
        inputs["graph_fallback_geometry"][2]["lattice_mat"][:] = 0
    with pytest.raises(ValueError, match="candidate-2:.*graph fallback"):
        loader.build_dataset_for_graph(inputs, None)


def test_finite_geometry_does_not_require_or_use_reference():
    inputs = make_inputs([3, 4, 6])
    expected = loader.build_dataset_for_graph(inputs, None)
    del inputs["graph_fallback_geometry"]
    actual = loader.build_dataset_for_graph(inputs, None)
    for before, after in zip(expected, actual):
        for key in loader.GRAPH_GEOMETRY_KEYS:
            np.testing.assert_array_equal(before["atoms"][key], after["atoms"][key])


def test_atomic_nan_is_not_repaired_by_geometry_fallback():
    inputs = make_inputs([3, 4, 6])
    inputs["atomic_distribution"][0, :86] = float("nan")
    dataset = loader.build_dataset_for_graph(inputs, None)
    assert np.isfinite(dataset[0]["atoms"]["coords"]).all()
    assert torch.isnan(inputs["atomic_distribution"][0, :86]).all()
    _, onehot, _, _ = calculate_onehot(
        {
            "normalized_dist": inputs["normalized_dist"],
            "sharpened_ox_mask": torch.ones(sum(inputs["size"]), 2),
            "atomic_distribution": inputs["atomic_distribution"],
            "atomic_mask": inputs["atomic_mask"],
        }
    )
    assert int(onehot[0].argmax()) == 1
    assert torch.isnan(inputs["atomic_distribution"][0, :86]).all()


def test_graph_rejects_nonfinite_geometry_and_node_mismatch():
    row = loader.build_dataset_for_graph(make_inputs([3, 4, 6]), None)[0]
    invalid = deepcopy(row)
    invalid["atoms"]["coords"][0, 0] = np.nan
    with pytest.raises(ValueError, match="candidate-0: nonfinite geometry"):
        graphs.load_graphs(pd.DataFrame([invalid]))
    row["atom_mask_for_all_ox_states"] = row["atom_mask_for_all_ox_states"][:2]
    with pytest.raises(ValueError, match="candidate-0: graph/mask/site_ids mismatch"):
        graphs.load_graphs(pd.DataFrame([row]))


def test_initialization_and_graph_parameter_update_share_reference(monkeypatch):
    memory_loader(monkeypatch)
    rows = loader.build_dataset_for_graph(make_inputs([3, 4, 6]), None)
    batches = loader.load_ALIGNN_initial_data(
        settings_dict={},
        dir_path="unused",
        intermid_dir_path="unused",
        mask_dict_path=None,
        batch_size=3,
        use_atomic_mask=True,
        num_workers=0,
        max_ox_states=2,
        mask_method="unused",
        crystal_system=None,
        angle_range=(30, 150),
        abc_range=(2, 20),
        input_dataframe=rows,
        wyckoff_expander=None,
        device="cpu",
    )
    inputs = batches[0][0]
    refs = inputs["graph_fallback_geometry"]
    np.testing.assert_array_equal(refs[2]["coords"], rows[2]["atoms"]["coords"])
    assert not np.shares_memory(refs[2]["coords"], rows[2]["atoms"]["coords"])
    targets = mutate(inputs, monkeypatch)
    inputs["batch_angle"] = inputs["batch_angle"].detach().clone()
    inputs["batch_angle"][2] = torch.tensor([30.0, 30.0, 90.0])
    updates.graph_parameter_update(
        TARGET_NAMES,
        targets,
        inputs,
        inputs,
        3,
        None,
        num_workers=0,
        scalers=inputs["_test_scalers"],
    )
    assert inputs["graph_fallback_geometry"] is refs
    assert refs[2]["source_jid"] == "candidate-0"
    assert inputs["g"].num_nodes() == 10
    assert int(inputs["num_edges"].sum()) == inputs["g"].num_edges()
