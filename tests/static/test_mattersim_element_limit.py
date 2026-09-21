"""Exercise the stage functions with in-memory I/O and mocked scientific work."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import math
import os
import sys
import time
from contextlib import nullcontext
from itertools import combinations
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Element, Lattice, Structure
from pymatgen.io.vasp import Poscar


def _load_stage_functions():
    # Importing the notebook-derived script initializes output directories and
    # GPU dependencies. Compile its actual stage functions without those effects.
    path = (
        Path(__file__).resolve().parents[2]
        / "scripts/calculate_mattersim_convex_hull_energy.py"
    )
    tree = ast.parse(path.read_text())
    names = {
        "_exceeds_mp_element_limit",
        "_elements_from_chemsys",
        "_normalize_chemsys",
        "chemsys_by_atomic_number",
        "get_elements_from_poscar",
        "_proposal_chemsys_map",
        "_sort_chemsys_desc",
        "_enumerate_subsystems",
        "_query_signature_mp",
        "_query_signature_alex",
        "_load_fetch_done",
        "_save_fetch_done",
        "_has_cache_for_chemsys",
        "_filter_records",
        "_resolve_relax_record_path",
        "stage_prefetch_mp_spec",
        "stage_phase1_spec",
        "stage_prefetch_alex_spec",
        "stage_phase2_spec",
        "stage_export_mp_only",
        "main",
    }
    constants = {
        "MP_PREFETCH_MAX_ELEMENTS",
        "MP_MISSING_EHULL_MAX_SITES",
        "MP_PREFETCH_ELEMENT_LIMIT_REASON",
        "FETCH_EHULL_DB_THRES",
        "CREATE_DFT_NEEDED_THRES",
        "SKIP_EHULL_REASON_TOO_CLOSE",
        "SKIP_EHULL_REASON_PHASE_DIAGRAM_TIMEOUT",
    }
    nodes = [
        node
        for node in tree.body
        if (isinstance(node, ast.FunctionDef) and node.name in names)
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id in constants for t in node.targets)
        )
    ]
    module = ModuleType("mattersim_stages")
    module.__dict__.update(
        Path=Path,
        Element=Element,
        Composition=Composition,
        StructureMatcher=StructureMatcher,
        argparse=argparse,
        hashlib=hashlib,
        json=json,
        os=os,
        time=time,
        combinations=combinations,
        pd=pd,
        np=np,
        PhaseDiagramTimeoutError=TimeoutError,
    )
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"),
        module.__dict__,
    )
    return module


@pytest.fixture
def pipeline(monkeypatch):
    cm = _load_stage_functions()
    root = Path("/in_memory/element_limit")
    args = SimpleNamespace(
        p_dir=root / "input",
        cache_dir=root / "cache",
        r_dir=root / "relaxed",
        rr_dir=root / "results",
        s_dir=root / "dft",
        steps_max=1,
        mp_api_key="mock",
        allow_run_params_fallback=False,
        phase_diagram_timeout_sec=None,
        alexandria_base_url="mock",
    )
    cm.DEFAULT_CACHE_DIR = args.cache_dir
    cm.GLOBAL_RELAX_DIR = root / "global"
    cm.DEFAULT_RELAX_DB_PATH = cm.GLOBAL_RELAX_DIR / "relax_results.json"
    cm.REPO_ROOT = root
    files, structures, csvs = {}, {}, {}
    poscars = []

    def read(path):
        return copy.deepcopy(files[Path(path)])

    def write(path, value):
        files[Path(path)] = copy.deepcopy(value)

    def save_csv(frame, path, **kwargs):
        csvs[Path(path)] = frame.copy()
        files[Path(path)] = "csv"

    def copy_structure(src, dest):
        structures[Path(dest)] = structures[Path(src)]

    def add(n_elements):
        path = args.p_dir / f"p{n_elements}.vasp"
        species = ["H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al"][
            :n_elements
        ]
        structures[path] = Structure(
            Lattice.cubic(20),
            species,
            [[i / n_elements] * 3 for i in range(n_elements)],
        )
        poscars.append(path)
        return path

    def relax(path, outdir, steps):
        relaxed = outdir / f"{path.stem}_relaxed.vasp"
        copy_structure(path, relaxed)
        return {
            "material_id": path.stem,
            "chem_sys": cm.chemsys_by_atomic_number(structures[path].symbol_set),
            "file": str(relaxed),
            "formula": structures[path].formula,
            "status": "relaxed",
            "E_final (eV)": -1.0,
        }

    def collect(cs, *args, **kwargs):
        proposal = next(
            p
            for p in poscars
            if cm.chemsys_by_atomic_number(structures[p].symbol_set) == cs
        )
        ref = cm.GLOBAL_RELAX_DIR / f"mp-{proposal.stem}_relaxed.vasp"
        copy_structure(proposal, ref)
        return [
            {
                "material_id": f"mp-{proposal.stem}",
                "chem_sys": cs,
                "formula": structures[ref].formula,
                "file": str(ref),
                "E_final (eV)": -1.0,
                "source": "mp",
            }
        ]

    monkeypatch.setattr(Path, "exists", lambda p: p in files or p in structures)
    monkeypatch.setattr(Path, "mkdir", lambda *a, **k: None)
    monkeypatch.setattr(Path, "touch", lambda path, **kwargs: write(path, ""))
    monkeypatch.setattr(
        Path,
        "glob",
        lambda p, pattern: iter(
            q for q in structures if q.parent == p and q.match(pattern)
        ),
    )
    monkeypatch.setattr(pd.DataFrame, "to_csv", save_csv)
    monkeypatch.setattr(pd, "read_csv", lambda path: csvs[Path(path)].copy())
    cm.Poscar = SimpleNamespace(
        from_file=Mock(side_effect=lambda path: Poscar(structures[Path(path)]))
    )
    cm.Structure = SimpleNamespace(
        from_file=Mock(side_effect=lambda path: structures[Path(path)])
    )
    cm.shutil = SimpleNamespace(copy=Mock(side_effect=copy_structure), rmtree=Mock())
    cm._list_poscars = lambda path: list(poscars)
    cm._read_json = cm._read_jsonl = cm._load_json_gz = cm._load_relax_results_file = (
        read
    )
    cm._write_json_atomic = write
    cm._write_jsonl_atomic = lambda path, records: write(path, list(records))
    cm._InterProcessFileLock = lambda path: nullcontext()
    cm._mp_lock_path = lambda cache: cache / "mp.lock"
    cm._find_superset_done = lambda *args: []
    cm._load_cached_entries = Mock(return_value=[])
    cm.fetch_mp_structures = Mock(return_value=[])
    cm.fetch_alex_structures = Mock(return_value=[])
    cm._load_known_material_ids = Mock(side_effect=lambda path: set())
    cm._ensure_prop_relaxed = Mock(side_effect=relax)
    cm._collect_records_for_phase_with_log = Mock(side_effect=collect)
    cm._count_ref_candidates = lambda *args: 1
    cm._write_relax_results_for_chemsys = lambda records, cs: write(
        cm.GLOBAL_RELAX_DIR / f"relax_results_{cs}.json.gz", records
    )
    cm._repair_run_params_in_relax_results_file = Mock(
        side_effect=lambda path, **kw: {"records": read(path)}
    )
    cm._record_has_agm = lambda record: record["material_id"].startswith("agm")
    cm._gather_relax_records_for_chemsys = Mock(
        side_effect=lambda cs, *a, **k: read(
            cm.GLOBAL_RELAX_DIR / f"relax_results_{cs}.json.gz"
        )
    )
    cm._compute_ehull_for_proposal = Mock(return_value=0.0)
    cm._compute_ehulls_phase2 = Mock(return_value=(0.0, float("nan")))
    cm._structural_novelty = Mock(return_value=1.0)
    cm._infer_record_source = lambda record: record["source"]
    cm._is_failed_relax_record = lambda record: record.get("status") == "failed"
    cm._ensure_record_chemsys = lambda record: record["chem_sys"]
    cm._make_entries = lambda records: [
        SimpleNamespace(entry_id=f"{r['material_id']}_ms") for r in records
    ]
    cm.MaterialsProject2020Compatibility = lambda **kw: SimpleNamespace(
        process_entries=lambda entries, **kw: entries
    )
    cm.PhaseDiagram = lambda entries: SimpleNamespace(
        get_e_above_hull=lambda entry: 0.0
    )
    for name in (
        "_append_progress",
        "_log_stage2_relax_event",
        "_log_stage4_relax_event",
        "_log_stage4_postprocess_event",
        "_assert_mp_ok_count_equals_proc_count",
    ):
        setattr(cm, name, Mock())
    return SimpleNamespace(
        cm=cm, args=args, files=files, structures=structures, csvs=csvs, add=add
    )


@pytest.mark.parametrize("n_elements", [9, 10, 11])
def test_element_boundary(pipeline, n_elements):
    p = pipeline
    p.add(n_elements)
    p.cm.stage_prefetch_mp_spec(p.args)
    p.cm.stage_phase1_spec(p.args)
    skipped = n_elements >= 10
    for mock in (
        p.cm.fetch_mp_structures,
        p.cm._ensure_prop_relaxed,
        p.cm._collect_records_for_phase_with_log,
        p.cm._compute_ehull_for_proposal,
        p.cm._load_known_material_ids,
    ):
        assert mock.call_count == int(not skipped)
    screen = p.files[p.args.cache_dir / "screening_results.jsonl"][0]
    record = p.files[p.args.cache_dir / "proposal_relax_records.jsonl"][0]
    if skipped:
        assert record["status"] == "failed"
        assert record["relaxation_attempted"] is False
        assert record["file"] is None and record["E_final (eV)"] is None
        assert (
            record["error"]
            == record["skip_ehull_mp_reason"]
            == screen["skip_ehull_mp_reason"]
            == "mp_prefetch_element_limit"
        )
        assert math.isnan(screen["e_hull_mp"])
        assert p.files[p.args.cache_dir / "chemsys_to_fetch.json"] == []
    else:
        assert record["status"] == "relaxed"
        assert screen["e_hull_mp"] == 0.0
        assert p.files[p.args.cache_dir / "chemsys_to_fetch.json"] == [
            screen["chemsys"]
        ]


def test_counts_distinct_elements(pipeline):
    assert not pipeline.cm._exceeds_mp_element_limit("H-He-Li-Be-B-C-N-O-F-H-He")


def test_mp_cache_requires_each_elemental_reference(pipeline):
    p = pipeline
    cache = p.args.cache_dir / "mp_structures.json"
    p.files[cache] = {}
    p.cm._load_cached_entries.return_value = [
        {"entry_chemsys": "Cl-Na"},
        {"entry_chemsys": "Na"},
    ]
    assert not p.cm._has_cache_for_chemsys(cache, "Cl-Na")
    p.cm._load_cached_entries.return_value.append({"entry_chemsys": "Cl"})
    assert p.cm._has_cache_for_chemsys(cache, "Cl-Na")


@pytest.mark.parametrize("stage", ["all", "all_MP_only"])
@pytest.mark.parametrize("counts", [[10, 11], [9, 10, 11]])
def test_cli_pipeline_exports_with_failed_candidates(
    pipeline, monkeypatch, stage, counts
):
    p = pipeline
    for count in counts:
        p.add(count)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "calculate",
            "--stage",
            stage,
            "--p_dir",
            str(p.args.p_dir),
            "--cache_dir",
            str(p.args.cache_dir),
            "--r_dir",
            str(p.args.r_dir),
            "--s_dir",
            str(p.args.s_dir),
            "--rr_dir",
            str(p.args.rr_dir),
            "--global_relax_dir",
            str(p.cm.GLOBAL_RELAX_DIR),
            "--mp_api_key",
            "mock",
        ],
    )
    p.cm.main()
    df = p.csvs[p.args.rr_dir / "relax_results.csv"].set_index("fname")
    normal = 9 in counts
    assert p.cm.fetch_mp_structures.call_count == int(normal)
    assert p.cm._ensure_prop_relaxed.call_count == int(normal)
    assert p.cm._compute_ehull_for_proposal.call_count == int(normal)
    if normal:
        assert df.loc["p9", "e_hull_mp"] == 0.0
    if stage == "all":
        assert set(df.index) == ({"p9"} if normal else set())
        assert p.cm._compute_ehulls_phase2.call_count == int(normal)
        for count in (10, 11):
            assert not any(
                path.parent == p.args.s_dir / f"p{count}" for path in p.structures
            )
    else:
        assert set(df.index) == {f"p{n}" for n in counts}
        for count in (10, 11):
            assert df.loc[f"p{count}"].drop(labels="chemsys").isna().all()
            assert p.csvs[p.args.s_dir / f"p{count}_reference.csv"].empty


def test_rerun_overwrites_old_results_and_ignores_stale_relaxation(pipeline):
    p = pipeline
    normal, failed = p.add(9), p.add(10)
    cs = p.cm.chemsys_by_atomic_number(p.structures[failed].symbol_set)
    p.structures[p.args.r_dir / "p10_relaxed.vasp"] = p.structures[failed]
    p.files[p.args.cache_dir / "mp_prefetch_failures/input.json"] = {"complete": False}
    p.files[p.args.cache_dir / "screening_results.jsonl"] = [
        {"fname": "input/p10.vasp", "chemsys": cs, "e_hull_mp": 0.0}
    ]
    p.files[p.args.cache_dir / "proposal_relax_records.jsonl"] = [
        {"fname": "input/p10.vasp", "status": "relaxed"}
    ]
    p.files[p.args.cache_dir / "chemsys_to_fetch.json"] = [cs]
    # Even a pre-existing relaxed reference cache must not override the policy.
    p.files[p.cm.GLOBAL_RELAX_DIR / f"relax_results_{cs}.json.gz"] = []
    for _ in range(2):
        p.cm.stage_prefetch_mp_spec(p.args)
        p.cm.stage_phase1_spec(p.args)
        p.cm.stage_export_mp_only(p.args)
    assert p.cm.fetch_mp_structures.call_count == 1
    assert [call.args[0] for call in p.cm._ensure_prop_relaxed.call_args_list] == [
        normal,
        normal,
    ]
    assert len(p.files[p.args.cache_dir / "screening_results.jsonl"]) == 2
    assert len(p.files[p.args.cache_dir / "proposal_relax_records.jsonl"]) == 2
    assert cs not in p.files[p.args.cache_dir / "chemsys_to_fetch.json"]
    df = p.csvs[p.args.rr_dir / "relax_results.csv"].set_index("fname")
    assert df.loc["p10"].drop(labels="chemsys").isna().all()
    assert not any(
        call.args[0].name == "p10_relaxed.vasp"
        for call in p.cm.Structure.from_file.call_args_list
    )


def test_fetch_error_propagates_and_same_cache_can_resume(pipeline):
    p = pipeline
    p.add(9)
    p.add(10)
    p.cm.fetch_mp_structures.side_effect = [ConnectionError("test network error"), []]
    with pytest.raises(ConnectionError, match="test network error"):
        p.cm.stage_prefetch_mp_spec(p.args)
    p.cm.stage_prefetch_mp_spec(p.args)
    p.cm.stage_phase1_spec(p.args)
    assert p.cm.fetch_mp_structures.call_count == 2
    assert len(p.files[p.args.cache_dir / "screening_results.jsonl"]) == 2


def test_stage2_only_rejects_missing_prefetch_for_supported_input(pipeline):
    p = pipeline
    p.add(10)
    p.cm.stage_phase1_spec(p.args)
    p.add(9)
    with pytest.raises(RuntimeError, match="Missing MP fetch_done"):
        p.cm.stage_phase1_spec(p.args)


def test_empty_input(pipeline):
    p = pipeline
    p.cm.stage_prefetch_mp_spec(p.args)
    p.cm.stage_phase1_spec(p.args)
    assert not p.files
    p.cm.fetch_mp_structures.assert_not_called()
    p.cm._load_known_material_ids.assert_not_called()
