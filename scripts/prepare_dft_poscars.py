#!/usr/bin/env python3
# coding: utf-8

import argparse
import gzip
import json
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Element, Structure
from pymatgen.io.vasp import Poscar


def _load_calc_module(calc_path: Path):
    import importlib.util
    import sys
    import types

    def _install_stub(name: str, attrs: Dict[str, object] | None = None):
        mod = types.ModuleType(name)
        if attrs:
            for key, value in attrs.items():
                setattr(mod, key, value)
        sys.modules[name] = mod
        return mod

    if "mattersim" not in sys.modules:
        _install_stub("mattersim.forcefield", {"MatterSimCalculator": object})
        _install_stub("mattersim.applications")
        _install_stub("mattersim.applications.relax", {"Relaxer": object})
        _install_stub("mattersim", {"forcefield": sys.modules["mattersim.forcefield"]})

    spec = importlib.util.spec_from_file_location("calc", calc_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _read_poscar_elements(path: Path) -> List[str]:
    if path.name.endswith(".vasp.gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        poscar = Poscar.from_str(content)
    else:
        poscar = Poscar.from_file(path)
    return [str(el) for el in poscar.site_symbols]


def _write_poscar_copy(src: Path, dst: Path) -> Path:
    if src.name.endswith(".vasp.gz"):
        with gzip.open(src, "rt", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        poscar = Poscar.from_str(content)
        out_path = dst.with_suffix("") if dst.suffix == ".gz" else dst
        if out_path.name.endswith(".vasp.gz"):
            out_path = Path(out_path.name[:-3])
        poscar.write_file(out_path)
        return out_path
    shutil.copy(src, dst)
    return dst

def _chemsys_by_atomic_number(elements: List[str]) -> str:
    return "-".join(sorted(elements, key=lambda e: Element(e).Z, reverse=True))


def _collect_target_poscars(target_dir: Path) -> List[Path]:
    poscar_map: Dict[str, Path] = {}
    for p in target_dir.iterdir():
        if p.is_file() and (p.name.endswith(".vasp") or p.name.endswith(".vasp.gz")):
            stem = p.name.replace(".vasp.gz", "").replace(".vasp", "")
            prev = poscar_map.get(stem)
            if prev is None:
                poscar_map[stem] = p
            else:
                # Prefer uncompressed .vasp when both exist.
                if prev.name.endswith(".vasp.gz") and p.name.endswith(".vasp"):
                    poscar_map[stem] = p
    return sorted(poscar_map.values())

def _resolve_relax_record_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.exists():
        return path
    path_str = str(path_value)
    marker = f"{os.sep}SMOACS{os.sep}"
    if marker in path_str:
        suffix = path_str.split(marker, 1)[1]
        candidate = Path(__file__).resolve().parents[1] / suffix
        if candidate.exists():
            return candidate
    return path

def _filter_by_structural_equivalence_min_ehull(
    df: pd.DataFrame,
    *,
    chem_col: str = "chem_sys",
    path_col: str = "path",
    ehull_col: str = "e_hull",
) -> pd.DataFrame:
    required = {chem_col, path_col, ehull_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {sorted(missing)}")

    sm = StructureMatcher()
    struct_cache: Dict[str, Structure] = {}

    def _get_structure(p: str) -> Structure:
        if p not in struct_cache:
            struct_cache[p] = Structure.from_file(p)
        return struct_cache[p]

    keep_indices: List[int] = []
    for chem_sys, g in df.groupby(chem_col, sort=False):
        idx_list = list(g.index)
        if len(idx_list) <= 1:
            keep_indices.extend(idx_list)
            continue

        parent = {i: i for i in idx_list}
        rank = {i: 0 for i in idx_list}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if rank[ra] < rank[rb]:
                parent[ra] = rb
            elif rank[ra] > rank[rb]:
                parent[rb] = ra
            else:
                parent[rb] = ra
                rank[ra] += 1

        for i in range(len(idx_list)):
            idx_i = idx_list[i]
            s_i = _get_structure(str(df.at[idx_i, path_col]))
            for j in range(i + 1, len(idx_list)):
                idx_j = idx_list[j]
                s_j = _get_structure(str(df.at[idx_j, path_col]))
                if sm.fit(s_i, s_j):
                    union(idx_i, idx_j)

        groups: Dict[int, List[int]] = {}
        for idx in idx_list:
            root = find(idx)
            groups.setdefault(root, []).append(idx)

        for comp in groups.values():
            best_idx = min(
                comp,
                key=lambda k: (df.at[k, ehull_col], str(df.at[k, path_col])),
            )
            keep_indices.append(best_idx)

    return df.loc[sorted(keep_indices)]

def _select_and_save_poscars_for_dft(
    json_path: Path,
    dft_ehull_thres: float,
    chemsys: str,
    relax_result_db_path: Path,
    recreate: bool,
) -> Path:
    chemsys_dir = relax_result_db_path.parent / f"DFT_POSCARS/{chemsys}"
    chemsys_dir.mkdir(parents=True, exist_ok=True)

    if (not recreate) and any(chemsys_dir.glob("*.vasp")):
        return chemsys_dir

    if recreate:
        shutil.rmtree(chemsys_dir, ignore_errors=True)
        chemsys_dir.mkdir(parents=True, exist_ok=True)

    with gzip.open(json_path, "rt", encoding="utf-8") as fp:
        mp_alex_collection = json.load(fp)
    df = pd.DataFrame(mp_alex_collection)
    if df.empty or "e_hull" not in df:
        return chemsys_dir

    list_for_df = []
    for poscar_path in df[df["e_hull"] < dft_ehull_thres].file.values:
        resolved_path = _resolve_relax_record_path(poscar_path)
        if not resolved_path.exists():
            continue
        list_for_df.append(
            {
                "e_hull": float(df[df["file"] == poscar_path].e_hull.values[0]),
                "chem_sys": _chemsys_by_atomic_number(_read_poscar_elements(resolved_path)),
                "mp_id": resolved_path.name.replace(".vasp", ""),
                "path": str(resolved_path),
            }
        )

    if not list_for_df:
        return chemsys_dir

    filtered = _filter_by_structural_equivalence_min_ehull(pd.DataFrame(list_for_df))
    filtered.to_csv(chemsys_dir / "filtered_poscars.csv", index=False)

    for poscar_path in filtered.path.values:
        shutil.copy(poscar_path, chemsys_dir / Path(poscar_path).name)

    return chemsys_dir


def _alexandria_ehull(attr: Dict[str, float]) -> float:
    return attr.get(
        "_alexandria_hull_distance",
        attr.get("_alexandria_phase_separation_energy", float("inf")),
    )


def _ensure_relax_results_with_alexandria(
    calc,
    chemsys: str,
    relax_results_path: Path,
    alexandria_ehull_max: float,
    max_workers: int,
    page_limit: int,
) -> None:
    base_url = "https://alexandria.icams.rub.de/pbe"
    chemsys_elems = chemsys.split("-")
    data_light = calc._gather_chemsys_light(
        base_url,
        chemsys_elems,
        require_3d=True,
        only_known=True,
        max_k=None,
        max_workers=max_workers,
        page_limit=page_limit,
    )
    candidates = []
    for item in data_light:
        attrs = item.get("attributes", {})
        e_hull = _alexandria_ehull(attrs)
        if e_hull <= alexandria_ehull_max:
            candidates.append(
                {
                    "id": item["id"],
                    "formula": attrs.get("chemical_formula_reduced", ""),
                    "e_hull": float(e_hull),
                }
            )

    if not candidates:
        print(f"[Alexandria] No candidates for {chemsys}.")
        return

    ids = [c["id"] for c in candidates]
    structures_map = calc._fetch_structures_by_ids(
        base_url, ids, page_limit=page_limit
    )
    alex_dir = relax_results_path.parent / "alexandria_poscars" / chemsys
    alex_dir.mkdir(parents=True, exist_ok=True)

    alex_records = []
    for c in candidates:
        struct_entry = structures_map.get(c["id"])
        if not struct_entry:
            continue
        try:
            pmg = calc._entry_to_pmgstructure(struct_entry)
        except Exception:
            continue
        poscar_path = alex_dir / f"{c['id']}_relaxed.vasp"
        Poscar(pmg).write_file(poscar_path)
        alex_records.append(
            {
                "material_id": c["id"],
                "formula": c["formula"],
                "e_hull": c["e_hull"],
                "file": str(poscar_path),
            }
        )

    if not alex_records:
        print(f"[Alexandria] No structures written for {chemsys}.")
        return

    existing = []
    if relax_results_path.exists():
        with gzip.open(relax_results_path, "rt") as f:
            existing = json.load(f)

    merged: Dict[str, Dict[str, object]] = {}
    for rec in existing:
        mid = rec.get("material_id")
        if mid:
            merged[mid] = rec
    for rec in alex_records:
        merged[rec["material_id"]] = rec

    with gzip.open(relax_results_path, "wt") as f:
        json.dump(list(merged.values()), f)

    print(
        f"[Alexandria] Added {len(alex_records)} records to "
        f"{relax_results_path.name}."
    )


def _ensure_chemsys_dft_poscars(
    calc,
    chemsys: str,
    relax_result_db_path: Path,
    dft_ehull_thres: float,
    alexandria_ehull_max: float,
    max_workers: int,
    page_limit: int,
    allow_alexandria: bool,
) -> Path:
    relax_results_path = (
        relax_result_db_path.parent / f"relax_results_{chemsys}.json.gz"
    )
    chemsys_dir = relax_result_db_path.parent / f"DFT_POSCARS/{chemsys}"
    chemsys_dir.mkdir(parents=True, exist_ok=True)

    has_agm = any(p.name.startswith("agm") for p in chemsys_dir.glob("agm*.vasp"))
    if not has_agm and allow_alexandria:
        print(f"[Alexandria] agm*.vasp missing for {chemsys}. Fetching...")
        _ensure_relax_results_with_alexandria(
            calc,
            chemsys,
            relax_results_path,
            alexandria_ehull_max,
            max_workers,
            page_limit,
        )
        _select_and_save_poscars_for_dft(
            json_path=relax_results_path,
            dft_ehull_thres=dft_ehull_thres,
            chemsys=chemsys,
            relax_result_db_path=relax_result_db_path,
            recreate=True,
        )
    elif not has_agm and not allow_alexandria:
        print(f"[WARN] agm*.vasp missing for {chemsys}, but Alexandria disabled.")

    return chemsys_dir


def _write_material_summary(
    calc,
    target_poscar_path: Path,
    chemsys_dir: Path,
    output_dir: Path,
) -> Path:
    filtered_csv = chemsys_dir / "filtered_poscars.csv"
    if not filtered_csv.exists():
        print(f"[WARN] Missing {filtered_csv}. material_summary.csv will be minimal.")
        exising_mate_df = pd.DataFrame(columns=["e_hull", "chem_sys", "mp_id", "path"])
    else:
        exising_mate_df = pd.read_csv(filtered_csv)

    chem_sys = calc.chemsys_by_atomic_number(_read_poscar_elements(target_poscar_path))
    proposed_row = pd.DataFrame(
        [
            {
                "e_hull": 999.0,
                "chem_sys": chem_sys,
                "mp_id": target_poscar_path.name.replace(".vasp", "").replace(".gz", ""),
                "path": str(output_dir / target_poscar_path.name.replace(".gz", "")),
            }
        ]
    )
    dft_df = pd.concat([proposed_row, exising_mate_df], ignore_index=True)
    out_csv = output_dir / "material_summary.csv"
    dft_df.to_csv(out_csv, index=False)
    return out_csv


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare DFT_poscars for target POSCARs without MatterSim relaxation."
    )
    parser.add_argument(
        "--target-poscars-dir",
        required=True,
        type=str,
        help="Directory containing target POSCARs (.vasp or .vasp.gz).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="./DFT_poscars",
        help="Output directory to store per-target DFT_poscars.",
    )
    parser.add_argument(
        "--relax-db-path",
        type=str,
        default="data/mp_mattersim_relaxed/relax_results.json",
        help="Path to relax_results.json (used to locate DFT_POSCARS and relax_results_*.json.gz).",
    )
    parser.add_argument(
        "--dft-ehull-thres",
        type=float,
        default=0.2,
        help="e_hull threshold for selecting DFT POSCARs.",
    )
    parser.add_argument(
        "--alexandria-ehull-max",
        type=float,
        default=1.0,
        help="Alexandria e_hull max for candidate selection.",
    )
    parser.add_argument(
        "--no-alexandria",
        action="store_true",
        help="Disable Alexandria fetch even when agm*.vasp is missing.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Max workers for Alexandria API requests.",
    )
    parser.add_argument(
        "--page-limit",
        type=int,
        default=200,
        help="Alexandria API page limit.",
    )
    args = parser.parse_args()

    target_dir = Path(args.target_poscars_dir)
    out_dir = Path(args.out_dir)
    relax_db_path = Path(args.relax_db_path)
    calc_path = Path(__file__).resolve().parent / "calculate_mattersim_convex_hull_energy.py"

    calc = _load_calc_module(calc_path)

    poscars = _collect_target_poscars(target_dir)
    if not poscars:
        print(f"No POSCARs found in {target_dir}")
        return 1

    chemsys_map: Dict[str, List[Path]] = {}
    for p in poscars:
        elems = _read_poscar_elements(p)
        chemsys = calc.chemsys_by_atomic_number(elems)
        chemsys_map.setdefault(chemsys, []).append(p)

    for chemsys, paths in chemsys_map.items():
        print(f"[chemsys] {chemsys}: {len(paths)} target POSCARs")
        chemsys_dir = _ensure_chemsys_dft_poscars(
            calc,
            chemsys,
            relax_db_path,
            args.dft_ehull_thres,
            args.alexandria_ehull_max,
            args.max_workers,
            args.page_limit,
            not args.no_alexandria,
        )

        for target_poscar_path in paths:
            stem = target_poscar_path.name.replace(".vasp.gz", "").replace(".vasp", "")
            target_out_dir = out_dir / stem
            target_out_dir.mkdir(parents=True, exist_ok=True)

            # Copy proposed POSCAR
            dst_name = target_poscar_path.name.replace(".gz", "")
            _write_poscar_copy(target_poscar_path, target_out_dir / dst_name)

            # Copy DFT POSCARs for the chemsys
            for poscar_path in chemsys_dir.glob("*.vasp"):
                shutil.copy(poscar_path, target_out_dir / poscar_path.name)

            _write_material_summary(calc, target_poscar_path, chemsys_dir, target_out_dir)
            print(f"[done] {target_out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
