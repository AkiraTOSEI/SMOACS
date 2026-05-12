#!/usr/bin/env python3
from __future__ import annotations

"""
Generate initial structures from Materials Project (nsites 5-30) and save
POSCAR files plus one or more pickle files containing oxidation masks.

Example:
    python scripts/data_generations/generate_initials_mp_nsites5_30.py \
        --n 2048 \
        --seed 0 \
        --out_dir data/raw_data/initial_mp_nsites5_30 \
        --density_ref scripts/data_generations/density_ref_superhard.json \
        --mask_paths data/intermidiate/mask_dict_super_common.npz
"""

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from jarvis.core.atoms import pmg_to_atoms
from pymatgen.ext.matproj import MPRester, MPRestError

from src.utils.oxidation import get_atom_masks_for_oxidation_states


HALF_METAL_TARGETS = (
    "CrO2",
    "NiMnSb",
    "Co2MnSi",
    "Sr2FeMoO6",
    "Fe3O4",
    "HgCr2Se4",
    "Co2MnGe",
    "Co2CrAl",
    "Mn2VAl",
)
DEFAULT_HALF_METAL_E_HULL_MAX = 0.4


@dataclass
class MaskSpec:
    path: Path
    mask_d: dict
    radii_d: dict | None
    name: str | None


def _dataset_name(nsites_range: tuple[int, int]) -> str:
    return f"initial_mp_nsites{nsites_range[0]}_{nsites_range[1]}"


def _default_out_dir(script_path: Path, nsites_range: tuple[int, int]) -> Path:
    return script_path.resolve().parents[2] / "data" / "raw_data" / _dataset_name(nsites_range)


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _load_density_ref(path: Path) -> dict:
    ref = json.loads(path.read_text())
    print(
        "rho bounds:",
        f"p1={ref['p1']:.6f}",
        f"p99={ref['p99']:.6f}",
        f"rho_min={ref['rho_min']:.6f}",
        f"rho_max={ref['rho_max']:.6f}",
    )
    return ref


def _doc_get(doc, key: str):
    if isinstance(doc, dict):
        return doc.get(key)
    return getattr(doc, key, None)


def _fetch_half_metal_structures(
    api_key: str,
    nsites_range: tuple[int, int],
    e_hull_max: float,
) -> list[dict]:
    from mp_api.client import MPRester as MPApiMPRester
    from pymatgen.core import Composition

    if api_key is None:
        raise ValueError("MP API key is required.")

    fields = [
        "material_id",
        "formula_pretty",
        "composition",
        "energy_above_hull",
        "nsites",
        "symmetry",
        "deprecated",
        "structure",
    ]

    rows: list[dict] = []
    seen = set()
    with MPApiMPRester(api_key) as mpr:
        for formula in HALF_METAL_TARGETS:
            target = Composition(formula).reduced_composition
            chemsys = "-".join(sorted(el.symbol for el in target.elements))
            docs = mpr.materials.summary.search(
                chemsys=chemsys,
                energy_above_hull=(0, e_hull_max),
                num_sites=(nsites_range[0], nsites_range[1]),
                fields=fields,
            )

            found = 0
            for doc in docs:
                if getattr(doc, "deprecated", False):
                    continue

                comp = getattr(doc, "composition", None)
                if comp is None or comp.reduced_composition != target:
                    continue

                material_id = getattr(doc, "material_id", None)
                structure = getattr(doc, "structure", None)
                if material_id is None or structure is None:
                    continue
                material_id = str(material_id)
                if material_id in seen:
                    continue

                nsites = int(getattr(doc, "nsites", structure.num_sites))
                if not (nsites_range[0] <= nsites <= nsites_range[1]):
                    continue

                energy_above_hull = getattr(doc, "energy_above_hull", None)
                rows.append(
                    {
                        "material_id": material_id,
                        "structure": structure,
                        "nsites": nsites,
                        "source": "half_metal",
                        "input_formula": formula,
                        "energy_above_hull_eV_atom": (
                            float(energy_above_hull) if energy_above_hull is not None else None
                        ),
                    }
                )
                seen.add(material_id)
                found += 1

            if found == 0:
                print(f"[WARN] no half-metal sample found for {formula}; continue.")

    rows.sort(
        key=lambda x: (
            x["energy_above_hull_eV_atom"] if x["energy_above_hull_eV_atom"] is not None else float("inf"),
            x["material_id"],
        )
    )
    print(f"[{_now_str()}] fetched half-metal structures: {len(rows)}")
    return rows


def _fetch_mp_structures_bulk(
    api_key: str,
    nsites_range: tuple[int, int],
    pool_size: int,
    seed: int | None,
) -> list[dict]:
    if api_key is None:
        raise ValueError("MP API key is required.")
    rng = np.random.default_rng(seed)
    with MPRester(api_key) as mpr:
        kwargs = {
            "nsites_min": nsites_range[0],
            "nsites_max": nsites_range[1],
            "fields": ["material_id", "structure", "nsites"],
            "chunk_size": pool_size,
            "num_chunks": 1,
        }
        try:
            docs = mpr.summary.search(**kwargs)
        except TypeError:
            kwargs.pop("chunk_size", None)
            kwargs.pop("num_chunks", None)
            docs = mpr.summary.search(**kwargs)
        except MPRestError:
            fallback = {
                "nsites_min": nsites_range[0],
                "nsites_max": nsites_range[1],
            }
            docs = mpr.summary.search(**fallback)
        docs = list(docs)
    rng.shuffle(docs)
    out: list[dict] = []
    seen = set()
    for doc in docs:
        material_id = _doc_get(doc, "material_id")
        structure = _doc_get(doc, "structure")
        if material_id is None or structure is None:
            continue
        material_id = str(material_id)
        if material_id in seen:
            continue
        seen.add(material_id)
        out.append({
            "material_id": material_id,
            "structure": structure,
            "nsites": int(structure.num_sites),
            "source": "mp_bulk",
        })
    return out


def _batch_capacities(n: int, batch_size: int) -> list[int]:
    if n <= 0:
        raise ValueError("n must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    batch_count = math.ceil(n / batch_size)
    capacities = [batch_size for _ in range(batch_count)]
    capacities[-1] = n - batch_size * (batch_count - 1)
    return capacities


def _assign_balanced_sample_ids(
    docs: list[dict],
    n: int,
    batch_size: int,
    rng: np.random.Generator,
) -> list[dict]:
    if len(docs) < n:
        raise ValueError(f"Need {n} docs, got {len(docs)} after filtering.")

    fixed_docs = [dict(d) for d in docs if d.get("source") == "half_metal"]
    if len(fixed_docs) > n:
        raise ValueError(f"half-metal docs ({len(fixed_docs)}) exceed n ({n}).")

    pool_docs = [dict(d) for d in docs if d.get("source") != "half_metal"]
    need = n - len(fixed_docs)
    if len(pool_docs) < need:
        raise ValueError(
            f"Need {need} non-half-metal docs, got {len(pool_docs)} after filtering."
        )

    if need > 0:
        selected = rng.choice(len(pool_docs), size=need, replace=False)
        selected_docs = fixed_docs + [pool_docs[i] for i in selected]
    else:
        selected_docs = fixed_docs

    for doc in selected_docs:
        doc["nsites"] = int(doc["structure"].num_sites)

    capacities = _batch_capacities(n=n, batch_size=batch_size)
    batches: list[list[dict]] = [[] for _ in capacities]
    sums = [0 for _ in capacities]
    preferred_half_batches = [idx for idx, cap in enumerate(capacities) if cap == batch_size]
    if not preferred_half_batches:
        preferred_half_batches = list(range(len(capacities)))

    for doc in sorted(
        selected_docs,
        key=lambda d: (
            d.get("source") != "half_metal",
            -d["nsites"],
        ),
    ):
        best_idx = None
        best_score = None
        candidate_indices: list[int]
        if doc.get("source") == "half_metal":
            candidate_indices = [
                idx for idx in preferred_half_batches if len(batches[idx]) < capacities[idx]
            ]
            if not candidate_indices:
                candidate_indices = [
                    idx for idx in range(len(capacities)) if len(batches[idx]) < capacities[idx]
                ]
        else:
            candidate_indices = [
                idx for idx in range(len(capacities)) if len(batches[idx]) < capacities[idx]
            ]

        for idx in candidate_indices:
            cap = capacities[idx]
            if len(batches[idx]) >= cap:
                continue
            if doc.get("source") == "half_metal":
                score = (sums[idx], len(batches[idx]), idx)
            else:
                score = (sums[idx], len(batches[idx]))
            if best_score is None or score < best_score:
                best_score = score
                best_idx = idx
        if best_idx is None:
            raise RuntimeError("Failed to assign batch: no space left.")
        batches[best_idx].append(doc)
        sums[best_idx] += doc["nsites"]

    ordered: list[dict] = []
    sample_id = 0
    for batch_idx, batch in enumerate(batches):
        half = [doc for doc in batch if doc.get("source") == "half_metal"]
        regular = [doc for doc in batch if doc.get("source") != "half_metal"]
        rng.shuffle(half)
        rng.shuffle(regular)
        for doc in half + regular:
            doc = dict(doc)
            doc["sample_id"] = sample_id
            doc["batch_id"] = batch_idx
            ordered.append(doc)
            sample_id += 1
    if sample_id != n:
        raise RuntimeError(f"Assigned {sample_id} samples, expected {n}.")
    return ordered


def _load_mask_specs(mask_paths: Iterable[Path], radii_paths: Iterable[Path] | None) -> list[MaskSpec]:
    masks = []
    radii_map = {}
    if radii_paths is not None:
        for p in radii_paths:
            radii_map[p.stem.replace("ionic_radii_dict_", "mask_dict_")] = p
    for path in mask_paths:
        mask_d = dict(np.load(str(path), allow_pickle=True))
        radii_path = radii_map.get(path.stem)
        radii_d = None
        if radii_path is not None:
            radii_d = dict(np.load(str(radii_path), allow_pickle=True))
        masks.append(MaskSpec(path=path, mask_d=mask_d, radii_d=radii_d, name=path.name))
    return masks


def _build_record(
    structure,
    material_id: str,
    sample_id: int,
    mask_spec: MaskSpec,
    max_ox_states: int,
    ox_timeout_sec: float | None,
) -> dict:
    atoms = pmg_to_atoms(structure)
    atom_mask, radii, ox_used_mask, _site_ids = get_atom_masks_for_oxidation_states(
        atoms=atoms,
        mask_d=mask_spec.mask_d,
        max_ox_states=max_ox_states,
        crystal_system=None,
        radii_d=mask_spec.radii_d,
        timeout_sec=ox_timeout_sec,
        mask_name=mask_spec.name,
    )
    elements = [site.species_string for site in structure.sites]
    jid = f"Sample-{sample_id:04d}__{material_id}.vasp"
    return {
        "coords_frac": np.array(structure.frac_coords, dtype=np.float32),
        "lattice_matrix": np.array(structure.lattice.matrix, dtype=np.float32),
        "abc": np.array(structure.lattice.abc, dtype=np.float32),
        "angles": np.array(structure.lattice.angles, dtype=np.float32),
        "elements": elements,
        "atom_mask_for_all_ox_states": atom_mask.numpy().astype(np.float32),
        "radii_for_all_ox_states": radii.numpy().astype(np.float32),
        "ox_states_used_mask": ox_used_mask.numpy().astype(np.float32),
        "jid": jid,
        "material_id": material_id,
        "sample_id": sample_id,
        "nsites": structure.num_sites,
    }


def generate_initial_structures(
    n: int,
    seed: int | None,
    out_dir: Path,
    density_ref_path: Path,
    mask_specs: list[MaskSpec],
    api_key: str,
    nsites_range: tuple[int, int],
    max_ox_states: int,
    max_density_trials: int,
    max_shape_trials: int,
    pool_size: int,
    batch_size: int,
    pkl_dir: Path,
    export_vasp: bool,
    ox_timeout_sec: float | None,
    half_metal_samples: bool,
    half_metal_e_hull_max: float,
    max_failed_mp_ids: int,
    log_every_attempts: int,
) -> None:
    from lattice_randomize import randomize_lattice_with_density

    rng = np.random.default_rng(seed)
    if max_failed_mp_ids < 0:
        raise ValueError("max_failed_mp_ids must be >= 0.")
    if log_every_attempts <= 0:
        raise ValueError("log_every_attempts must be > 0.")
    ref = _load_density_ref(density_ref_path)
    rho_min = float(ref["rho_min"])
    rho_max = float(ref["rho_max"])

    half_docs: list[dict] = []
    if half_metal_samples:
        half_docs = _fetch_half_metal_structures(
            api_key=api_key,
            nsites_range=nsites_range,
            e_hull_max=half_metal_e_hull_max,
        )
        if len(half_docs) > n:
            raise RuntimeError(
                f"Fetched {len(half_docs)} half-metal structures but n={n}. Increase n."
            )
        print(
            f"[{_now_str()}] half-metal mode: include={len(half_docs)} "
            f"and fill from MP={n - len(half_docs)}"
        )

    docs = _fetch_mp_structures_bulk(
        api_key=api_key,
        nsites_range=nsites_range,
        pool_size=pool_size,
        seed=seed,
    )
    docs = [d for d in docs if nsites_range[0] <= d["nsites"] <= nsites_range[1]]

    if half_docs:
        half_ids = {d["material_id"] for d in half_docs}
        docs = [d for d in docs if d["material_id"] not in half_ids]
        docs = half_docs + docs

    pick_n = n + max_failed_mp_ids
    if len(docs) < pick_n:
        raise RuntimeError(
            f"Fetched {len(docs)} structures; need at least {pick_n} "
            f"(n={n} + max_failed_mp_ids={max_failed_mp_ids})."
        )

    ordered_docs = _assign_balanced_sample_ids(docs, n=pick_n, batch_size=batch_size, rng=rng)

    out_dir.mkdir(parents=True, exist_ok=True)
    pkl_dir.mkdir(parents=True, exist_ok=True)

    records_by_mask = {spec.path: [] for spec in mask_specs}

    written = 0
    failed = 0
    attempted = 0
    failed_material_ids: list[str] = []
    t_start = time.perf_counter()
    for doc in ordered_docs:
        if written >= n:
            break
        attempted += 1
        material_id = doc["material_id"]
        source_sample_id = int(doc["sample_id"])
        structure = doc["structure"]
        try:
            result = randomize_lattice_with_density(
                structure=structure,
                rho_min=rho_min,
                rho_max=rho_max,
                min_dist_threshold=1.0,
                rng=rng,
                max_density_trials=max_density_trials,
                max_shape_trials=max_shape_trials,
            )
        except Exception as exc:
            failed += 1
            failed_material_ids.append(material_id)
            print(
                f"[WARN] randomize failed for {material_id}: {exc} "
                f"(failed={failed}/{max_failed_mp_ids})"
            )
            if failed > max_failed_mp_ids:
                raise RuntimeError(
                    f"Randomize failures exceeded limit: failed={failed}, "
                    f"max_failed_mp_ids={max_failed_mp_ids}, last_material_id={material_id}"
                ) from exc
            if attempted % log_every_attempts == 0:
                elapsed = max(time.perf_counter() - t_start, 1e-9)
                speed = attempted / elapsed
                print(
                    f"[{_now_str()}] progress attempted={attempted}/{pick_n} "
                    f"success={written}/{n} failed={failed}/{max_failed_mp_ids} "
                    f"speed={speed:.2f} docs/s last_material_id={material_id}",
                    flush=True,
                )
            continue

        sample_id = written

        if export_vasp:
            jid = f"Sample-{sample_id:04d}__{material_id}.vasp"
            result.structure.to(fmt="poscar", filename=str(out_dir / jid))

        for spec in mask_specs:
            record = _build_record(
                structure=result.structure,
                material_id=material_id,
                sample_id=sample_id,
                mask_spec=spec,
                max_ox_states=max_ox_states,
                ox_timeout_sec=ox_timeout_sec,
            )
            record.update(
                {
                    "rho_min": rho_min,
                    "rho_max": rho_max,
                    "rho_target": result.rho_target,
                    "volume": result.structure.volume,
                    "min_dist": result.min_dist,
                    "iso_scale": result.iso_scale,
                    "shape_params": {"u1": result.u1, "u2": result.u2},
                    "retry_counts": result.retry_counts,
                    "source": doc.get("source"),
                    "batch_id": doc.get("batch_id"),
                    "source_sample_id": source_sample_id,
                    "input_formula": doc.get("input_formula"),
                    "energy_above_hull_eV_atom": doc.get("energy_above_hull_eV_atom"),
                }
            )
            records_by_mask[spec.path].append(record)

        written += 1
        if attempted % log_every_attempts == 0 or written % 50 == 0:
            elapsed = max(time.perf_counter() - t_start, 1e-9)
            speed = attempted / elapsed
            remain_success = n - written
            eta_sec = remain_success / max(speed, 1e-9)
            fail_budget_left = max_failed_mp_ids - failed
            print(
                f"[{_now_str()}] progress attempted={attempted}/{pick_n} "
                f"success={written}/{n} failed={failed}/{max_failed_mp_ids} "
                f"fail_budget_left={fail_budget_left} speed={speed:.2f} docs/s "
                f"eta={eta_sec:.1f}s last_material_id={material_id}",
                flush=True,
            )

    for spec in mask_specs:
        if len(records_by_mask[spec.path]) != n:
            raise RuntimeError(
                f"Generated {len(records_by_mask[spec.path])}/{n} records for {spec.path.name}. "
                f"failed={failed}, max_failed_mp_ids={max_failed_mp_ids}"
            )
        mask_stem = spec.path.stem
        pkl_path = pkl_dir / f"{_dataset_name(nsites_range)}__{mask_stem}.pkl"
        with pkl_path.open("wb") as fh:
            import pickle

            pickle.dump(records_by_mask[spec.path], fh, protocol=4)
        print(f"[{_now_str()}] saved {len(records_by_mask[spec.path])} records to {pkl_path}")

    if failed > 0:
        unique_failed = sorted(set(failed_material_ids))
        preview = ",".join(unique_failed[:10])
        print(
            f"[{_now_str()}] randomize failures summary: failed={failed}, "
            f"unique_failed={len(unique_failed)}, examples={preview}"
        )
    elapsed = max(time.perf_counter() - t_start, 1e-9)
    speed = attempted / elapsed
    print(
        f"[{_now_str()}] done attempted={attempted}/{pick_n} success={written}/{n} "
        f"failed={failed}/{max_failed_mp_ids} elapsed={elapsed:.1f}s speed={speed:.2f} docs/s"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MP initial structures (nsites 5-30).")
    parser.add_argument("--n", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--pkl_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "raw_data",
    )
    parser.add_argument(
        "--density_ref",
        type=Path,
        default=Path(__file__).resolve().parent / "density_ref_superhard.json",
    )
    parser.add_argument(
        "--mask_paths",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "intermidiate" / "mask_dict_super_common.npz"),
        help="Comma-separated mask npz paths.",
    )
    parser.add_argument(
        "--radii_paths",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data" / "intermidiate" / "ionic_radii_dict_super_common.npz"),
        help="Comma-separated radii npz paths (optional).",
    )
    parser.add_argument("--nsites_min", type=int, default=5)
    parser.add_argument("--nsites_max", type=int, default=30)
    parser.add_argument("--max_ox_states", type=int, default=10)
    parser.add_argument("--max_density_trials", type=int, default=30)
    parser.add_argument("--max_shape_trials", type=int, default=50)
    parser.add_argument(
        "--max_failed_mp_ids",
        type=int,
        default=30,
        help="Allow up to this many MP IDs to fail randomization while still filling n records.",
    )
    parser.add_argument(
        "--log_every_attempts",
        type=int,
        default=50,
        help="Print progress log every this many attempted structures.",
    )
    parser.add_argument("--api_key_env", type=str, default="MP_API_KEY")
    parser.add_argument(
        "--pool_size",
        type=int,
        default=0,
        help="MP bulk fetch size (0 uses 4*n).",
    )
    parser.add_argument("--batch_size", type=int, default=150)
    parser.add_argument(
        "--half_metal_samples",
        action="store_true",
        default=False,
        help="Fetch half-metal samples first, then fill remaining samples from MP.",
    )
    parser.add_argument(
        "--half_metal_e_hull_max",
        type=float,
        default=DEFAULT_HALF_METAL_E_HULL_MAX,
        help="Upper bound for half-metal energy_above_hull filtering.",
    )
    parser.add_argument(
        "--export_vasp",
        dest="export_vasp",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_export_vasp",
        dest="export_vasp",
        action="store_false",
    )
    parser.add_argument(
        "--ox_timeout_sec",
        type=float,
        default=60.0,
        help="Timeout for oxidation-state search in seconds.",
    )
    args = parser.parse_args()

    import os

    api_key = os.getenv(args.api_key_env)
    if api_key is None:
        raise ValueError(f"Environment variable {args.api_key_env} is required.")

    mask_paths = [Path(p.strip()) for p in args.mask_paths.split(",") if p.strip()]
    radii_paths = [Path(p.strip()) for p in args.radii_paths.split(",") if p.strip()]
    mask_specs = _load_mask_specs(mask_paths, radii_paths)

    nsites_range = (args.nsites_min, args.nsites_max)
    out_dir = args.out_dir or _default_out_dir(Path(__file__), nsites_range)

    pool_size = args.pool_size or max(args.n * 4, args.n)

    generate_initial_structures(
        n=args.n,
        seed=args.seed,
        out_dir=out_dir,
        density_ref_path=args.density_ref,
        mask_specs=mask_specs,
        api_key=api_key,
        nsites_range=nsites_range,
        max_ox_states=args.max_ox_states,
        max_density_trials=args.max_density_trials,
        max_shape_trials=args.max_shape_trials,
        pool_size=pool_size,
        batch_size=args.batch_size,
        pkl_dir=args.pkl_dir,
        export_vasp=args.export_vasp,
        ox_timeout_sec=args.ox_timeout_sec,
        half_metal_samples=args.half_metal_samples,
        half_metal_e_hull_max=args.half_metal_e_hull_max,
        max_failed_mp_ids=args.max_failed_mp_ids,
        log_every_attempts=args.log_every_attempts,
    )


if __name__ == "__main__":
    main()
