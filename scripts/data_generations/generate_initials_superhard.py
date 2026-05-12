#!/usr/bin/env python3
from __future__ import annotations

"""
Generate initial structures from Materials Project and save npz/json outputs.

Example:
    python scripts/data_generations/generate_initials_superhard.py \\
        --n 20 \\
        --seed 0 \\
        --out_dir data/raw_data/initial_superhard \\
        --density_ref scripts/data_generations/density_ref_superhard.json \\
        --mask_path data/intermidiate/mask_superhard_materials.npz \\
        --api_key_env MP_API_KEY
"""

import json
<<<<<<< ours
import pickle
=======
>>>>>>> theirs
import shutil
import time
from pathlib import Path

import numpy as np

from lattice_randomize import randomize_lattice_with_density
from mp_fetch import fetch_mp_structures, load_cached_structures
from wyckoff_encoding import encode_structure_no_wyckoff


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


def _now_str() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def _filter_docs(
    docs: list[dict],
    allowed_elements: set[str] | None,
    nsites_range: tuple[int, int],
) -> list[dict]:
    filtered = []
    for doc in docs:
        structure = doc["structure"]
        if allowed_elements:
            elements = {el.symbol for el in structure.composition.elements}
            if not elements.issubset(allowed_elements):
                continue
        nsites = structure.num_sites
        if not (nsites_range[0] <= nsites <= nsites_range[1]):
            continue
        filtered.append(doc)
    return filtered


def _record_timing(timings: dict[str, list[float]], key: str, value: float) -> None:
    timings.setdefault(key, []).append(value)


def generate_initial_structures(
    n: int,
    seed: int | None,
    out_dir: Path,
    density_ref_path: Path,
    mask_path: Path,
    api_key: str | None,
    allowed_elements: set[str] | None,
    nsites_range: tuple[int, int],
    max_ox_states: int,
    max_density_trials: int,
    max_shape_trials: int,
    cache_path: Path | None,
    cache_only: bool,
    clean_out_dir: bool,
<<<<<<< ours
    mp_pool_size: int,
    mp_chunk_size: int | None,
    profile: bool,
    verbose: bool,
    export_vasp: bool,
    pkl_path: Path,
    timing_out: Path | None = None,
    print_debug: bool = False,
    ox_timeout_sec: float | None = None,
=======
>>>>>>> theirs
) -> None:
    run_start = time.perf_counter()
    rng = np.random.default_rng(seed)
    ref = _load_density_ref(density_ref_path)
    rho_min = float(ref["rho_min"])
    rho_max = float(ref["rho_max"])

    def log(msg: str) -> None:
        if profile or verbose:
            print(msg)

    timings: dict[str, list[float]] = {}
    counters = {
        "attempted": 0,
        "accepted": 0,
        "randomize_fail": 0,
        "encode_fail": 0,
        "save_fail": 0,
        "min_dist_fail": 0,
        "rho_fail": 0,
        "density_trials": 0,
        "shape_trials": 0,
        "mp_fetch_calls": 0,
        "mp_docs_total": 0,
    }
    mp_stats: dict[str, float] = {}
    seen_material_ids: set[str] = set()

    docs: list[dict] = []
    records: list[dict] = []
    if cache_path is not None:
        if profile or verbose:
            print(f"[{_now_str()}] cache load start: {cache_path}", flush=True)
        t_cache = time.perf_counter()
        docs = load_cached_structures(cache_path, allowed_elements, nsites_range)
        t_cache = time.perf_counter() - t_cache
        _record_timing(timings, "cache_load", t_cache)
        log(f"Loaded {len(docs)} cached structures from {cache_path} in {t_cache:.2f}s")

    if cache_only and len(docs) < n:
        raise RuntimeError(
            f"Cache has {len(docs)} structures; need {n}. "
            "Re-run with --cache_only false to allow MP fetch."
        )

    if clean_out_dir and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = 0
    pool_index = 0
    docs_pool = docs
    if mp_pool_size <= 0:
        mp_pool_size = max(n * 4, n)

    while written < n:
        if pool_index >= len(docs_pool):
            if cache_only:
                break
            log(f"[{_now_str()}] MP fetch start (pool_size={mp_pool_size})")
            t_fetch_start = time.perf_counter()
            counters["mp_fetch_calls"] += 1
            fetched = fetch_mp_structures(
                api_key=api_key,
                n=mp_pool_size,
                allowed_elements=allowed_elements,
                nsites_range=nsites_range,
                cache_path=cache_path,
                chunk_size=mp_chunk_size,
                stats=mp_stats if profile else None,
                verbose=profile or verbose,
            )
            t_fetch = time.perf_counter() - t_fetch_start
            _record_timing(timings, "mp_fetch", t_fetch)
            log(
                f"[{_now_str()}] MP fetch done: {len(fetched)} docs in {t_fetch:.1f}s"
            )
            t_filter = time.perf_counter()
            docs_pool = _filter_docs(fetched, allowed_elements, nsites_range)
            t_filter = time.perf_counter() - t_filter
            _record_timing(timings, "local_filter", t_filter)
            counters["mp_docs_total"] += len(docs_pool)
            pool_index = 0
            if not docs_pool:
                raise RuntimeError("MP fetch returned no usable documents.")

        doc = docs_pool[pool_index]
        pool_index += 1
        material_id = doc["material_id"]
        if material_id in seen_material_ids:
            continue
        seen_material_ids.add(material_id)

        if verbose or print_debug:
            print(
                f"[{_now_str()}] attempt {counters['attempted']+1} "
                f"material_id={material_id} nsites={doc['structure'].num_sites}",
                flush=True,
            )
        counters["attempted"] += 1
        t_item = time.perf_counter()
        structure = doc["structure"]
        randomize_profile = {} if profile else None
        rand_elapsed = 0.0
        try:
            if print_debug:
                print(
                    f"[timing-start] material_id={material_id} stage=randomize",
                    flush=True,
                )
            t_rand = time.perf_counter()
            result = randomize_lattice_with_density(
                structure=structure,
                rho_min=rho_min,
                rho_max=rho_max,
                min_dist_threshold=1.0,
                rng=rng,
                max_density_trials=max_density_trials,
                max_shape_trials=max_shape_trials,
                profile=randomize_profile,
                verbose=verbose,
            )
            rand_elapsed = time.perf_counter() - t_rand
            _record_timing(timings, "randomize_total", rand_elapsed)
        except Exception as exc:
            counters["randomize_fail"] += 1
            if print_debug:
                print(
                    f"[timing-fail] material_id={material_id} stage=randomize "
                    f"elapsed={time.perf_counter()-t_item:.3f}s error={type(exc).__name__}",
                    flush=True,
                )
            if verbose:
                print(
                    f"[{_now_str()}] randomize failed after "
                    f"{time.perf_counter()-t_item:.2f}s: {exc}",
                    flush=True,
                )
            if randomize_profile:
                counters["min_dist_fail"] += int(randomize_profile.get("min_dist_fail", 0))
                counters["rho_fail"] += int(randomize_profile.get("rho_fail", 0))
                counters["density_trials"] += int(randomize_profile.get("density_trials", 0))
                counters["shape_trials"] += int(randomize_profile.get("shape_trials", 0))
                _record_timing(
                    timings,
                    "min_dist_time",
                    float(randomize_profile.get("min_dist_time", 0.0)),
                )
            continue

        if randomize_profile:
            counters["min_dist_fail"] += int(randomize_profile.get("min_dist_fail", 0))
            counters["rho_fail"] += int(randomize_profile.get("rho_fail", 0))
            counters["density_trials"] += int(randomize_profile.get("density_trials", 0))
            counters["shape_trials"] += int(randomize_profile.get("shape_trials", 0))
            _record_timing(
                timings,
                "min_dist_time",
                float(randomize_profile.get("min_dist_time", 0.0)),
            )

        encode_profile = {} if profile else None
        encode_elapsed = 0.0
        try:
            if print_debug:
                print(
                    f"[timing-start] material_id={material_id} stage=encode",
                    flush=True,
                )
            t_encode = time.perf_counter()
            npz_data, json_meta = encode_structure_no_wyckoff(
                structure=result.structure,
                material_id=material_id,
                sample_id=written,
                ox_mask_path=mask_path,
                max_ox_states=max_ox_states,
                ox_timeout_sec=ox_timeout_sec,
                profile=encode_profile,
            )
            encode_elapsed = time.perf_counter() - t_encode
            _record_timing(timings, "encode_total", encode_elapsed)
        except Exception as exc:
            counters["encode_fail"] += 1
            if print_debug:
                print(
                    f"[timing-fail] material_id={material_id} stage=encode "
                    f"elapsed={time.perf_counter()-t_item:.3f}s error={type(exc).__name__}",
                    flush=True,
                )
            if verbose:
                print(
                    f"[{_now_str()}] encode failed after "
                    f"{time.perf_counter()-t_item:.2f}s: {exc}",
                    flush=True,
                )
            continue

        if encode_profile:
            _record_timing(
                timings,
                "pmg_to_atoms_time",
                float(encode_profile.get("pmg_to_atoms_time", 0.0)),
            )
            _record_timing(
                timings,
                "mask_load_time",
                float(encode_profile.get("mask_load_time", 0.0)),
            )
            _record_timing(
                timings,
                "mask_build_time",
                float(encode_profile.get("mask_build_time", 0.0)),
            )
            _record_timing(
                timings,
                "ox_state_search_time",
                float(encode_profile.get("ox_state_search_time", 0.0)),
            )
            _record_timing(
                timings,
                "ox_mask_loop_time",
                float(encode_profile.get("ox_mask_loop_time", 0.0)),
            )
            _record_timing(
                timings,
                "ox_tensor_time",
                float(encode_profile.get("ox_tensor_time", 0.0)),
            )
            if verbose:
                ox_time = float(encode_profile.get("ox_state_search_time", 0.0))
                if encode_elapsed >= 1.0 or ox_time >= 1.0:
                    print(
                        f"[{_now_str()}] encode profile material_id={material_id} "
                        f"encode_total={encode_elapsed:.2f}s "
                        f"ox_state_search={ox_time:.2f}s "
                        f"elements={json_meta.get('elements')}",
                        flush=True,
                    )

        json_meta.update(
            {
                "sample_id": written,
                "rho_min": rho_min,
                "rho_max": rho_max,
                "rho_target": result.rho_target,
                "volume": result.structure.volume,
                "min_dist": result.min_dist,
                "iso_scale": result.iso_scale,
                "shape_params": {"u1": result.u1, "u2": result.u2},
                "retry_counts": result.retry_counts,
            }
        )

        base = f"sample_{written:04d}__{material_id}"
        t_save = time.perf_counter()
        save_elapsed = 0.0
        try:
            if print_debug:
                print(
                    f"[timing-start] material_id={material_id} stage=save",
                    flush=True,
                )
            if export_vasp:
                result.structure.to(fmt="poscar", filename=str(out_dir / f"{base}.vasp"))
        except Exception as exc:
            counters["save_fail"] += 1
            if print_debug:
                print(
                    f"[timing-fail] material_id={material_id} stage=save "
                    f"elapsed={time.perf_counter()-t_item:.3f}s error={type(exc).__name__}",
                    flush=True,
                )
            if verbose:
                print(
                    f"[{_now_str()}] save failed after "
                    f"{time.perf_counter()-t_item:.2f}s: {exc}",
                    flush=True,
                )
            continue
        save_elapsed = time.perf_counter() - t_save
        _record_timing(timings, "save_total", save_elapsed)

        record = {}
        record.update(json_meta)
        record.update(npz_data)
        records.append(record)

        written += 1
        counters["accepted"] += 1
        t_elapsed = time.perf_counter() - t_item
        if print_debug:
            print(
                f"[timing] material_id={material_id} nsites={structure.num_sites} "
                f"randomize={rand_elapsed:.3f}s encode={encode_elapsed:.3f}s "
                f"save={save_elapsed:.3f}s total={t_elapsed:.3f}s",
                flush=True,
            )
        log(f"wrote {written}/{n} ({material_id}) in {t_elapsed:.2f}s")

    if written < n:
        raise RuntimeError(f"Only generated {written} structures; need {n}.")

    t_pkl = time.perf_counter()
    pkl_path.parent.mkdir(parents=True, exist_ok=True)
    with pkl_path.open("wb") as fh:
        pickle.dump(records, fh, protocol=4)
    _record_timing(timings, "pkl_save", time.perf_counter() - t_pkl)
    log(f"saved {len(records)} records to {pkl_path}")

    if profile:
        def _mean(values: list[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        def _max(values: list[float]) -> float:
            return max(values) if values else 0.0

        total_attempts = counters["attempted"]
        print("---- profile summary ----")
        print(
            "MP fetch calls:",
            counters["mp_fetch_calls"],
            "MP summary calls:",
            int(mp_stats.get("mp_summary_calls", 0)),
            "MP summary time:",
            f"{mp_stats.get('mp_summary_time', 0.0):.2f}s",
        )
        print(
            "docs fetched:",
            counters["mp_docs_total"],
            "attempted:",
            total_attempts,
            "accepted:",
            counters["accepted"],
        )
        if total_attempts:
            print(
                "acceptance rate:",
                f"{counters['accepted'] / total_attempts:.3f}",
            )
        print(
            "failures:",
            "randomize",
            counters["randomize_fail"],
            "encode",
            counters["encode_fail"],
            "save",
            counters["save_fail"],
        )
        print(
            "randomize failures:",
            "min_dist",
            counters["min_dist_fail"],
            "rho",
            counters["rho_fail"],
        )
        print(
            "retry totals:",
            "density_trials",
            counters["density_trials"],
            "shape_trials",
            counters["shape_trials"],
        )

        stages = {
            "mp_fetch_total": sum(timings.get("mp_fetch", [])),
            "local_filter_total": sum(timings.get("local_filter", [])),
            "randomize_total": sum(timings.get("randomize_total", [])),
            "encode_total": sum(timings.get("encode_total", [])),
            "save_total": sum(timings.get("save_total", [])),
            "pkl_save_total": sum(timings.get("pkl_save", [])),
            "min_dist_time": sum(timings.get("min_dist_time", [])),
            "mask_load_time": sum(timings.get("mask_load_time", [])),
            "mask_build_time": sum(timings.get("mask_build_time", [])),
            "ox_state_search_time": sum(timings.get("ox_state_search_time", [])),
            "ox_mask_loop_time": sum(timings.get("ox_mask_loop_time", [])),
            "ox_tensor_time": sum(timings.get("ox_tensor_time", [])),
        }
        top3 = sorted(stages.items(), key=lambda x: x[1], reverse=True)[:3]
        print("top3 time sinks:", ", ".join(f"{k}={v:.2f}s" for k, v in top3))
        print(
            "randomize avg/max:",
            f"{_mean(timings.get('randomize_total', [])):.3f}s",
            f"{_max(timings.get('randomize_total', [])):.3f}s",
        )
        print(
            "encode avg/max:",
            f"{_mean(timings.get('encode_total', [])):.3f}s",
            f"{_max(timings.get('encode_total', [])):.3f}s",
        )
        print(
            "save avg/max:",
            f"{_mean(timings.get('save_total', [])):.3f}s",
            f"{_max(timings.get('save_total', [])):.3f}s",
        )

    if timing_out is not None:
        from timing_report import write_timing_report

        run_total = time.perf_counter() - run_start
        stage_totals = {
            "cache_load_total": sum(timings.get("cache_load", [])),
            "mp_fetch_total": sum(timings.get("mp_fetch", [])),
            "local_filter_total": sum(timings.get("local_filter", [])),
            "randomize_total": sum(timings.get("randomize_total", [])),
            "encode_total": sum(timings.get("encode_total", [])),
            "save_total": sum(timings.get("save_total", [])),
            "pkl_save_total": sum(timings.get("pkl_save", [])),
            "total": run_total,
        }
        stage_shares = {}
        if run_total > 0:
            for key, value in stage_totals.items():
                if key != "total":
                    stage_shares[key] = value / run_total

        payload = {
            "run": {
                "n": n,
                "seed": seed,
                "nsites_min": nsites_range[0],
                "nsites_max": nsites_range[1],
                "cache_only": cache_only,
                "mp_pool_size": mp_pool_size,
                "mp_chunk_size": mp_chunk_size,
            },
            "counts": {
                "attempted": counters["attempted"],
                "accepted": counters["accepted"],
                "randomize_fail": counters["randomize_fail"],
                "encode_fail": counters["encode_fail"],
                "save_fail": counters["save_fail"],
            },
            "mp_stats": dict(mp_stats),
            "stage_totals_sec": {k: round(v, 6) for k, v in stage_totals.items()},
            "stage_shares": {k: round(v, 6) for k, v in stage_shares.items()},
            "details_sec": {
                "min_dist_time": round(sum(timings.get("min_dist_time", [])), 6),
                "mask_load_time": round(sum(timings.get("mask_load_time", [])), 6),
                "mask_build_time": round(sum(timings.get("mask_build_time", [])), 6),
                "ox_state_search_time": round(sum(timings.get("ox_state_search_time", [])), 6),
                "ox_mask_loop_time": round(sum(timings.get("ox_mask_loop_time", [])), 6),
                "ox_tensor_time": round(sum(timings.get("ox_tensor_time", [])), 6),
            },
        }
        write_timing_report(timing_out, payload)


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate initial structures.")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--out_dir",
        type=Path,
<<<<<<< ours
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "raw_data"
        / "initial_superhard",
=======
        default=Path(__file__).resolve().parents[2] / "data" / "raw_data" / "superhard",
>>>>>>> theirs
    )
    parser.add_argument(
        "--density_ref",
        type=Path,
        default=Path(__file__).resolve().parent / "density_ref_superhard.json",
    )
    parser.add_argument(
        "--mask_path",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "intermidiate"
        / "mask_superhard_materials.npz",
    )
    parser.add_argument("--allowed_elements", type=str, default="B,C,N,O,Si,H,Al")
    parser.add_argument("--nsites_min", type=int, default=2)
    parser.add_argument("--nsites_max", type=int, default=30)
    parser.add_argument("--max_ox_states", type=int, default=10)
    parser.add_argument("--max_density_trials", type=int, default=30)
    parser.add_argument("--max_shape_trials", type=int, default=50)
    parser.add_argument("--api_key_env", type=str, default="MP_API_KEY")
    parser.add_argument("--profile", action="store_true", help="Enable profiling logs.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logs.")
    parser.add_argument(
        "--mp_pool_size",
        type=int,
        default=0,
        help="Pool size for MP bulk fetch (default: 4*n).",
    )
    parser.add_argument(
        "--mp_chunk_size",
        type=int,
        default=0,
        help="Chunk size for MP summary search (0 disables).",
    )
    parser.add_argument(
        "--export_vasp",
        dest="export_vasp",
        action="store_true",
        default=True,
        help="Write .vasp files (default: true).",
    )
    parser.add_argument(
        "--no_export_vasp",
        dest="export_vasp",
        action="store_false",
        help="Disable .vasp output.",
    )
    parser.add_argument(
        "--pkl_path",
        type=Path,
        default=None,
        help="Output pickle path (default: data/raw_data/initial_superhard.pkl).",
    )
    parser.add_argument(
        "--cache_path",
        type=Path,
        default=Path(__file__).resolve().parent / "mp_structures_cache.jsonl",
    )
    parser.add_argument(
        "--cache_only",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Use cached MP structures only (default: true).",
    )
    parser.add_argument(
        "--clean_out_dir",
        type=lambda x: x.lower() == "true",
        default=True,
        help="Remove out_dir before writing (default: true).",
    )
<<<<<<< ours
    parser.add_argument(
        "--timing_out",
        type=Path,
        default=None,
        help="Write timing report (json/csv) when set.",
    )
    parser.add_argument(
        "--ox_timeout_sec",
        type=float,
        default=60.0,
        help="Timeout for oxidation-state search in seconds.",
    )
    parser.add_argument(
        "--print_debug",
        action="store_true",
        help="Print per-structure timing lines.",
    )
=======
>>>>>>> theirs
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    allowed_elements = {e.strip() for e in args.allowed_elements.split(",") if e.strip()}
    if not allowed_elements:
        allowed_elements = None

    default_pkl = Path(__file__).resolve().parents[2] / "data" / "raw_data" / "initial_superhard.pkl"
    pkl_path = args.pkl_path or default_pkl

    timing_out = args.timing_out or os.getenv("TIMING_OUT")

    generate_initial_structures(
        n=args.n,
        seed=args.seed,
        out_dir=args.out_dir,
        density_ref_path=args.density_ref,
        mask_path=args.mask_path,
        api_key=api_key,
        allowed_elements=allowed_elements,
        nsites_range=(args.nsites_min, args.nsites_max),
        max_ox_states=args.max_ox_states,
        max_density_trials=args.max_density_trials,
        max_shape_trials=args.max_shape_trials,
        cache_path=args.cache_path,
        cache_only=args.cache_only,
        clean_out_dir=args.clean_out_dir,
<<<<<<< ours
        mp_pool_size=args.mp_pool_size,
        mp_chunk_size=args.mp_chunk_size or None,
        profile=args.profile,
        verbose=args.verbose,
        export_vasp=args.export_vasp,
        pkl_path=pkl_path,
        timing_out=Path(timing_out) if timing_out else None,
        print_debug=args.print_debug,
        ox_timeout_sec=args.ox_timeout_sec,
=======
>>>>>>> theirs
    )


if __name__ == "__main__":
    main()
