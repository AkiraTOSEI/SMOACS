#!/usr/bin/env python3
from __future__ import annotations

"""
Generate initial structures from Materials Project and save npz/json outputs.

Example:
    python scripts/data_generations/generate_initials.py \\
        --n 20 \\
        --seed 0 \\
        --out_dir data/raw_data/initial_superhard \\
        --density_ref scripts/data_generations/density_ref_superhard.json \\
        --mask_path data/intermidiate/mask_superhard_materials.npz \\
        --api_key_env MP_API_KEY
"""

import json
from pathlib import Path

import numpy as np

from lattice_randomize import randomize_lattice_with_density
from mp_fetch import fetch_mp_structures
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


def generate_initial_structures(
    n: int,
    seed: int | None,
    out_dir: Path,
    density_ref_path: Path,
    mask_path: Path,
    api_key: str | None,
    allowed_elements: set[str],
    nsites_range: tuple[int, int],
    max_ox_states: int,
    max_density_trials: int,
    max_shape_trials: int,
) -> None:
    rng = np.random.default_rng(seed)
    ref = _load_density_ref(density_ref_path)
    rho_min = float(ref["rho_min"])
    rho_max = float(ref["rho_max"])

<<<<<<< ours
    docs = fetch_mp_structures(
        api_key=api_key,
        n=n,
=======
    fetch_n = max(n * 4, n)
    docs = fetch_mp_structures(
        api_key=api_key,
        n=fetch_n,
>>>>>>> theirs
        allowed_elements=allowed_elements,
        nsites_range=nsites_range,
    )
    if len(docs) < n:
        raise RuntimeError(f"Only {len(docs)} structures matched filters; need {n}.")

    out_dir.mkdir(parents=True, exist_ok=True)

<<<<<<< ours
    for i, doc in enumerate(docs):
        material_id = doc["material_id"]
        structure = doc["structure"]
        result = randomize_lattice_with_density(
            structure=structure,
            rho_min=rho_min,
            rho_max=rho_max,
            min_dist_threshold=1.0,
            rng=rng,
            max_density_trials=max_density_trials,
            max_shape_trials=max_shape_trials,
        )

        npz_data, json_meta = encode_structure_no_wyckoff(
            structure=result.structure,
            material_id=material_id,
            sample_id=i,
            ox_mask_path=mask_path,
            max_ox_states=max_ox_states,
        )
=======
    written = 0
    for i, doc in enumerate(docs):
        material_id = doc["material_id"]
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
        except Exception:
            continue

        try:
            npz_data, json_meta = encode_structure_no_wyckoff(
                structure=result.structure,
                material_id=material_id,
                sample_id=written,
                ox_mask_path=mask_path,
                max_ox_states=max_ox_states,
            )
        except Exception:
            continue
>>>>>>> theirs
        json_meta.update(
            {
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

<<<<<<< ours
        base = f"sample_{i:04d}__{material_id}"
        np.savez(out_dir / f"{base}.npz", **npz_data)
        (out_dir / f"{base}.json").write_text(json.dumps(json_meta, indent=2))
=======
        base = f"sample_{written:04d}__{material_id}"
        np.savez(out_dir / f"{base}.npz", **npz_data)
        (out_dir / f"{base}.json").write_text(json.dumps(json_meta, indent=2))
        written += 1
        if written >= n:
            break

    if written < n:
        raise RuntimeError(f"Only generated {written} structures; need {n}.")
>>>>>>> theirs


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser(description="Generate initial structures.")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out_dir", type=Path, required=True)
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
    parser.add_argument("--nsites_max", type=int, default=50)
    parser.add_argument("--max_ox_states", type=int, default=10)
    parser.add_argument("--max_density_trials", type=int, default=30)
    parser.add_argument("--max_shape_trials", type=int, default=50)
    parser.add_argument("--api_key_env", type=str, default="MP_API_KEY")
    args = parser.parse_args()

    api_key = os.getenv(args.api_key_env)
    allowed_elements = {e.strip() for e in args.allowed_elements.split(",") if e.strip()}

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
    )


if __name__ == "__main__":
    main()
