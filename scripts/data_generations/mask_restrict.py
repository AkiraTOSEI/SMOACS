#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from periodictable import elements as pd_elements


ALLOWED_ELEMENTS = {"B", "C", "N", "O", "Si", "H", "Al"}


def _allowed_indices(allowed_elements: set[str]) -> set[int]:
    return {pd_elements.symbol(sym).number - 1 for sym in allowed_elements}


def build_mask_superhard(
    in_path: Path,
    out_path: Path,
    allowed_elements: set[str] | None = None,
) -> dict:
    if allowed_elements is None:
        allowed_elements = set(ALLOWED_ELEMENTS)
    allowed_idx = _allowed_indices(allowed_elements)

    mask_npz = np.load(in_path, allow_pickle=True)
    out_dict = {}
    for key in mask_npz.files:
        mask = np.array(mask_npz[key], dtype=float)
        if mask.shape != (98,):
            raise ValueError(f"Unexpected mask shape for key {key}: {mask.shape}")
        filtered = np.zeros_like(mask)
        for i in allowed_idx:
            filtered[i] = mask[i]
        out_dict[key] = filtered

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **out_dict)

    meta = {
        "allowed_elements": sorted(allowed_elements),
        "source_mask": str(in_path),
        "keys": sorted(out_dict.keys()),
    }
    meta_path = out_path.with_suffix(".json")
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def build_atomic_mask_for_oxidation_patterns_A(
    ox_mask_path: Path,
    oxidation_patterns: Sequence[Sequence[int]],
    num_atom_at_WP: Sequence[int],
    max_ox_patterns: int,
) -> np.ndarray:
    try:
        from wyckoff_encoding import build_atomic_mask_for_oxidation_patterns
    except ImportError as exc:
        raise RuntimeError(
            "wyckoff_encoding.py is required for Wyckoff-pattern mask expansion. "
            "It is not part of the current minimal public migration."
        ) from exc

    mask_dict = np.load(ox_mask_path, allow_pickle=True)
    allowed_oxi = set(mask_dict.files)
    filtered = [
        pattern
        for pattern in oxidation_patterns
        if all(str(ox) in allowed_oxi for ox in pattern)
    ]
    return build_atomic_mask_for_oxidation_patterns(
        ox_mask_path=ox_mask_path,
        oxidation_table=filtered,
        num_atom_at_WP=num_atom_at_WP,
        max_ox_patterns=max_ox_patterns,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Restrict oxidation mask to allowed elements.")
    parser.add_argument(
        "--in_path",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "intermidiate"
        / "mask_dict_super_common.npz",
    )
    parser.add_argument(
        "--out_path",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "intermidiate"
        / "mask_superhard_materials.npz",
    )
    parser.add_argument("--allowed_elements", type=str, default="B,C,N,O,Si,H,Al")
    args = parser.parse_args()

    allowed_elements = {e.strip() for e in args.allowed_elements.split(",") if e.strip()}
    meta = build_mask_superhard(args.in_path, args.out_path, allowed_elements)
    print("Wrote mask:", args.out_path)
    print("Meta:", meta)


if __name__ == "__main__":
    main()
