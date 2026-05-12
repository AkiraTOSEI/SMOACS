#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _load_banned(path: Path) -> dict[int, list[int]]:
    data = json.loads(path.read_text())
    out = {}
    for level, nums in data.items():
        out[int(level)] = [int(n) for n in nums]
    return out


def _filter_mask(mask: np.ndarray, banned: set[int]) -> np.ndarray:
    filtered = mask.copy()
    for z in banned:
        idx = z - 1
        if 0 <= idx < filtered.shape[0]:
            filtered[idx] = 0.0
    return filtered


def generate_level_masks(
    banned_path: Path,
    in_path: Path,
    out_dir: Path,
) -> list[Path]:
    banned_by_level = _load_banned(banned_path)
    mask_npz = np.load(in_path, allow_pickle=True)
    out_paths: list[Path] = []

    out_dir.mkdir(parents=True, exist_ok=True)
    for level in sorted(banned_by_level):
        banned = set(banned_by_level[level])
        out_dict = {}
        for key in mask_npz.files:
            mask = np.array(mask_npz[key], dtype=float)
            if mask.shape != (98,):
                raise ValueError(f"Unexpected mask shape for key {key}: {mask.shape}")
            out_dict[key] = _filter_mask(mask, banned)
        out_path = out_dir / f"mask_dict_super_common_level{level}.npz"
        np.savez(out_path, **out_dict)
        meta = {
            "level": level,
            "banned_atomic_numbers": sorted(banned),
            "source_mask": str(in_path),
            "keys": sorted(out_dict.keys()),
        }
        out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2))
        out_paths.append(out_path)
    return out_paths


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate level-specific mask npz files.")
    parser.add_argument(
        "--banned_path",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "workspace"
        / "sonata_ws"
        / "banned_atomic_numbers_by_level.json",
    )
    parser.add_argument(
        "--in_path",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "intermidiate"
        / "mask_dict_super_common.npz",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "intermidiate",
    )
    args = parser.parse_args()

    out_paths = generate_level_masks(args.banned_path, args.in_path, args.out_dir)
    for path in out_paths:
        print(f"Wrote mask: {path}")


if __name__ == "__main__":
    main()
