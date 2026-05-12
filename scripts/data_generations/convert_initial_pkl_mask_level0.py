#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import pickle
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def _vector_key(vec: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(np.asarray(vec, dtype=np.float32).reshape(-1))
    return arr.tobytes()


def _build_mask_reverse_map(
    mask_d: Mapping[str, np.ndarray]
) -> tuple[dict[bytes, int], int]:
    reverse: dict[bytes, int] = {}
    vec_len: int | None = None
    for ox_key in sorted(mask_d.keys(), key=lambda x: int(x)):
        try:
            ox = int(ox_key)
        except ValueError as exc:
            raise ValueError(f"Invalid oxidation key: {ox_key!r}") from exc

        vec = np.asarray(mask_d[ox_key], dtype=np.float32).reshape(-1)
        if vec_len is None:
            vec_len = int(vec.shape[0])
        elif int(vec.shape[0]) != vec_len:
            raise ValueError(
                f"Mask vector length mismatch: expected {vec_len}, got {vec.shape[0]} for ox={ox}"
            )

        key = _vector_key(vec)
        if key in reverse and reverse[key] != ox:
            raise ValueError(
                "Ambiguous mask->oxidation mapping detected. "
                f"mask is shared by ox={reverse[key]} and ox={ox}."
            )
        reverse[key] = ox

    if vec_len is None:
        raise ValueError("Empty mask dictionary.")
    return reverse, vec_len


def _validated_dst_mask(
    mask_d: Mapping[str, np.ndarray], vec_len: int
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for key, value in mask_d.items():
        vec = np.asarray(value, dtype=np.float32).reshape(-1)
        if int(vec.shape[0]) != vec_len:
            raise ValueError(
                f"Destination mask vector length mismatch for ox={key}: "
                f"expected {vec_len}, got {vec.shape[0]}"
            )
        out[key] = vec
    return out


def _remap_atom_mask(
    atom_mask: np.ndarray,
    ox_states_used_mask: np.ndarray,
    mask_reverse: Mapping[bytes, int],
    mask_vec_len: int,
    dst_mask_d: Mapping[str, np.ndarray],
) -> np.ndarray:
    atom_mask = np.asarray(atom_mask, dtype=np.float32)
    ox_states_used_mask = np.asarray(ox_states_used_mask, dtype=np.float32).reshape(-1)

    if atom_mask.ndim != 3:
        raise ValueError(
            f"atom_mask_for_all_ox_states ndim must be 3, got {atom_mask.ndim}"
        )
    if atom_mask.shape[1] != mask_vec_len:
        raise ValueError(
            "atom_mask_for_all_ox_states second dimension mismatch: "
            f"expected {mask_vec_len}, got {atom_mask.shape[1]}"
        )
    if atom_mask.shape[2] != int(ox_states_used_mask.shape[0]):
        raise ValueError(
            "ox_states_used_mask length mismatch: "
            f"atom_mask.shape[2]={atom_mask.shape[2]}, ox_states_used_mask={ox_states_used_mask.shape[0]}"
        )
    if not np.all(np.isin(ox_states_used_mask, (0.0, 1.0))):
        raise ValueError(
            "ox_states_used_mask must be binary (0 or 1). "
            f"Got values: {np.unique(ox_states_used_mask).tolist()}"
        )

    remapped = np.array(atom_mask, copy=True, dtype=np.float32)
    n_sites, _, n_patterns = atom_mask.shape
    for pattern_idx in range(n_patterns):
        if float(ox_states_used_mask[pattern_idx]) == 0.0:
            continue
        for site_idx in range(n_sites):
            key = _vector_key(atom_mask[site_idx, :, pattern_idx])
            ox = mask_reverse.get(key)
            if ox is None:
                raise ValueError(
                    f"Cannot recover oxidation state from source mask at site={site_idx}, "
                    f"pattern={pattern_idx}."
                )
            dst_vec = dst_mask_d.get(str(ox))
            if dst_vec is None:
                raise ValueError(
                    f"Oxidation state {ox} is missing in destination mask."
                )
            remapped[site_idx, :, pattern_idx] = dst_vec

    return remapped


def convert_records_level0(
    records: list[dict[str, Any]],
    src_mask_d: Mapping[str, np.ndarray],
    dst_mask_d: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    mask_reverse, mask_vec_len = _build_mask_reverse_map(src_mask_d)
    dst_mask_vec = _validated_dst_mask(dst_mask_d, mask_vec_len)

    converted: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        if "atom_mask_for_all_ox_states" not in record:
            raise KeyError(f"Record {idx} missing key: atom_mask_for_all_ox_states")
        if "ox_states_used_mask" not in record:
            raise KeyError(f"Record {idx} missing key: ox_states_used_mask")
        new_record = copy.deepcopy(record)
        new_record["atom_mask_for_all_ox_states"] = _remap_atom_mask(
            atom_mask=record["atom_mask_for_all_ox_states"],
            ox_states_used_mask=record["ox_states_used_mask"],
            mask_reverse=mask_reverse,
            mask_vec_len=mask_vec_len,
            dst_mask_d=dst_mask_vec,
        )
        converted.append(new_record)
    return converted


def _validate_invariants(
    src_records: list[dict[str, Any]],
    dst_records: list[dict[str, Any]],
) -> None:
    if len(src_records) != len(dst_records):
        raise ValueError(
            f"Record count mismatch after conversion: src={len(src_records)} dst={len(dst_records)}"
        )
    for idx, (src, dst) in enumerate(zip(src_records, dst_records)):
        for key in (
            "jid",
            "material_id",
            "sample_id",
            "nsites",
            "elements",
            "coords_frac",
            "lattice_matrix",
        ):
            if key not in src:
                raise KeyError(f"Source record {idx} missing required key: {key}")
            if key not in dst:
                raise KeyError(f"Destination record {idx} missing required key: {key}")
        for key in ("jid", "material_id", "sample_id", "nsites", "elements"):
            if src[key] != dst[key]:
                raise ValueError(f"Invariant mismatch at record={idx}, key={key!r}")
        for key in ("coords_frac", "lattice_matrix"):
            src_arr = np.asarray(src[key])
            dst_arr = np.asarray(dst[key])
            if src_arr.shape != dst_arr.shape or not np.array_equal(src_arr, dst_arr):
                raise ValueError(f"Invariant mismatch at record={idx}, key={key!r}")


def _load_pkl(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as fh:
        data = pickle.load(fh)
    if not isinstance(data, list):
        raise TypeError(f"Expected list in pickle: {path}")
    return data


def _save_pkl(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(data, fh, protocol=4)


def _load_mask(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create level0 initial MP pickle from a level-specific pickle by replacing "
            "atom_mask_for_all_ox_states with mask_dict_super_common.npz."
        )
    )
    parser.add_argument("--src_pkl", type=Path, required=True)
    parser.add_argument("--src_mask_path", type=Path, required=True)
    parser.add_argument(
        "--dst_mask_path",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data"
        / "intermidiate"
        / "mask_dict_super_common.npz",
    )
    parser.add_argument("--out_pkl", type=Path, required=True)
    args = parser.parse_args()

    src_records = _load_pkl(args.src_pkl)
    src_mask_d = _load_mask(args.src_mask_path)
    dst_mask_d = _load_mask(args.dst_mask_path)

    converted = convert_records_level0(
        records=src_records,
        src_mask_d=src_mask_d,
        dst_mask_d=dst_mask_d,
    )
    _validate_invariants(src_records, converted)
    _save_pkl(args.out_pkl, converted)
    print(f"Converted records: {len(converted)}")
    print(f"Source pkl: {args.src_pkl}")
    print(f"Source mask: {args.src_mask_path}")
    print(f"Destination mask: {args.dst_mask_path}")
    print(f"Output pkl: {args.out_pkl}")


if __name__ == "__main__":
    main()
