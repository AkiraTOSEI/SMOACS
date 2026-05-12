from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.common import create_element_list


SUPPORTED_ELEMENT_CONSTRAINT_KEYS = {"mask_dict_path", "forbidden_elements"}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_forbidden_elements(elements: object) -> list[str]:
    if not isinstance(elements, (list, tuple)):
        raise ValueError("element_constraints.forbidden_elements must be a list.")

    valid_elements = create_element_list()
    valid_set = set(valid_elements)
    normalized: list[str] = []
    seen: set[str] = set()
    for item in elements:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(
                "element_constraints.forbidden_elements must contain element symbols."
            )
        symbol = item.strip()
        if symbol not in valid_set:
            raise ValueError(
                "Unknown forbidden element symbol. "
                f"Expected one of the first 98 elements, got: {symbol}"
            )
        if symbol not in seen:
            normalized.append(symbol)
            seen.add(symbol)
    return normalized


def get_element_constraints(config: Mapping[str, Any] | None) -> dict[str, Any]:
    if config is None:
        return {"mask_dict_path": None, "forbidden_elements": []}
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping.")

    raw_constraints = config.get("element_constraints")
    if raw_constraints is None:
        return {"mask_dict_path": None, "forbidden_elements": []}
    if not isinstance(raw_constraints, Mapping):
        raise ValueError("element_constraints must be a mapping.")

    unknown_keys = set(raw_constraints) - SUPPORTED_ELEMENT_CONSTRAINT_KEYS
    if unknown_keys:
        raise ValueError(
            "Unsupported element_constraints keys: "
            f"{sorted(unknown_keys)}. Supported keys are "
            f"{sorted(SUPPORTED_ELEMENT_CONSTRAINT_KEYS)}."
        )

    mask_dict_path = raw_constraints.get("mask_dict_path")
    if mask_dict_path is not None and (
        not isinstance(mask_dict_path, str) or not mask_dict_path.strip()
    ):
        raise ValueError("element_constraints.mask_dict_path must be a path string.")

    if "forbidden_elements" in raw_constraints:
        forbidden_elements = validate_forbidden_elements(
            raw_constraints["forbidden_elements"]
        )
    else:
        forbidden_elements = []

    return {
        "mask_dict_path": mask_dict_path.strip() if mask_dict_path else None,
        "forbidden_elements": forbidden_elements,
    }


def select_mask_dict_path(
    cli_mask_dict_path: str | None,
    element_constraints: Mapping[str, Any],
) -> str | None:
    mask_dict_path = cli_mask_dict_path or element_constraints.get("mask_dict_path")
    forbidden_elements = element_constraints.get("forbidden_elements", [])
    if forbidden_elements and mask_dict_path is None:
        raise ValueError(
            "element_constraints.forbidden_elements requires --mask_dict_path "
            "or element_constraints.mask_dict_path."
        )
    return mask_dict_path


def write_forbidden_element_mask(
    base_mask_path: Path,
    forbidden_elements: Sequence[str],
    out_dir: Path,
) -> Path:
    forbidden = validate_forbidden_elements(list(forbidden_elements))
    if not forbidden:
        return base_mask_path

    base_mask_path = base_mask_path.expanduser().resolve()
    if not base_mask_path.is_file() or base_mask_path.suffix != ".npz":
        raise ValueError(f"Base mask must be a .npz file: {base_mask_path}")

    source_sha256 = file_sha256(base_mask_path)
    element_to_index = {symbol: idx for idx, symbol in enumerate(create_element_list())}
    forbidden_indices = [element_to_index[symbol] for symbol in forbidden]

    payload = {
        "source_mask_path": str(base_mask_path),
        "source_mask_sha256": source_sha256,
        "forbidden_elements": sorted(forbidden),
    }
    payload_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"mask_forbid_{payload_hash}.npz"
    meta_path = out_path.with_suffix(".json")

    out_dict: dict[str, np.ndarray] = {}
    with np.load(base_mask_path, allow_pickle=True) as mask_npz:
        for key in mask_npz.files:
            mask = np.array(mask_npz[key], copy=True)
            if mask.shape != (98,):
                raise ValueError(f"Unexpected mask shape for key {key}: {mask.shape}")
            mask[forbidden_indices] = 0
            out_dict[key] = mask

    np.savez(out_path, **out_dict)
    generated_sha256 = file_sha256(out_path)

    meta = {
        **payload,
        "generated_mask_path": str(out_path),
        "generated_mask_sha256": generated_sha256,
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
