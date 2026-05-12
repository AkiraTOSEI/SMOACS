#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.spg_wp.Amm2 import Amm2Wyckoff
from src.spg_wp.Fbar43m import Fbar43mWyckoff
from src.spg_wp.P2m import P2mWyckoff
from src.spg_wp.P321 import P321Wyckoff
from src.spg_wp.P4mmm import P4mmmWyckoff
from src.spg_wp.P6_3m import P63mWyckoff
from src.spg_wp.R3m import R3mWyckoff


@dataclass
class FailureRecord:
    space_group: str
    mode: str
    jid: str
    pattern_index: int
    total_charge: int | None
    site_ids: list[int]
    multiplicities: list[int]
    oxidation_pattern: list[int] | None
    reason: str


def _vector_key(vec: np.ndarray) -> bytes:
    arr = np.ascontiguousarray(np.asarray(vec, dtype=np.float32).reshape(-1))
    return arr.tobytes()


def build_mask_reverse_map(mask_path: Path) -> tuple[dict[bytes, int], int]:
    mask_dict = np.load(mask_path, allow_pickle=True)
    reverse: dict[bytes, int] = {}
    vec_len: int | None = None

    for ox_key in sorted(mask_dict.files, key=lambda x: int(x)):
        try:
            ox = int(ox_key)
        except ValueError as exc:
            raise ValueError(f"Invalid oxidation key in mask file: {ox_key!r}") from exc

        vec = np.asarray(mask_dict[ox_key], dtype=np.float32).reshape(-1)
        if vec_len is None:
            vec_len = int(vec.shape[0])
        elif vec.shape[0] != vec_len:
            raise ValueError(
                f"Mask vector length mismatch in {mask_path}: "
                f"expected {vec_len}, got {vec.shape[0]} for ox={ox}"
            )

        key = _vector_key(vec)
        if key in reverse and reverse[key] != ox:
            raise ValueError(
                "Ambiguous mask->oxidation mapping detected. "
                f"mask is shared by ox={reverse[key]} and ox={ox}."
            )
        reverse[key] = ox

    if vec_len is None:
        raise ValueError(f"Empty mask file: {mask_path}")
    return reverse, vec_len


def classify_mode_from_jid(jid: str) -> str:
    if jid.startswith("Sample-Rand__"):
        return "Random"
    if jid.startswith("Sample-MPPerturb-"):
        return "MPPerturb"
    if re.match(r"^Sample-\d+__", jid):
        return "MP"
    return "Unknown"


def parse_space_group_from_info_path(path: Path) -> str:
    m = re.match(r"^initial_MP_based_(.+)_info\.pkl$", path.name)
    if not m:
        raise ValueError(f"Cannot parse space group from info path: {path}")
    return m.group(1)


def get_num_equivalent_sites_for_space_group(space_group: str) -> list[int] | None:
    builders: dict[str, Any] = {
        "Amm2": Amm2Wyckoff.build,
        "P2m": P2mWyckoff.build,
        "Fbar43m": Fbar43mWyckoff.build,
        "R3m": R3mWyckoff.build,
        "P6_3m": P63mWyckoff.build,
        "P63m": P63mWyckoff.build,
        "P321": P321Wyckoff.build,
        "P4mmm": P4mmmWyckoff.build,
        "P4/mmm": P4mmmWyckoff.build,
    }
    build_fn = builders.get(space_group)
    if build_fn is None:
        return None
    return list(build_fn().num_equivalent_sites)


def _row_failure(
    *,
    space_group: str,
    mode: str,
    jid: str,
    site_ids: np.ndarray,
    multiplicities: np.ndarray,
    reason: str,
    pattern_index: int = -1,
    total_charge: int | None = None,
    oxidation_pattern: list[int] | None = None,
) -> FailureRecord:
    return FailureRecord(
        space_group=space_group,
        mode=mode,
        jid=jid,
        pattern_index=pattern_index,
        total_charge=total_charge,
        site_ids=[int(v) for v in site_ids.tolist()],
        multiplicities=[int(v) for v in multiplicities.tolist()],
        oxidation_pattern=oxidation_pattern,
        reason=reason,
    )


def validate_one_sample(
    *,
    row: pd.Series,
    space_group: str,
    num_equivalent_sites: list[int],
    mask_reverse: dict[bytes, int],
    mask_vec_len: int,
) -> tuple[int, list[FailureRecord]]:
    jid = str(row.get("jid", "UNKNOWN_JID"))
    mode = classify_mode_from_jid(jid)

    failures: list[FailureRecord] = []

    try:
        site_ids = np.asarray(row["site_ids"], dtype=int).reshape(-1)
    except Exception as exc:
        failures.append(
            _row_failure(
                space_group=space_group,
                mode=mode,
                jid=jid,
                site_ids=np.array([], dtype=int),
                multiplicities=np.array([], dtype=int),
                reason=f"site_ids parse error: {exc}",
            )
        )
        return 0, failures

    if site_ids.size == 0:
        failures.append(
            _row_failure(
                space_group=space_group,
                mode=mode,
                jid=jid,
                site_ids=site_ids,
                multiplicities=np.array([], dtype=int),
                reason="empty site_ids",
            )
        )
        return 0, failures

    try:
        multiplicities = np.asarray(
            [num_equivalent_sites[int(sid)] for sid in site_ids], dtype=int
        )
    except Exception as exc:
        failures.append(
            _row_failure(
                space_group=space_group,
                mode=mode,
                jid=jid,
                site_ids=site_ids,
                multiplicities=np.array([], dtype=int),
                reason=f"failed to map site_ids -> multiplicities: {exc}",
            )
        )
        return 0, failures

    try:
        ox_used_mask = np.asarray(row["ox_states_used_mask"], dtype=float).reshape(-1)
    except Exception as exc:
        failures.append(
            _row_failure(
                space_group=space_group,
                mode=mode,
                jid=jid,
                site_ids=site_ids,
                multiplicities=multiplicities,
                reason=f"ox_states_used_mask parse error: {exc}",
            )
        )
        return 0, failures

    try:
        atom_mask = np.asarray(row["atom_mask_for_all_ox_states"], dtype=np.float32)
    except Exception as exc:
        failures.append(
            _row_failure(
                space_group=space_group,
                mode=mode,
                jid=jid,
                site_ids=site_ids,
                multiplicities=multiplicities,
                reason=f"atom_mask_for_all_ox_states parse error: {exc}",
            )
        )
        return 0, failures

    if atom_mask.ndim != 3:
        failures.append(
            _row_failure(
                space_group=space_group,
                mode=mode,
                jid=jid,
                site_ids=site_ids,
                multiplicities=multiplicities,
                reason=f"atom_mask_for_all_ox_states ndim must be 3, got {atom_mask.ndim}",
            )
        )
        return 0, failures

    if atom_mask.shape[0] != site_ids.size:
        failures.append(
            _row_failure(
                space_group=space_group,
                mode=mode,
                jid=jid,
                site_ids=site_ids,
                multiplicities=multiplicities,
                reason=(
                    "N_wp mismatch: atom_mask.shape[0] != len(site_ids), "
                    f"{atom_mask.shape[0]} != {site_ids.size}"
                ),
            )
        )
        return 0, failures

    if atom_mask.shape[1] != mask_vec_len:
        failures.append(
            _row_failure(
                space_group=space_group,
                mode=mode,
                jid=jid,
                site_ids=site_ids,
                multiplicities=multiplicities,
                reason=(
                    "Mask vector length mismatch: "
                    f"expected {mask_vec_len}, got {atom_mask.shape[1]}"
                ),
            )
        )
        return 0, failures

    used_idx = np.where(ox_used_mask > 0.5)[0]
    if used_idx.size == 0:
        failures.append(
            _row_failure(
                space_group=space_group,
                mode=mode,
                jid=jid,
                site_ids=site_ids,
                multiplicities=multiplicities,
                reason="No used oxidation patterns (ox_states_used_mask has no ones).",
            )
        )
        return 0, failures

    pattern_count = 0
    for p in used_idx:
        pattern_count += 1

        if int(p) >= atom_mask.shape[2]:
            failures.append(
                _row_failure(
                    space_group=space_group,
                    mode=mode,
                    jid=jid,
                    site_ids=site_ids,
                    multiplicities=multiplicities,
                    reason=(
                        f"Pattern index {int(p)} out of bounds for "
                        f"atom_mask depth={atom_mask.shape[2]}"
                    ),
                    pattern_index=int(p),
                )
            )
            continue

        decoded: list[int] = []
        decode_ok = True
        for i in range(site_ids.size):
            vec = atom_mask[i, :, int(p)]
            ox = mask_reverse.get(_vector_key(vec))
            if ox is None:
                failures.append(
                    _row_failure(
                        space_group=space_group,
                        mode=mode,
                        jid=jid,
                        site_ids=site_ids,
                        multiplicities=multiplicities,
                        reason=f"Failed to decode mask vector at wp_index={i}, pattern={int(p)}",
                        pattern_index=int(p),
                    )
                )
                decode_ok = False
                break
            decoded.append(int(ox))

        if not decode_ok:
            continue

        total_charge = int(
            np.dot(multiplicities.astype(np.int64), np.asarray(decoded, dtype=np.int64))
        )
        if total_charge != 0:
            failures.append(
                _row_failure(
                    space_group=space_group,
                    mode=mode,
                    jid=jid,
                    site_ids=site_ids,
                    multiplicities=multiplicities,
                    reason="Charge neutrality violated.",
                    pattern_index=int(p),
                    total_charge=total_charge,
                    oxidation_pattern=decoded,
                )
            )

    return pattern_count, failures


def run_self_test() -> bool:
    vec_len = 98
    mask_for_2 = np.zeros(vec_len, dtype=np.float32)
    mask_for_m1 = np.zeros(vec_len, dtype=np.float32)
    mask_for_m2 = np.zeros(vec_len, dtype=np.float32)
    mask_for_2[1] = 1.0
    mask_for_m1[2] = 1.0
    mask_for_m2[3] = 1.0

    reverse = {
        _vector_key(mask_for_2): 2,
        _vector_key(mask_for_m1): -1,
        _vector_key(mask_for_m2): -2,
    }

    atom_mask = np.zeros((2, vec_len, 3), dtype=np.float32)
    atom_mask[0, :, 0] = mask_for_2
    atom_mask[1, :, 0] = mask_for_m1
    atom_mask[0, :, 1] = mask_for_2
    atom_mask[1, :, 1] = mask_for_m2

    row = pd.Series(
        {
            "jid": "Sample-0__self-test.vasp",
            "site_ids": np.array([0, 1], dtype=int),
            "ox_states_used_mask": np.array([1.0, 1.0, 0.0], dtype=float),
            "atom_mask_for_all_ox_states": atom_mask,
        }
    )

    pattern_count, failures = validate_one_sample(
        row=row,
        space_group="Amm2",
        num_equivalent_sites=[1, 2],
        mask_reverse=reverse,
        mask_vec_len=vec_len,
    )

    expected_patterns = 2
    expected_failures = 1
    ok = (pattern_count == expected_patterns) and (len(failures) == expected_failures)

    print("=== self_test ===")
    print(f"pattern_count={pattern_count} (expected {expected_patterns})")
    print(f"failure_count={len(failures)} (expected {expected_failures})")
    if failures:
        f = failures[0]
        print(
            "first_failure:",
            f"jid={f.jid}",
            f"pattern={f.pattern_index}",
            f"total_charge={f.total_charge}",
            f"reason={f.reason}",
        )
    print(f"self_test_result={'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check charge neutrality from initial_MP_based_*_info.pkl."
    )
    parser.add_argument(
        "--info_glob",
        type=str,
        default="data/raw_data/initial_MP_based_*_info.pkl",
        help="Glob pattern for info pickle files.",
    )
    parser.add_argument(
        "--mask_path",
        type=Path,
        default=Path("data/intermidiate/mask_dict_super_common.npz"),
        help="Path to oxidation mask dictionary .npz file.",
    )
    parser.add_argument(
        "--max_fail_print",
        type=int,
        default=20,
        help="Max number of failure records to print.",
    )
    parser.add_argument(
        "--self_test",
        action="store_true",
        help="Run built-in synthetic test and continue to main checks.",
    )
    parser.add_argument(
        "--self_test_only",
        action="store_true",
        help="Run only built-in synthetic test and exit.",
    )
    parser.add_argument(
        "--fail_on_unknown_space_group",
        action="store_true",
        help="Fail when an unknown initial_MP_based_<spg>_info.pkl key is found.",
    )
    args = parser.parse_args()

    if args.self_test:
        ok = run_self_test()
        if not ok:
            return 1
    if args.self_test_only:
        if not args.self_test:
            print("[ERROR] --self_test_only requires --self_test")
            return 2
        return 0

    try:
        mask_reverse, mask_vec_len = build_mask_reverse_map(args.mask_path)
    except Exception as exc:
        print(f"[ERROR] Failed to build mask reverse map: {exc}")
        return 2

    info_paths = sorted(Path(p) for p in glob.glob(args.info_glob))
    if not info_paths:
        print(f"[ERROR] No info pkl matched: {args.info_glob}")
        return 2

    summary: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "structures": 0,
            "patterns": 0,
            "pattern_fail": 0,
            "failed_jids": set(),
        }
    )
    failures: list[FailureRecord] = []
    skipped_paths: list[Path] = []

    required_cols = {
        "jid",
        "site_ids",
        "ox_states_used_mask",
        "atom_mask_for_all_ox_states",
    }

    for info_path in info_paths:
        try:
            space_group = parse_space_group_from_info_path(info_path)
        except ValueError as exc:
            print(f"[WARN] {exc}")
            skipped_paths.append(info_path)
            continue

        num_equivalent_sites = get_num_equivalent_sites_for_space_group(space_group)
        if num_equivalent_sites is None:
            msg = f"[WARN] Unsupported space-group key in file name: {space_group} ({info_path})"
            if args.fail_on_unknown_space_group:
                print(msg)
                return 2
            print(msg)
            skipped_paths.append(info_path)
            continue

        try:
            df = pd.read_pickle(info_path)
        except Exception as exc:
            print(f"[ERROR] Failed to read pickle {info_path}: {exc}")
            return 2

        missing = required_cols.difference(df.columns)
        if missing:
            print(f"[ERROR] Missing required columns in {info_path}: {sorted(missing)}")
            return 2

        for _, row in df.iterrows():
            jid = str(row["jid"])
            mode = classify_mode_from_jid(jid)
            key = (space_group, mode)
            summary[key]["structures"] += 1

            pattern_count, row_failures = validate_one_sample(
                row=row,
                space_group=space_group,
                num_equivalent_sites=num_equivalent_sites,
                mask_reverse=mask_reverse,
                mask_vec_len=mask_vec_len,
            )
            summary[key]["patterns"] += pattern_count
            summary[key]["pattern_fail"] += len(row_failures)
            if row_failures:
                summary[key]["failed_jids"].add(jid)
                failures.extend(row_failures)

    print("=== Neutrality Check Summary ===")
    print("space_group\tmode\tstructures\tpatterns\tpattern_fail\tstructure_fail")
    for space_group, mode in sorted(summary):
        s = summary[(space_group, mode)]
        print(
            f"{space_group}\t{mode}\t{s['structures']}\t{s['patterns']}\t"
            f"{s['pattern_fail']}\t{len(s['failed_jids'])}"
        )

    if skipped_paths:
        print(f"Skipped files: {len(skipped_paths)}")
        for p in skipped_paths[:10]:
            print(f"  - {p}")
        if len(skipped_paths) > 10:
            print(f"  ... and {len(skipped_paths) - 10} more")

    print(f"Total failures: {len(failures)}")
    if failures:
        print("=== Failure Details (truncated) ===")
        for rec in failures[: args.max_fail_print]:
            print(
                f"space_group={rec.space_group} mode={rec.mode} jid={rec.jid} "
                f"pattern={rec.pattern_index} total_charge={rec.total_charge} reason={rec.reason} "
                f"site_ids={rec.site_ids} multiplicities={rec.multiplicities} "
                f"oxidation_pattern={rec.oxidation_pattern}"
            )
        if len(failures) > args.max_fail_print:
            print(f"... {len(failures) - args.max_fail_print} more failures")
        return 1

    print("All checked oxidation patterns satisfy charge neutrality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
