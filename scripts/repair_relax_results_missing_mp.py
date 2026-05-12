#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

from pymatgen.core.composition import Composition

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from calculate_mattersim_convex_hull_energy import (
    DEFAULT_RELAX_DB_PATH,
    _elements_from_chemsys,
    _ensure_record_chemsys,
    _load_json_db,
    _load_json_gz,
    _update_subchemsys_files,
    _write_json_gz,
    attach_ehull,
    collect_mp_alex_entries,
    fetch_mp_structures,
)


def _parse_chemsys_from_filename(path: Path) -> str:
    name = path.name
    prefix = "relax_results_"
    suffix = ".json.gz"
    if not (name.startswith(prefix) and name.endswith(suffix)):
        raise ValueError(f"Unexpected relax_results filename: {name}")
    return name[len(prefix) : -len(suffix)]


def _count_prefix(records: list[dict], prefix: str) -> int:
    return sum(
        1
        for rec in records
        if str(rec.get("material_id", "")).startswith(prefix)
    )


def _terminal_elements_from_records(records: list[dict]) -> set[str]:
    terminal_elements: set[str] = set()
    for rec in records:
        if rec.get("status") == "failed":
            continue
        formula = rec.get("formula")
        if not formula:
            continue
        try:
            comp = Composition(formula)
        except Exception:
            continue
        elems = {str(el) for el in comp.elements}
        if len(elems) == 1:
            terminal_elements.update(elems)
    return terminal_elements


def _check_record_chemsys(
    records: list[dict],
    target_elems: set[str],
) -> tuple[int, int]:
    missing_chemsys = 0
    mismatched = 0
    for rec in records:
        chem_sys = rec.get("chem_sys")
        if not chem_sys:
            chem_sys = _ensure_record_chemsys(rec)
        if not chem_sys:
            missing_chemsys += 1
            continue
        elems = set(_elements_from_chemsys(chem_sys))
        if elems and not elems.issubset(target_elems):
            mismatched += 1
    return missing_chemsys, mismatched


def _move_to_backup(path: Path, backup_dir: Path) -> Path:
    backup_dir.mkdir(parents=True, exist_ok=True)
    dest = backup_dir / path.name
    if dest.exists():
        ts = time.strftime("%Y%m%d-%H%M%S")
        dest = backup_dir / f"{path.name}.{ts}.bak"
    shutil.move(str(path), str(dest))
    return dest


def _merge_records(mp_records: list[dict], existing: list[dict]) -> list[dict]:
    combined = {rec.get("material_id"): rec for rec in mp_records}
    for rec in existing:
        combined[rec.get("material_id")] = rec
    return list(combined.values())


def _log(message: str, log_path: Path | None) -> None:
    print(message)
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(f"[{ts}] {message}\n")


def _expected_mp_ids(api_key: str, chemsys: str, max_n: int | None) -> list[str]:
    elems = _elements_from_chemsys(chemsys)
    data = fetch_mp_structures(api_key, elems, max_n)
    return [mid for mid, _ in data]


def _missing_ids_from_cache(
    expected_ids: list[str],
    relax_db_path: Path,
    limit_ids: set[str] | None = None,
) -> list[str]:
    cache = _load_json_db(relax_db_path)
    missing = []
    for mid in expected_ids:
        if limit_ids is not None and mid not in limit_ids:
            continue
        if mid not in cache:
            missing.append(mid)
    return missing


def _read_failed_ids(path: Path) -> dict[str, set[str]]:
    if not path.exists():
        return {}
    data: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            chemsys = rec.get("chemsys")
            mid = rec.get("material_id")
            if not chemsys or not mid:
                continue
            data.setdefault(chemsys, set()).add(mid)
    return data


def _append_failed_ids(path: Path, chemsys: str, ids: list[str]) -> None:
    if not ids:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as fp:
        for mid in ids:
            rec = {"ts": ts, "chemsys": chemsys, "material_id": mid}
            fp.write(json.dumps(rec, ensure_ascii=True) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Detect relax_results_<chemsys>.json.gz files that lack MP entries "
            "and optionally re-fetch and merge MP entries."
        )
    )
    parser.add_argument(
        "--relax_results_dir",
        type=Path,
        default=DEFAULT_RELAX_DB_PATH.parent,
        help="Directory that contains relax_results_<chemsys>.json.gz",
    )
    parser.add_argument(
        "--relax_db_path",
        type=Path,
        default=DEFAULT_RELAX_DB_PATH,
        help="Path to relax_results.json (MP cache DB).",
    )
    parser.add_argument(
        "--mp_api_key",
        type=str,
        default=None,
        help="Materials Project API key (required with --add).",
    )
    parser.add_argument(
        "--steps_max",
        type=int,
        default=400,
        help="Max steps for MatterSim relaxation when adding MP entries.",
    )
    parser.add_argument(
        "--max_n",
        type=int,
        default=None,
        help="Optional cap on number of MP structures to fetch.",
    )
    parser.add_argument(
        "--backup_dir",
        type=Path,
        default=None,
        help="When set, move existing file here before writing.",
    )
    parser.add_argument(
        "--add",
        action="store_true",
        help="Actually fetch and append MP entries (default: dry-run).",
    )
    parser.add_argument(
        "--chemsys",
        type=str,
        default=None,
        help="Comma-separated list of chemsys names to limit processing.",
    )
    parser.add_argument(
        "--log_path",
        type=Path,
        default=None,
        help="Append progress log to this path (default: relax_results_dir/repair_relax_results_missing_mp.log).",
    )
    parser.add_argument(
        "--retry_failed",
        type=int,
        default=0,
        help="Retry missing MP entries up to N times after initial attempt.",
    )
    parser.add_argument(
        "--retry_delay_s",
        type=float,
        default=30.0,
        help="Sleep this many seconds between retry attempts.",
    )
    parser.add_argument(
        "--failed_ids_path",
        type=Path,
        default=None,
        help="Write failed MP IDs as JSONL to this path.",
    )
    parser.add_argument(
        "--retry_failed_from",
        type=Path,
        default=None,
        help="Retry only IDs listed in this JSONL file (chemsys scoped).",
    )
    args = parser.parse_args()

    relax_dir = args.relax_results_dir
    if not relax_dir.exists():
        raise FileNotFoundError(f"relax_results_dir not found: {relax_dir}")

    log_path = args.log_path
    if log_path is None:
        log_path = relax_dir / "repair_relax_results_missing_mp.log"

    failed_ids_path = args.failed_ids_path
    if failed_ids_path is None:
        failed_ids_path = relax_dir / "repair_relax_results_missing_mp.failed.jsonl"

    filter_chemsys = None
    if args.chemsys:
        filter_chemsys = {cs.strip() for cs in args.chemsys.split(",") if cs.strip()}

    retry_failed_by_chemsys: dict[str, set[str]] = {}
    if args.retry_failed_from:
        retry_failed_by_chemsys = _read_failed_ids(args.retry_failed_from)
        if filter_chemsys is not None:
            retry_failed_by_chemsys = {
                cs: ids
                for cs, ids in retry_failed_by_chemsys.items()
                if cs in filter_chemsys
            }

    forced_chemsys = set(retry_failed_by_chemsys.keys())

    paths = sorted(relax_dir.glob("relax_results_*.json.gz"))
    if filter_chemsys:
        paths = [
            p for p in paths if _parse_chemsys_from_filename(p) in filter_chemsys
        ]

    targets: list[tuple[Path, list[dict], str, int, int, int, bool]] = []
    for path in paths:
        chemsys = _parse_chemsys_from_filename(path)
        records = _load_json_gz(path)
        mp_count = _count_prefix(records, "mp-")
        agm_count = _count_prefix(records, "agm-")
        other_count = len(records) - mp_count - agm_count
        forced = chemsys in forced_chemsys
        if mp_count == 0 or forced:
            targets.append(
                (path, records, chemsys, mp_count, agm_count, other_count, forced)
            )

    if not targets:
        _log("No relax_results files missing MP entries.", log_path)
        return 0

    _log(f"Targets: {len(targets)}", log_path)
    for path, records, chemsys, mp_count, agm_count, other_count, forced in targets:
        msg = f"{path.name} mp_count={mp_count} agm_count={agm_count}"
        if other_count:
            msg += f" other_count={other_count}"
        if forced:
            msg += " forced_retry"
        _log(msg, log_path)
        target_elems = set(_elements_from_chemsys(chemsys))
        missing_chemsys, mismatched = _check_record_chemsys(records, target_elems)
        if missing_chemsys:
            _log(f"  warn: missing chem_sys in {missing_chemsys} records", log_path)
        if mismatched:
            _log(f"  warn: {mismatched} records outside target chemsys", log_path)
        missing_terminals = target_elems - _terminal_elements_from_records(records)
        if missing_terminals:
            missing_str = "-".join(sorted(missing_terminals))
            _log(f"  warn: missing terminal entries for {missing_str}", log_path)

    if not args.add:
        _log("Dry-run only. Re-run with --add to fetch and merge MP entries.", log_path)
        return 0

    if not args.mp_api_key:
        raise ValueError("--mp_api_key is required with --add")

    for path, records, chemsys, mp_count, agm_count, _other_count, forced in targets:
        _log(f"Adding MP entries to {path.name} (chemsys={chemsys})", log_path)
        mp_records_latest = collect_mp_alex_entries(
            api_key=args.mp_api_key,
            chemsys=_elements_from_chemsys(chemsys),
            db_path=args.relax_db_path,
            max_n=args.max_n,
            steps_max=args.steps_max,
            use_alexandria=False,
        )
        combined = _merge_records(mp_records_latest, records)
        combined_with_ehull = attach_ehull(combined)

        if args.backup_dir:
            backup_path = _move_to_backup(path, args.backup_dir)
            _log(f"  moved backup to {backup_path}", log_path)

        _write_json_gz(path, combined_with_ehull)
        _update_subchemsys_files(combined_with_ehull, args.relax_db_path.parent)

        should_retry = args.retry_failed > 0 or (forced and retry_failed_by_chemsys)
        expected_ids: list[str] | None = None
        if should_retry:
            expected_ids = _expected_mp_ids(args.mp_api_key, chemsys, args.max_n)

        limit_ids = retry_failed_by_chemsys.get(chemsys)
        if args.retry_failed > 0 and expected_ids is not None:
            for attempt in range(1, args.retry_failed + 1):
                missing = _missing_ids_from_cache(
                    expected_ids,
                    args.relax_db_path,
                    limit_ids=limit_ids,
                )
                if not missing:
                    _log("  retry: no missing IDs remain", log_path)
                    break
                _log(
                    f"  retry attempt {attempt}/{args.retry_failed}: "
                    f"missing={len(missing)}",
                    log_path,
                )
                _log(
                    f"  waiting {args.retry_delay_s:.1f}s before retry",
                    log_path,
                )
                time.sleep(args.retry_delay_s)
                mp_records_latest = collect_mp_alex_entries(
                    api_key=args.mp_api_key,
                    chemsys=_elements_from_chemsys(chemsys),
                    db_path=args.relax_db_path,
                    max_n=args.max_n,
                    steps_max=args.steps_max,
                    use_alexandria=False,
                )

        if should_retry:
            combined = _merge_records(mp_records_latest, records)
            combined_with_ehull = attach_ehull(combined)
            _write_json_gz(path, combined_with_ehull)
            _update_subchemsys_files(combined_with_ehull, args.relax_db_path.parent)

        if expected_ids is None:
            expected_ids = _expected_mp_ids(args.mp_api_key, chemsys, args.max_n)

        missing_final = _missing_ids_from_cache(
            expected_ids,
            args.relax_db_path,
            limit_ids=limit_ids,
        )
        if missing_final:
            _log(
                f"  remaining missing IDs after retries: {len(missing_final)}",
                log_path,
            )
            _append_failed_ids(failed_ids_path, chemsys, missing_final)

        mp_after = _count_prefix(combined_with_ehull, "mp-")
        agm_after = _count_prefix(combined_with_ehull, "agm-")
        _log(f"  updated counts: mp_count={mp_after} agm_count={agm_after}", log_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
