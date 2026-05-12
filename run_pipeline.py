#!/usr/bin/env python3
# Usage examples:
#   python run_pipeline.py --workdir /path/to/workdir --all
#   python run_pipeline.py --workdir /path/to/workdir --all --resume
#   python run_pipeline.py --workdir /path/to/workdir --all --overwrite
#   python run_pipeline.py --workdir /path/to/workdir --all --dry-run --verbose
import argparse
import json
import logging
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    yaml = None


PATH_FLAGS = {
    "--initial_structures_path",
    "--init_dir_path",
    "--mask_dict_path",
    "--mask_path",
    "--config",
    "--mattersim_model",
    "--alignn_weights_dir",
}


@dataclass
class Condition:
    condition_id: str
    args: List[str]
    exp_name: str
    source_exp_name: str
    result_dir: Path
    completion_files: List[Path]
    missing_inputs: List[Path]


def _load_conditions(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"conditions file not found: {path}")
    if path.suffix in {".json"}:
        return json.loads(path.read_text()).get("conditions", [])
    if path.suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML conditions files")
        data = yaml.safe_load(path.read_text())
        if not isinstance(data, dict) or "conditions" not in data:
            raise ValueError("conditions file must contain a 'conditions' key")
        return data["conditions"]
    raise ValueError(f"unsupported conditions file extension: {path}")


def _get_arg(args: List[str], flag: str, default: Optional[str] = None) -> Optional[str]:
    for idx, item in enumerate(args):
        if item == flag and idx + 1 < len(args):
            return args[idx + 1]
    return default


def _compute_exp_name(args: List[str]) -> str:
    model_name = _get_arg(args, "--model_name")
    task = _get_arg(args, "--task")
    target_val = _get_arg(args, "--target_val")
    if model_name is None or task is None:
        raise ValueError("--model_name and --task are required in condition args")
    target_label = target_val if target_val is not None else "None"
    exp_name = f"{model_name}__{task}__{target_label}"
    spg = _get_arg(args, "--space_group")
    if spg is not None:
        try:
            if int(spg) > 0:
                exp_name += f"__spg-{spg}"
        except ValueError:
            pass
    return exp_name


def _sanitize_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in value)


def _find_missing_paths(args: List[str]) -> List[Path]:
    missing: List[Path] = []
    for idx, item in enumerate(args):
        if item in PATH_FLAGS and idx + 1 < len(args):
            candidate = Path(args[idx + 1]).expanduser()
            if not candidate.exists():
                missing.append(candidate)
    return missing


def _build_conditions(workdir: Path, raw_conditions: List[Dict[str, object]]) -> List[Condition]:
    conditions: List[Condition] = []
    for item in raw_conditions:
        if not isinstance(item, dict):
            raise ValueError("condition entries must be mapping objects")
        cond_id = item.get("id")
        args = item.get("args")
        exp_name_override = item.get("exp_name")
        if cond_id is None or args is None:
            raise ValueError("each condition must define 'id' and 'args'")
        if isinstance(args, str):
            args = shlex.split(args)
        if not isinstance(args, list):
            raise ValueError("condition args must be a list or shell string")
        args_list = [str(a) for a in args]
        source_exp_name = _compute_exp_name(args_list)
        if exp_name_override is not None:
            exp_name = str(exp_name_override)
        else:
            exp_name = source_exp_name
        result_dir = workdir / "results" / exp_name
        completion_files = [
            result_dir / "result.npz",
            result_dir / "result.csv",
            result_dir / "result_summary.csv",
        ]
        missing_inputs = _find_missing_paths(args_list)
        conditions.append(
            Condition(
                condition_id=_sanitize_id(str(cond_id)),
                args=args_list,
                exp_name=exp_name,
                source_exp_name=source_exp_name,
                result_dir=result_dir,
                completion_files=completion_files,
                missing_inputs=missing_inputs,
            )
        )
    return conditions


def _format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def _ensure_workdir(workdir: Path, dry_run: bool) -> None:
    if dry_run:
        return
    workdir.mkdir(parents=True, exist_ok=True)


def _write_manifest(path: Path, data: Dict[str, object], dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def _stage_poscars(
    conditions: List[Condition],
    staging_dir: Path,
    manifest_path: Path,
    dry_run: bool,
) -> List[Path]:
    poscar_inputs: List[Path] = []
    for cond in conditions:
        poscar_dir = cond.result_dir / "poscar"
        if poscar_dir.is_dir():
            for poscar in poscar_dir.glob("*.vasp"):
                poscar_inputs.append(poscar)
    if dry_run:
        return poscar_inputs

    staging_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, str] = {}
    for poscar in poscar_inputs:
        prefix = _sanitize_id(poscar.parent.parent.name)
        dest_name = f"{prefix}__{poscar.name}"
        dest = staging_dir / dest_name
        shutil.copy2(poscar, dest)
        manifest[str(dest)] = str(poscar)
    _write_manifest(manifest_path, manifest, dry_run)
    return poscar_inputs


def _status_label(
    *,
    missing_inputs: bool,
    complete: bool,
    resume: bool,
    overwrite: bool,
    outputs_exist: bool,
) -> str:
    if missing_inputs:
        return "MISSING_INPUT"
    if resume and complete:
        return "SKIP"
    if overwrite and outputs_exist:
        return "WOULD_OVERWRITE"
    return "RUN予定"


def _outputs_exist(files: Iterable[Path]) -> bool:
    return any(path.exists() for path in files)


def run_main_steps(
    *,
    workdir: Path,
    conditions: List[Condition],
    python_bin: str,
    main_script: Path,
    resume: bool,
    overwrite: bool,
    dry_run: bool,
    main_extra_args: List[str],
) -> None:
    for cond in conditions:
        outputs_exist = _outputs_exist(cond.completion_files)
        complete = all(path.exists() for path in cond.completion_files)
        missing_inputs = len(cond.missing_inputs) > 0
        status = _status_label(
            missing_inputs=missing_inputs,
            complete=complete,
            resume=resume,
            overwrite=overwrite,
            outputs_exist=outputs_exist,
        )

        cmd = [python_bin, str(main_script)] + cond.args + main_extra_args
        if "--exp_name" not in cmd:
            cmd += ["--exp_name", cond.exp_name]
        logging.info("condition %s status=%s", cond.condition_id, status)
        logging.info("main cmd: %s", _format_cmd(cmd))

        if dry_run:
            continue
        if missing_inputs:
            raise RuntimeError(
                f"missing input paths for {cond.condition_id}: {cond.missing_inputs}"
            )
        if resume and complete:
            continue
        if overwrite and outputs_exist and cond.result_dir.exists():
            shutil.rmtree(cond.result_dir)

        _ensure_workdir(workdir, dry_run)
        proc = subprocess.run(cmd, cwd=workdir)
        if proc.returncode != 0:
            raise RuntimeError(f"main.py failed for {cond.condition_id}")
        if cond.source_exp_name != cond.exp_name:
            src_dir = workdir / "results" / cond.source_exp_name
            dest_dir = cond.result_dir
            if src_dir.exists():
                if dest_dir.exists():
                    if overwrite:
                        shutil.rmtree(dest_dir)
                    else:
                        raise RuntimeError(
                            f"result dir exists and overwrite is disabled: {dest_dir}"
                        )
                shutil.move(str(src_dir), str(dest_dir))


def run_hull_step(
    *,
    workdir: Path,
    conditions: List[Condition],
    python_bin: str,
    hull_script: Path,
    hull_relax_result_dir: Path,
    hull_relax_db_path: Path,
    hull_save_dir: Path,
    hull_prop_relaxed: Path,
    resume: bool,
    overwrite: bool,
    dry_run: bool,
    hull_extra_args: List[str],
    mp_api_key: Optional[str],
) -> None:
    staging_dir = workdir / "staging" / "poscars_all"
    manifest_path = workdir / "manifests" / "poscar_manifest.json"
    poscar_inputs = _stage_poscars(conditions, staging_dir, manifest_path, dry_run)

    planned_main_runs = []
    for cond in conditions:
        outputs_exist = _outputs_exist(cond.completion_files)
        complete = all(path.exists() for path in cond.completion_files)
        missing_inputs = len(cond.missing_inputs) > 0
        planned_main_runs.append(
            _status_label(
                missing_inputs=missing_inputs,
                complete=complete,
                resume=resume,
                overwrite=overwrite,
                outputs_exist=outputs_exist,
            )
            == "RUN予定"
        )

    if dry_run and any(planned_main_runs):
        missing_inputs = False
    else:
        missing_inputs = len(poscar_inputs) == 0
    outputs_exist = hull_relax_db_path.exists() or (hull_relax_result_dir / "relax_results.csv").exists()
    complete = hull_relax_db_path.exists() and (hull_relax_result_dir / "relax_results.csv").exists()

    status = _status_label(
        missing_inputs=missing_inputs,
        complete=complete,
        resume=resume,
        overwrite=overwrite,
        outputs_exist=outputs_exist,
    )

    cmd = [
        python_bin,
        str(hull_script),
        "--poscar_dir_path",
        str(staging_dir),
        "--save_dir_path",
        str(hull_save_dir),
        "--prop_relaxed_poscars",
        str(hull_prop_relaxed),
        "--relax_result_dir",
        str(hull_relax_result_dir),
        "--relax_result_db_path",
        str(hull_relax_db_path),
    ] + hull_extra_args
    if mp_api_key:
        cmd += ["--mp_api_key", mp_api_key]

    logging.info("hull status=%s", status)
    logging.info("hull cmd: %s", _format_cmd(cmd))

    if dry_run:
        return
    if missing_inputs:
        raise RuntimeError("no POSCARs found to run hull calculation")
    if resume and complete:
        return
    if overwrite and hull_relax_db_path.exists():
        hull_relax_db_path.unlink()
    if overwrite and hull_relax_result_dir.exists():
        shutil.rmtree(hull_relax_result_dir)
    hull_relax_result_dir.mkdir(parents=True, exist_ok=True)
    hull_relax_db_path.parent.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(cmd, cwd=workdir)
    if proc.returncode != 0:
        raise RuntimeError("calculate_mattersim_convex_hull failed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SMOACS pipeline")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument(
        "--conditions-file",
        type=Path,
        default=Path("configs/pipeline_conditions.yaml"),
    )
    parser.add_argument("--conditions", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--python", dest="python_bin", default=sys.executable)
    parser.add_argument("--main-script", type=Path, default=Path("main.py"))
    parser.add_argument(
        "--hull-script",
        type=Path,
        default=Path("scripts/calculate_mattersim_convex_hull_energy.py"),
    )
    parser.add_argument(
        "--hull-relax-db",
        type=Path,
        default=None,
        help="Path to relax_results.json",
    )
    parser.add_argument(
        "--hull-relax-results-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--hull-save-dir",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--hull-prop-relaxed-poscars",
        type=Path,
        default=None,
    )
    parser.add_argument("--mp-api-key", type=str, default=None)
    parser.add_argument("--main-arg", action="append", default=[])
    parser.add_argument("--hull-arg", action="append", default=[])
    parser.add_argument("--skip-hull", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.resume:
        args.overwrite = False

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    repo_root = Path(__file__).resolve().parent
    workdir = args.workdir.expanduser().resolve()
    main_script = args.main_script
    if not main_script.is_absolute():
        main_script = (repo_root / main_script).resolve()
    hull_script = args.hull_script
    if not hull_script.is_absolute():
        hull_script = (repo_root / hull_script).resolve()

    conditions_path = args.conditions_file.expanduser().resolve()

    raw_conditions = _load_conditions(conditions_path)
    conditions = _build_conditions(workdir, raw_conditions)
    if args.conditions:
        selected_ids = {_sanitize_id(c) for c in args.conditions}
        conditions = [c for c in conditions if c.condition_id in selected_ids]
        missing = selected_ids - {c.condition_id for c in conditions}
        if missing:
            raise ValueError(f"unknown condition IDs: {sorted(missing)}")
    elif not args.all:
        args.all = True

    if not conditions:
        raise ValueError("no conditions selected")

    logging.info("conditions: %d", len(conditions))
    for cond in conditions:
        outputs_exist = _outputs_exist(cond.completion_files)
        complete = all(path.exists() for path in cond.completion_files)
        status = _status_label(
            missing_inputs=bool(cond.missing_inputs),
            complete=complete,
            resume=args.resume,
            overwrite=args.overwrite,
            outputs_exist=outputs_exist,
        )
        logging.info("cond=%s exp=%s status=%s", cond.condition_id, cond.exp_name, status)
        logging.info("result_dir: %s", cond.result_dir)
        for path in cond.completion_files:
            logging.info("completion: %s", path)
        if cond.missing_inputs:
            logging.info("missing_inputs: %s", [str(p) for p in cond.missing_inputs])
        cmd_preview = [args.python_bin, str(main_script)] + cond.args + args.main_arg
        logging.info("main cmd: %s", _format_cmd(cmd_preview))

    mp_api_key = args.mp_api_key or os.environ.get("MP_API_KEY")
    hull_relax_db = args.hull_relax_db or (workdir / "hull" / "relax_db" / "relax_results.json")
    hull_relax_results_dir = args.hull_relax_results_dir or (workdir / "hull")
    hull_save_dir = args.hull_save_dir or (workdir / "hull" / "DFT_POSCARS")
    hull_prop_relaxed = args.hull_prop_relaxed_poscars or (workdir / "hull" / "prop_relaxed_poscars")

    logging.info("hull relax db: %s", hull_relax_db)
    logging.info("hull relax results dir: %s", hull_relax_results_dir)

    run_main_steps(
        workdir=workdir,
        conditions=conditions,
        python_bin=args.python_bin,
        main_script=main_script,
        resume=args.resume,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        main_extra_args=args.main_arg,
    )

    if not args.skip_hull:
        run_hull_step(
            workdir=workdir,
            conditions=conditions,
            python_bin=args.python_bin,
            hull_script=hull_script,
            hull_relax_result_dir=hull_relax_results_dir,
            hull_relax_db_path=hull_relax_db,
            hull_save_dir=hull_save_dir,
            hull_prop_relaxed=hull_prop_relaxed,
            resume=args.resume,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
            hull_extra_args=args.hull_arg,
            mp_api_key=mp_api_key,
        )

    _write_manifest(
        workdir / "manifests" / "conditions.json",
        {
            "conditions": [
                {
                    "id": c.condition_id,
                    "exp_name": c.exp_name,
                    "result_dir": str(c.result_dir),
                    "args": c.args,
                }
                for c in conditions
            ]
        },
        args.dry_run,
    )


if __name__ == "__main__":
    main()
