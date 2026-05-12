#!/usr/bin/env python
"""
Simplified MatterSim relaxation for proposal structures only.
- Relaxes POSCAR/vasp files under --p_dir.
- Saves relaxed structures under --r_dir.
- Does NOT touch relax_results.json/gz; no reference phases are fetched.
"""

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
from pymatgen.core import Structure

# Reuse existing helpers from the main script
from calculate_mattersim_convex_hull_energy import (
    _relax_and_save_with_retry,
    chemsys_by_atomic_number,
)


def list_poscars(p_dir: Path) -> List[Path]:
    exts = {"poscar", "vasp"}
    files = []
    for p in sorted(p_dir.glob("**/*")):
        if p.is_file() and p.suffix.lower().lstrip(".") in exts:
            files.append(p)
    return files


def relax_proposals(p_dir: Path, r_dir: Path, steps_max: int, log_path: Path) -> None:
    r_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    poscars = list_poscars(p_dir)
    total = len(poscars)
    if total == 0:
        print("No proposal structures found.")
        return
    t_start = time.time()
    with log_path.open("a", encoding="utf-8") as flog:
        for idx, poscar in enumerate(poscars, 1):
            struct = Structure.from_file(poscar)
            rec = _relax_and_save_with_retry(
                mp_id=poscar.stem,
                struct=struct,
                outdir_path=r_dir,
                steps_max=steps_max,
            )
            rec["chem_sys"] = chemsys_by_atomic_number(struct.symbol_set)
            rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            rec["idx"] = idx
            rec["total"] = total
            flog.write(json.dumps(rec, ensure_ascii=True) + "\n")
            flog.flush()
            print(f"[{idx}/{total}] {poscar.name}: status={rec.get('status','ok')} file={rec.get('file')}")
    print(f"Done. proposals={total}, elapsed={time.time()-t_start:.1f}s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Relax proposals only with MatterSim.")
    parser.add_argument("--p_dir", required=True, help="Proposal POSCAR directory")
    parser.add_argument("--r_dir", default="./prop_relaxed_poscars", help="Output directory for relaxed proposals")
    parser.add_argument("--steps_max", type=int, default=400, help="Max relaxation steps (MatterSim)")
    parser.add_argument("--log", default="./logs/proposal_relax.log", help="Path to JSONL log")
    args = parser.parse_args()

    p_dir = Path(args.p_dir)
    r_dir = Path(args.r_dir)
    log_path = Path(args.log)

    relax_proposals(p_dir, r_dir, args.steps_max, log_path)


if __name__ == "__main__":
    main()
