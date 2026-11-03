"""Prepare unchanged SMOACS POSCARs and record each input selection decision."""

import argparse
import csv
import hashlib
import math
import re
from pathlib import Path


def count_elements(data):
    """Count occupied VASP5 species without changing the structure."""
    lines = data.decode("utf-8").splitlines()
    elements = lines[5].split()
    counts = [int(value) for value in lines[6].split()]
    if len(elements) != len(counts) or any(count < 0 for count in counts):
        raise ValueError("Invalid POSCAR species/counts")
    if not elements or not all(re.fullmatch(r"[A-Z][a-z]?", item) for item in elements):
        raise ValueError("Expected VASP5 element symbols")
    occupied = {element for element, count in zip(elements, counts) if count > 0}
    if not occupied:
        raise ValueError("Empty POSCAR")
    return len(occupied)


def selection(section, row, n_elements):
    if section == "section_2_3":
        return "not_applied", "not_applied", True
    gap = float(row["bandgap_onehot"])
    if not math.isfinite(gap):
        raise ValueError(f"Nonfinite bandgap_onehot: {row['original_fnames']}")
    distance = row["minbond_less_than_0.5"]
    if distance not in ("True", "False"):
        raise ValueError(f"Invalid saved distance check: {row['original_fnames']}")
    stage1 = gap <= 0.05 and n_elements <= 5
    return "pass" if stage1 else "fail", distance, stage1 and distance == "True"


def plan_inputs(section, experiment, result_dir, output_dir):
    """Validate every source before creating any output; also usable read-only."""
    if section not in ("section_2_3", "section_2_4"):
        raise ValueError(f"Unsupported section: {section}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", experiment):
        raise ValueError("Experiment must be a single directory name")
    with (result_dir / "result.csv").open(newline="") as handle:
        results = list(csv.DictReader(handle))
    if not results:
        raise ValueError(f"No result rows: {result_dir}")
    records, copies, seen = [], [], set()
    for row in results:
        original = row["original_fnames"]
        if Path(original).name != original or not original.endswith(".vasp"):
            raise ValueError(f"Invalid original_fnames: {original}")
        if original in seen:
            raise ValueError(f"Duplicate original_fnames: {original}")
        seen.add(original)
        source = result_dir / "poscar" / f"Optimized_based_on_{original}"
        data = source.read_bytes()
        n_elements = count_elements(data)
        stage1, distance, selected = selection(section, row, n_elements)
        name = f"{source.stem}__{experiment}.vasp"
        destination = output_dir / name
        records.append(
            {
                "experiment_id": experiment,
                "original_fnames": original,
                "mattersim_fname": name.removesuffix(".vasp"),
                "bandgap_onehot": row.get("bandgap_onehot", ""),
                "n_elements": n_elements,
                "stage1_bg_elements_status": stage1,
                "saved_distance_check": distance,
                "selected_for_mattersim": selected,
                "source_poscar": str(
                    Path("results") / experiment / "poscar" / source.name
                ),
                "input_poscar": (
                    str(Path("mattersim/to_relax_poscars") / experiment / name)
                    if selected
                    else ""
                ),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        if selected:
            copies.append((destination, data))
    expected = {f"Optimized_based_on_{original}" for original in seen}
    actual = {path.name for path in (result_dir / "poscar").glob("*.vasp")}
    if actual != expected:
        raise ValueError("POSCAR files and result.csv do not match exactly")
    return records, copies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("section", choices=("section_2_3", "section_2_4"))
    parser.add_argument("experiment", help="Directory name under <repo_root>/results")
    args = parser.parse_args()
    repro = Path(__file__).resolve().parent
    repo = repro.parent
    output = repro / args.section / "mattersim/to_relax_poscars" / args.experiment
    manifest = (
        repro
        / args.section
        / "screening_logs"
        / f"{args.experiment}_mattersim_inputs.csv"
    )
    if output.exists() or manifest.exists():
        raise FileExistsError(
            "MatterSim inputs/manifest already exist; inspect the existing run"
        )
    records, copies = plan_inputs(
        args.section, args.experiment, repo / "results" / args.experiment, output
    )
    output.mkdir(parents=True)
    for destination, data in copies:
        with destination.open("xb") as handle:
            handle.write(data)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(f"Selected {len(copies)}/{len(records)} POSCARs: {output}")
    print(f"Selection and ID mapping: {manifest}")


if __name__ == "__main__":
    main()
