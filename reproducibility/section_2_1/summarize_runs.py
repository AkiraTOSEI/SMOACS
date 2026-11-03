"""Average the four 128-candidate runs for each Section 2.1 condition."""

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean


INITIALIZATIONS = (
    "random-random",
    "random-AtomicMask",
    "Struct-random",
    "Struct-AtomicMask",
)
METHODS = ("grad", "grad_sa", "mutation", "mutation_sa")
METRICS = (
    "success",
    "bandgap_success",
    "Ef_success",
    "valid_structure",
    "minbond_less_than_0.5",
    "is_neutral_cmn",
    "perov_success",
    "perov_coords",
    "tolerance_success",
    "tolerance",
    "element_diversity",
    "success_element_diversity",
)


def summarize_runs(output_root: Path, seeds=(42, 43, 44, 45)) -> Path:
    if len(seeds) != 4 or len(set(seeds)) != 4:
        raise ValueError("Exactly four distinct seeds are required")
    means = []
    for initialization in INITIALIZATIONS:
        for method in METHODS:
            rows = []
            reference_config = None
            for seed in seeds:
                exp_name = f"section_2_1_{initialization}_{method}__seed{seed}"
                run_dir = output_root / exp_name
                config = json.loads((run_dir / "run_config.json").read_text())
                expected = {
                    "initialization": initialization,
                    "method": method,
                    "seed": seed,
                    "exp_name": exp_name,
                    "num_candidate": 128,
                    "b_size": 128,
                    "num_steps": 200,
                }
                if any(config.get(key) != value for key, value in expected.items()):
                    raise ValueError(f"Unexpected run configuration: {run_dir}")
                shared_config = {
                    key: value
                    for key, value in config.items()
                    if key not in {"seed", "exp_name"}
                }
                if reference_config is None:
                    reference_config = shared_config
                elif shared_config != reference_config:
                    raise ValueError(f"Settings differ between repetitions: {run_dir}")
                with (run_dir / "result_summary.csv").open(newline="") as handle:
                    saved_rows = list(csv.DictReader(handle))
                if len(saved_rows) != 1:
                    raise ValueError(f"Expected one summary row: {run_dir}")
                row = saved_rows[0]
                expected_summary = {
                    "initialization": initialization,
                    "seed": str(seed),
                    "num_candidate": "128",
                    "batch_size": "128",
                    "num_steps": "200",
                }
                if any(row.get(key) != value for key, value in expected_summary.items()):
                    raise ValueError(f"Summary does not match the run: {run_dir}")
                rows.append(row)
            means.append(
                {
                    "initialization": initialization,
                    "method": method,
                    "n_runs": len(seeds),
                    "seeds": " ".join(map(str, seeds)),
                    "num_candidate": 128,
                    "batch_size": 128,
                    "num_steps": 200,
                    **{
                        f"{metric}_mean": fmean(
                            float(row[metric]) if row[metric] else float("nan")
                            for row in rows
                        )
                        for metric in METRICS
                    },
                }
            )
    output_path = output_root / "section_2_1_mean.csv"
    with output_path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(means[0]))
        writer.writeheader()
        writer.writerows(means)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "results",
    )
    parser.add_argument("--seeds", type=int, nargs=4, default=[42, 43, 44, 45])
    args = parser.parse_args()
    print(summarize_runs(args.output_root, args.seeds))


if __name__ == "__main__":
    main()
