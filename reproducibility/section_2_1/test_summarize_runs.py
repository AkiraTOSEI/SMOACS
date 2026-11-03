"""Check that Section 2.1 means use complete, comparable repetitions."""

import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from reproducibility.section_2_1.summarize_runs import summarize_runs


class Section21SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.output_root = Path(self.temp_dir.name)
        self.run_dirs = []
        for initialization in (
            "random-random",
            "random-AtomicMask",
            "Struct-random",
            "Struct-AtomicMask",
        ):
            for method in ("grad", "grad_sa", "mutation", "mutation_sa"):
                for seed, rate in zip((42, 43, 44, 45), (0.0, 0.25, 0.5, 1.0)):
                    exp_name = f"section_2_1_{initialization}_{method}__seed{seed}"
                    run_dir = self.output_root / exp_name
                    run_dir.mkdir()
                    self.run_dirs.append(run_dir)
                    config = {
                        "initialization": initialization,
                        "method": method,
                        "seed": seed,
                        "exp_name": exp_name,
                        "num_candidate": 128,
                        "b_size": 128,
                        "num_steps": 200,
                        "template_ids": ["template-a", "template-b"],
                    }
                    (run_dir / "run_config.json").write_text(json.dumps(config))
                    row = {
                        "initialization": initialization,
                        "seed": seed,
                        "num_candidate": 128,
                        "batch_size": 128,
                        "num_steps": 200,
                        "success": rate,
                        "bandgap_success": 0.75,
                        "Ef_success": 0.5,
                        "valid_structure": 1.0,
                        "minbond_less_than_0.5": 0.0,
                        "is_neutral_cmn": 1.0,
                        "perov_success": 1.0,
                        "perov_coords": 1.0,
                        "tolerance_success": 1.0,
                        "tolerance": "",
                        "element_diversity": 0.25,
                        "success_element_diversity": "" if seed == 42 else 0.5,
                    }
                    with (run_dir / "result_summary.csv").open(
                        "w", newline=""
                    ) as handle:
                        writer = csv.DictWriter(handle, fieldnames=list(row))
                        writer.writeheader()
                        writer.writerow(row)

    def test_averages_all_four_runs_without_dropping_undefined_metrics(self):
        path = summarize_runs(self.output_root)
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 16)
        self.assertEqual(
            len({(row["initialization"], row["method"]) for row in rows}), 16
        )
        for row in rows:
            self.assertEqual(row["n_runs"], "4")
            self.assertEqual(row["seeds"], "42 43 44 45")
            self.assertEqual(float(row["success_mean"]), 0.4375)
            self.assertEqual(float(row["bandgap_success_mean"]), 0.75)
            self.assertTrue(math.isnan(float(row["success_element_diversity_mean"])))
            self.assertTrue(math.isnan(float(row["tolerance_mean"])))

    def test_missing_run_does_not_produce_partial_means(self):
        (self.run_dirs[-1] / "result_summary.csv").unlink()
        with self.assertRaises(FileNotFoundError):
            summarize_runs(self.output_root)
        self.assertFalse((self.output_root / "section_2_1_mean.csv").exists())

    def test_rejects_wrong_candidate_count_and_mismatched_inputs(self):
        path = self.run_dirs[1] / "run_config.json"
        original = json.loads(path.read_text())
        for change in ({"num_candidate": 256}, {"template_ids": ["different"]}):
            with self.subTest(change=change):
                path.write_text(json.dumps(original | change))
                with self.assertRaises(ValueError):
                    summarize_runs(self.output_root)
                self.assertFalse((self.output_root / "section_2_1_mean.csv").exists())

    def test_rejects_duplicate_seeds(self):
        with self.assertRaises(ValueError):
            summarize_runs(self.output_root, (42, 42, 43, 44))

    def test_does_not_overwrite_existing_summary(self):
        path = self.output_root / "section_2_1_mean.csv"
        path.write_text("existing results\n")
        with self.assertRaises(FileExistsError):
            summarize_runs(self.output_root)
        self.assertEqual(path.read_text(), "existing results\n")


if __name__ == "__main__":
    unittest.main()
