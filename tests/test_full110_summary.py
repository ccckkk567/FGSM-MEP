from __future__ import annotations

import contextlib
import csv
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("full110_summary", ROOT / "summarize_cifar10_eps32_alpha8_full110.py")
summary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(summary)


class Full110SummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_run(self, *, count: int = 110, name: str | None = None) -> Path:
        run = self.root / (name or summary.RUN_NAMES[0])
        run.mkdir()
        (run / "config.yaml").write_text(yaml.safe_dump({"train": {"feature_weight": 50, "epochs": 110}}), encoding="utf-8")
        with (run / "epochs.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", *summary.METRICS, "vact_B"])
            writer.writeheader()
            for epoch in range(count):
                writer.writerow(dict(epoch=epoch, train_loss=2.3, train_accuracy=0.4, monitor_clean_accuracy=0.5, monitor_pgd10_accuracy=0.15 if epoch in (4, 101) else 0.1, lr=0.001 if epoch >= 105 else 0.1, vact_B=0.2))
        with (run / "loss_components.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", *summary.LOSSES])
            writer.writeheader()
            # Deliberately use the opposite ordering to the metric CSV.
            for epoch in reversed(range(count)):
                writer.writerow(dict(epoch=epoch, train_ce_loss=2.2, train_logit_mse=0.01, train_feature_mse=0.0001))
        (run / "final_metrics.json").write_text(json.dumps({"clean_accuracy": 0.51, "pgd10_accuracy": 0.11}), encoding="utf-8")
        return run

    def test_complete_history_and_latest_tied_best(self) -> None:
        result = summary.summarize_run(self.make_run())
        self.assertEqual(result["status"], "COMPLETE")
        self.assertEqual(result["epochs"], 110)
        self.assertEqual(result["best_epoch"], 101)
        self.assertEqual(result["final_lr"], 0.001)
        self.assertEqual(result["final_pgd"], 0.11)

    def test_copied_40_epoch_final_is_not_full_completion(self) -> None:
        result = summary.summarize_run(self.make_run(count=40))
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["best_epoch"], 4)
        self.assertIsNone(result["final_pgd"])

    def test_missing_and_incomplete_are_distinct(self) -> None:
        run = self.root / "missing"
        self.assertEqual(summary.summarize_run(run)["status"], "MISSING")
        run.mkdir()
        self.assertEqual(summary.summarize_run(run)["status"], "INCOMPLETE")

    def test_final_metrics_are_required(self) -> None:
        run = self.make_run()
        (run / "final_metrics.json").unlink()
        self.assertEqual(summary.summarize_run(run)["status"], "INCOMPLETE")

    def test_loss_log_is_required_but_valid_best_is_preserved(self) -> None:
        run = self.make_run()
        (run / "loss_components.csv").unlink()
        result = summary.summarize_run(run)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(result["best_epoch"], 101)
        self.assertIn("loss_components.csv absent", result["detail"])

    def test_gap_is_incomplete_even_with_final_metrics(self) -> None:
        run = self.make_run()
        path = run / "epochs.csv"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(lines[:41] + lines[42:]), encoding="utf-8")
        self.assertEqual(summary.summarize_run(run)["status"], "INCOMPLETE")

    def test_duplicate_epochs_in_either_csv_invalidate_best(self) -> None:
        for filename in ("epochs.csv", "loss_components.csv"):
            with self.subTest(filename=filename):
                run = self.make_run(name=filename)
                path = run / filename
                lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
                path.write_text("".join(lines + [lines[1]]), encoding="utf-8")
                result = summary.summarize_run(run)
                self.assertEqual(result["status"], "INVALID")
                self.assertIsNone(result["best_epoch"])

    def test_nonfinite_epoch_loss_or_final_metrics_is_invalid(self) -> None:
        cases = (("epochs.csv", "2.3", "nan"), ("loss_components.csv", "0.01", "inf"), ("final_metrics.json", "0.11", "NaN"))
        for filename, old, new in cases:
            with self.subTest(filename=filename):
                run = self.make_run(name=filename)
                path = run / filename
                path.write_text(path.read_text(encoding="utf-8").replace(old, new, 1), encoding="utf-8")
                result = summary.summarize_run(run)
                self.assertEqual(result["status"], "INVALID")
                self.assertIsNone(result["best_epoch"])

    def test_nonfinite_diagnostic_overrides_complete_history(self) -> None:
        run = self.make_run()
        (run / "nonfinite_diagnostic.json").write_text("{}", encoding="utf-8")
        self.assertEqual(summary.summarize_run(run)["status"], "INVALID")

    def test_out_of_range_accuracies_in_logs_and_final_are_invalid(self) -> None:
        cases = [
            ("epochs.csv", field)
            for field in summary.METRICS if field.endswith("_accuracy")
        ] + [("final_metrics.json", field) for field in ("clean_accuracy", "pgd10_accuracy")]
        for filename, field in cases:
            for value in (-0.001, 1.001):
                with self.subTest(filename=filename, field=field, value=value):
                    run = self.make_run(name=f"{filename}_{field}_{value}")
                    path = run / filename
                    if filename.endswith("csv"):
                        with path.open(newline="", encoding="utf-8") as handle:
                            rows = list(csv.DictReader(handle))
                        rows[0][field] = value
                        with path.open("w", newline="", encoding="utf-8") as handle:
                            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                            writer.writeheader()
                            writer.writerows(rows)
                    else:
                        data = json.loads(path.read_text(encoding="utf-8"))
                        data[field] = value
                        path.write_text(json.dumps(data), encoding="utf-8")
                    result = summary.summarize_run(run)
                    self.assertEqual(result["status"], "INVALID")
                    self.assertIsNone(result["best_epoch"])
                    self.assertIn("accuracy outside [0, 1]", result["detail"])

    def test_accuracy_range_endpoints_are_allowed(self) -> None:
        run = self.make_run()
        (run / "final_metrics.json").write_text(
            json.dumps({"clean_accuracy": 1.0, "pgd10_accuracy": 0.0}), encoding="utf-8"
        )
        self.assertEqual(summary.summarize_run(run)["status"], "COMPLETE")
        self.assertEqual(summary._accuracy(0), 0)
        self.assertEqual(summary._accuracy(1), 1)

    def test_loss_epoch_coverage_is_compared_by_epoch(self) -> None:
        run = self.make_run()
        path = run / "loss_components.csv"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        path.write_text("".join(lines[:-1]), encoding="utf-8")
        result = summary.summarize_run(run)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertIn("epoch sets differ", result["detail"])

    def test_main_prints_protocol_and_returns_completion_status(self) -> None:
        for name in summary.RUN_NAMES:
            self.make_run(name=name)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(summary.main([str(self.root)]), 0)
        self.assertIn("not a held-out", output.getvalue())
        self.assertIn("εE=32/255", output.getvalue())
        self.assertEqual(output.getvalue().count("| COMPLETE |"), 3)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(summary.main([str(self.root / "absent")]), 1)


if __name__ == "__main__":
    unittest.main()
