from __future__ import annotations

import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "high_eps_trajectory_screen", ROOT / "run_cifar10_high_eps_trajectory_screen.py"
)
screen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = screen
SPEC.loader.exec_module(screen)
SUMMARY_SPEC = importlib.util.spec_from_file_location(
    "high_eps_trajectory_summary", ROOT / "summarize_cifar10_high_eps_trajectory_screen.py"
)
summary = importlib.util.module_from_spec(SUMMARY_SPEC)
sys.modules[SUMMARY_SPEC.name] = summary
SUMMARY_SPEC.loader.exec_module(summary)


class HighEpsTrajectoryScreenTests(unittest.TestCase):
    def test_specs_cover_two_finite_candidates_per_radius(self) -> None:
        self.assertEqual(len(screen.SPECS), 6)
        self.assertEqual({spec.epsilon for spec in screen.SPECS}, {32, 48, 64})
        self.assertEqual(len({screen.run_name(spec) for spec in screen.SPECS}), 6)
        for epsilon in (32, 48, 64):
            subset = [spec for spec in screen.SPECS if spec.epsilon == epsilon]
            self.assertEqual(len(subset), 2)
            self.assertLess(subset[0].alpha, subset[1].alpha)

    def test_config_changes_only_explicit_diagnostic_fields(self) -> None:
        spec = screen.ScreenSpec(48, 12, 0.03, "test")
        config = screen.build_config(spec, data_root=Path("/tmp/data"), output_root=Path("/tmp/out"))
        baseline = copy.deepcopy(screen.load_config(ROOT / "configs" / "train" / "ours_fd_eps48.yaml"))
        baseline.pop("_config_path")
        baseline.update(name=screen.run_name(spec), deterministic=True, device="cuda:0")
        baseline["data"]["root"] = "/tmp/data"
        baseline["output"]["root"] = "/tmp/out"
        baseline["train"].update(
            epochs=40, epsilon=48, alpha=12, lr=0.03, monitor_pgd_steps=10,
            monitor_pgd_step_size=12, track_features=True, abort_on_nonfinite=True,
        )
        self.assertEqual(config, baseline)

    def test_summary_reports_best_to_final_pgd_drop(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "run"
            run.mkdir()
            with (run / "epochs.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("epoch", "train_loss", "monitor_clean_accuracy", "monitor_pgd10_accuracy", "vact_B"),
                )
                writer.writeheader()
                writer.writerow({"epoch": 0, "train_loss": 2.0, "monitor_clean_accuracy": 0.3,
                                 "monitor_pgd10_accuracy": 0.2, "vact_B": 1.0})
                writer.writerow({"epoch": 1, "train_loss": 2.1, "monitor_clean_accuracy": 0.4,
                                 "monitor_pgd10_accuracy": 0.1, "vact_B": 2.0})
            (run / "final_metrics.json").write_text(
                json.dumps({"clean_accuracy": 0.41, "pgd10_accuracy": 0.12}), encoding="utf-8"
            )
            result = summary.summarize_job({"run_dir": str(run)}, screen_epochs=2)
            self.assertEqual(result["status"], "COMPLETE")
            self.assertEqual(result["best_epoch"], 0)
            self.assertAlmostEqual(result["pgd_drop"], 0.08)
            self.assertAlmostEqual(result["final_vact_b"], 2.0)


if __name__ == "__main__":
    unittest.main()
