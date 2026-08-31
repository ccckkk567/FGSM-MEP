from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "high_eps_trajectory_continuation", ROOT / "continue_cifar10_high_eps_trajectory_full110.py"
)
continuation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = continuation
SPEC.loader.exec_module(continuation)


class HighEpsTrajectoryContinuationTests(unittest.TestCase):
    def test_only_selected_48_and_64_candidates_are_continued(self) -> None:
        self.assertEqual([(run.epsilon, run.alpha, run.lr) for run in continuation.RUNS], [(48, 12, 0.03), (64, 16, 0.03)])
        self.assertEqual(len({run.name for run in continuation.RUNS}), 2)

    def test_continuation_changes_only_duration_and_location(self) -> None:
        source = {
            "name": "trajectory_ours_fd_eps48_alpha12_lr0p03",
            "seed": 0,
            "deterministic": True,
            "device": "cuda:0",
            "data": {"root": "/source/data", "num_workers": 0},
            "output": {"root": "/source/output"},
            "train": {
                "objective": "ours_fd", "backend": "mep", "epsilon": 48, "alpha": 12,
                "lr": 0.03, "mep_logit_weight": 10.0, "feature_node": "B", "feature_weight": 200.0,
                "fd_include_mep_logit": True, "monitor_pgd_steps": 10, "monitor_pgd_step_size": 12,
                "track_features": True, "abort_on_nonfinite": True, "epochs": 40,
            },
        }
        spec = continuation.RUNS[0]
        continuation._check_recipe(source, spec, target_epochs=40)
        actual = continuation._continuation_config(source, spec, Path("/target/output"))
        actual["data"]["root"] = "/target/data"
        expected = copy.deepcopy(source)
        expected.update(name=spec.name, device="cuda:0")
        expected["data"]["root"] = "/target/data"
        expected["output"]["root"] = "/target/output"
        expected["train"]["epochs"] = 110
        self.assertEqual(actual, expected)
        self.assertEqual(continuation._scientific_config(actual), continuation._scientific_config(source))

    def test_prepare_validates_copied_source_before_archiving_its_final_artifacts(self) -> None:
        spec = continuation.RUNS[0]
        source_config = {
            "name": spec.source_name, "seed": 0, "deterministic": True, "device": "cuda:0",
            "data": {"root": "/source/data"}, "output": {"root": "/source/output"},
            "train": {"epochs": 40},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            source = source_root / spec.source_name
            source.mkdir(parents=True)
            for name in ("config.yaml", "resume.pt", "best.pt", "final.pt", "final_metrics.json", "environment.json"):
                (source / name).write_text(name, encoding="utf-8")
            seen_final_artifacts: list[bool] = []

            def inspect(run: Path, expected: dict, *, source: bool) -> tuple[int, bool]:
                if source:
                    seen_final_artifacts.append((run / "final.pt").is_file() and (run / "final_metrics.json").is_file())
                return (39, True)

            with patch.object(continuation, "_source_config", return_value=source_config), patch.object(
                continuation, "_inspect_run", side_effect=inspect
            ):
                epoch, complete = continuation.prepare(
                    spec, source_root=source_root, output_root=root / "out", data_root=root / "data"
                )

            prepared = root / "out" / spec.name
            self.assertEqual((epoch, complete), (39, False))
            self.assertEqual(seen_final_artifacts, [True, True])
            self.assertTrue((prepared / "source_trajectory" / "final.pt").is_file())
            self.assertFalse((prepared / "final.pt").exists())


if __name__ == "__main__":
    unittest.main()
