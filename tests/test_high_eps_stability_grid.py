from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "high_eps_stability_grid", ROOT / "run_cifar10_high_eps_stability_grid.py"
)
grid = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = grid
SPEC.loader.exec_module(grid)


class HighEpsStabilityGridTests(unittest.TestCase):
    def test_grid_has_all_ratios_lrs_and_unique_names(self) -> None:
        jobs = grid.candidates()
        self.assertEqual(len(jobs), 27)
        self.assertEqual(len({grid.run_name(*job) for job in jobs}), 27)
        for epsilon in grid.EPSILONS:
            subset = [(alpha, lr) for value, alpha, lr in jobs if value == epsilon]
            self.assertEqual({alpha for alpha, _ in subset}, {epsilon // 8, epsilon // 4, epsilon // 2})
            self.assertEqual({lr for _, lr in subset}, set(grid.LEARNING_RATES))

    def test_config_changes_only_declared_diagnostic_axes(self) -> None:
        base = grid.load_config(ROOT / "configs" / "train" / "ours_fd_eps48.yaml")
        config = grid.build_config(
            epsilon=48, alpha=12, lr=0.03, data_root=Path("/tmp/data"), output_root=Path("/tmp/out")
        )
        baseline = copy.deepcopy(base)
        baseline.pop("_config_path")
        baseline.update(name=grid.run_name(48, 12, 0.03), deterministic=True, device="cuda:0")
        baseline["data"]["root"] = "/tmp/data"
        baseline["output"]["root"] = "/tmp/out"
        baseline["train"].update(
            epochs=1, epsilon=48, alpha=12, lr=0.03, monitor_pgd_steps=10,
            monitor_pgd_step_size=12, track_features=False, abort_on_nonfinite=True,
        )
        self.assertEqual(config, baseline)


if __name__ == "__main__":
    unittest.main()
