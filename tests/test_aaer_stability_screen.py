from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aaer_stability_screen", ROOT / "run_aaer_ours_fd_stability_screen.py"
)
screen = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = screen
SPEC.loader.exec_module(screen)


class AAERStabilityScreenTests(unittest.TestCase):
    def test_full_matrix_has_unique_all_epsilon_three_seed_candidates(self) -> None:
        jobs = screen.candidates()
        expected = (
            len(screen.EPSILON_ALPHAS)
            * len(screen.FEATURE_WEIGHTS)
            * len(screen.LEARNING_RATES)
            * len(screen.SEEDS)
        )
        self.assertEqual(len(jobs), expected)
        self.assertEqual(len({screen.run_name(job) for job in jobs}), expected)
        self.assertEqual({job.epsilon for job in jobs}, {8, 12, 16, 32, 48, 64})
        for epsilon, alpha in screen.EPSILON_ALPHAS:
            subset = [job for job in jobs if job.epsilon == epsilon]
            self.assertEqual({job.alpha for job in subset}, {alpha})
            self.assertEqual({job.feature_weight for job in subset}, set(screen.FEATURE_WEIGHTS))
            self.assertEqual({job.lr for job in subset}, set(screen.LEARNING_RATES))
            self.assertEqual({job.seed for job in subset}, set(screen.SEEDS))

    def test_screen_config_is_aaer_preact_but_does_not_keep_large_resume_state(self) -> None:
        spec = screen.ScreenSpec(epsilon=32, alpha=8, feature_weight=10.0, lr=0.03, seed=2)
        config = screen.build_config(spec, data_root=Path("/tmp/data"), output_root=Path("/tmp/out"))
        self.assertEqual(config["name"], screen.run_name(spec))
        self.assertEqual(config["model"]["name"], "preactresnet18")
        self.assertEqual(config["model"]["input_normalization"], "cifar10")
        self.assertEqual(config["data"]["augmentation"], "aaer")
        self.assertEqual(config["train"]["objective"], "ours_fd")
        self.assertEqual(config["train"]["epochs"], 40)
        self.assertEqual(config["train"]["epsilon"], 32)
        self.assertEqual(config["train"]["alpha"], 8)
        self.assertEqual(config["train"]["feature_weight"], 10)
        self.assertEqual(config["train"]["lr"], 0.03)
        self.assertEqual(config["train"]["monitor_subset"], 1000)
        self.assertEqual(config["train"]["save_resume"], False)
        self.assertEqual(config["train"]["monitor_pgd_step_size"], 8)


if __name__ == "__main__":
    unittest.main()
