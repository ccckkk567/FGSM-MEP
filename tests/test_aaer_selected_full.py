from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "aaer_selected_full", ROOT / "run_aaer_ours_fd_selected_full.py"
)
selected = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = selected
SPEC.loader.exec_module(selected)


class AAERSelectedFullTests(unittest.TestCase):
    def test_frozen_selection_covers_six_epsilons_and_three_seeds(self) -> None:
        specs = selected.SELECTED_SPECS
        self.assertEqual([spec.epsilon for spec in specs], [8, 12, 16, 32, 48, 64])
        self.assertEqual(len({selected.run_name(spec, seed) for spec in specs for seed in (0, 1, 2)}), 18)
        self.assertEqual((specs[-1].feature_weight, specs[-1].lr), (10.0, 0.01))

    def test_final_config_uses_original_schedule_and_final_only_provenance(self) -> None:
        spec = next(item for item in selected.SELECTED_SPECS if item.epsilon == 32)
        config = selected.build_config(
            spec, 1, data_root=Path("/tmp/data"), output_root=Path("/tmp/runs")
        )
        self.assertEqual(config["name"], selected.run_name(spec, 1))
        self.assertEqual(config["model"]["name"], "preactresnet18")
        self.assertEqual(config["model"]["input_normalization"], "cifar10")
        self.assertEqual(config["data"]["augmentation"], "aaer")
        self.assertEqual(config["train"]["objective"], "ours_fd")
        self.assertEqual(config["train"]["epochs"], 110)
        self.assertEqual(config["train"]["milestones"], [100, 105])
        self.assertEqual(config["train"]["epsilon"], 32)
        self.assertEqual(config["train"]["alpha"], 8)
        self.assertEqual(config["train"]["feature_weight"], 25)
        self.assertEqual(config["train"]["mep_logit_weight"], 10.0)
        self.assertTrue(config["train"]["save_resume"])
        self.assertEqual(config["train"]["monitor_subset"], 1000)


if __name__ == "__main__":
    unittest.main()
