from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "native_high_eps_audit_summary", ROOT / "summarize_cifar10_native_high_eps_audit.py"
)
summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary
SPEC.loader.exec_module(summary)


def config(epsilon: int) -> dict:
    return {
        "train": {
            "objective": "ours_fd", "backend": "mep", "epochs": 1,
            "epsilon": epsilon, "alpha": epsilon, "lr": 0.1,
            "mep_logit_weight": 10, "fd_include_mep_logit": True,
            "feature_node": "B", "feature_weight": 200, "abort_on_nonfinite": True,
        }
    }


class NativeHighEpsAuditSummaryTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def make_run(self, epsilon: int) -> Path:
        run = self.root / f"native_audit_ours_fd_eps{epsilon}"
        run.mkdir()
        (run / "config.yaml").write_text(yaml.safe_dump(config(epsilon)), encoding="utf-8")
        return run

    def test_finite_prefix_is_not_reported_as_success(self) -> None:
        run = self.make_run(32)
        (run / "final.pt").write_bytes(b"checkpoint")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(summary.main([str(self.root)]), 1)
        self.assertIn("FINITE_PREFIX", output.getvalue())
        self.assertIn("continue unchanged", output.getvalue())

    def test_nonfinite_diagnostic_is_reported_as_na_without_attack(self) -> None:
        run = self.make_run(32)
        (run / "nonfinite_diagnostic.json").write_text(
            json.dumps(
                {"stage": "backward", "epoch": 0, "batch": 8,
                 "tensors": {"loss": {"finite_fraction": 0.0}}}
            ), encoding="utf-8",
        )
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(summary.main([str(self.root)]), 1)
        text = output.getvalue()
        self.assertIn("NUMERICAL_DIVERGENCE", text)
        self.assertIn("record as N/A; do not run attacks", text)

    def test_invalid_config_is_not_mislabeled_as_baseline_failure(self) -> None:
        run = self.make_run(32)
        (run / "config.yaml").write_text(yaml.safe_dump({"train": {"epsilon": 32}}), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(summary.main([str(self.root)]), 1)
        self.assertIn("INVALID_CONFIG", output.getvalue())


if __name__ == "__main__":
    unittest.main()
