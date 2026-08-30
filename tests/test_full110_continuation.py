from __future__ import annotations

import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from contextlib import redirect_stderr, redirect_stdout

import yaml

from co_blessing.config import load_config


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "full110_continuation", ROOT / "continue_cifar10_eps32_alpha8_full110.py"
)
continuation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = continuation  # dataclasses resolves postponed annotations here.
SPEC.loader.exec_module(continuation)


def fixture_loader(path: Path) -> dict:
    """Lightweight metadata fixtures; real tensor finiteness has a separate test."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("mep"):
        for name in ("delta", "momentum"):
            value["mep"][name] = SimpleNamespace(shape=tuple(value["mep"][name]["shape"]))
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_run(run: Path, config: dict, epoch: int = 39, *, final: bool = True) -> None:
    run.mkdir(parents=True, exist_ok=True)
    config = {key: value for key, value in config.items() if not key.startswith("_")}
    (run / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    write_json(run / "environment.json", {"torch": "fixture", "git_revision": "fixture"})
    steps = (epoch + 1) * continuation.BATCHES_PER_EPOCH
    milestones = {100 * continuation.BATCHES_PER_EPOCH: 1, 105 * continuation.BATCHES_PER_EPOCH: 1}
    lr = 0.1 * 0.1 ** sum(steps >= value for value in milestones)
    with (run / "epochs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "epoch", "lr", "train_loss", "train_accuracy", "monitor_clean_accuracy",
            "monitor_pgd10_accuracy",
        ))
        writer.writeheader()
        for index in range(epoch + 1):
            writer.writerow(dict(epoch=index, lr=lr, train_loss=2.2, train_accuracy=0.3,
                                 monitor_clean_accuracy=0.4,
                                 monitor_pgd10_accuracy=0.15 if index == 4 else 0.1))
    with (run / "loss_components.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(
            "epoch", "train_ce_loss", "train_logit_mse", "train_feature_mse",
        ))
        writer.writeheader()
        for index in range(epoch + 1):
            writer.writerow(dict(epoch=index, train_ce_loss=2.1, train_logit_mse=0.01,
                                 train_feature_mse=0.0))
    saved = {
        "epoch": epoch, "config": config, "model": {"weight": 1}, "best_pgd": 0.15,
        "optimizer": {"param_groups": [{"lr": lr}]},
        "scheduler": {"last_epoch": steps, "milestones": milestones, "gamma": 0.1},
        "mep": {"sample_count": 50000, "image_shape": [3, 32, 32],
                "epsilon": 32 / 255, "alpha": 8 / 255, "last_reset_epoch": (epoch // 40) * 40,
                "delta": {"shape": [50000, 3, 32, 32]},
                "momentum": {"shape": [50000, 3, 32, 32]}},
        "rng": {"python": 1, "numpy": 2, "torch": 3, "loader_generator": 4, "cuda": [5]},
    }
    write_json(run / "resume.pt", saved)
    write_json(run / "best.pt", {"epoch": 4, "config": config, "model": {"weight": 1},
                                 "metrics": {"clean_accuracy": 0.4, "pgd10_accuracy": 0.15}})
    if final:
        metrics = {"clean_accuracy": 0.4, "pgd10_accuracy": 0.11}
        write_json(run / "final_metrics.json", metrics)
        write_json(run / "final.pt", {"epoch": epoch, "config": config,
                                      "model": {"weight": 1}, "metrics": metrics})


class Full110ContinuationTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source_root = self.root / "pilots"
        self.output_root = self.root / "full110"
        self.spec = continuation.RUNS[0]
        self.source = self.source_root / self.spec.source_name
        self.run = self.output_root / self.spec.name
        self.expected = load_config(
            continuation.ROOT / "configs" / "train" / f"{self.spec.source_name}.yaml"
        )
        write_run(self.source, self.expected)

    def prepare(self) -> continuation.RunState:
        return continuation.prepare_run(
            self.spec, self.source_root, self.output_root, self.root / "data",
            checkpoint_loader=fixture_loader,
        )

    def inspect(self, run: Path, *, source: bool = False) -> continuation.RunState:
        return continuation.inspect_run(
            run, self.expected, source=source, checkpoint_loader=fixture_loader,
        )

    def mutate_checkpoint(self, filename: str, mutation, *, run: Path | None = None) -> None:
        path = (run or self.source) / filename
        saved = json.loads(path.read_text(encoding="utf-8"))
        mutation(saved)
        write_json(path, saved)

    def test_copies_state_and_best_but_leaves_source_untouched(self) -> None:
        original = {p.name: p.read_bytes() for p in self.source.iterdir() if p.is_file()}
        state = self.prepare()
        self.assertEqual(state, continuation.RunState(39, 4, False))
        self.assertEqual(original, {p.name: p.read_bytes() for p in self.source.iterdir() if p.is_file()})
        for filename in ("resume.pt", "best.pt", "epochs.csv", "loss_components.csv"):
            self.assertEqual((self.run / filename).read_bytes(), original[filename])
        for filename in ("config.yaml", "environment.json", "final.pt", "final_metrics.json"):
            self.assertEqual((self.run / "source_pilot" / filename).read_bytes(), original[filename])
        self.assertFalse((self.run / "final.pt").exists())
        self.assertFalse((self.run / "final_metrics.json").exists())
        config = load_config(self.run / "continuation_config.yaml")
        self.assertEqual(config["train"]["epochs"], 110)
        self.assertEqual(config["name"], self.spec.name)
        self.assertEqual(config["train"]["milestones"], [100, 105])
        self.assertEqual(config["train"]["monitor_pgd_step_size"], 8)
        self.assertEqual(continuation.scientific_config(config), continuation.scientific_config(self.expected))
        self.assertEqual(self.prepare(), state)

    def test_progressed_destination_is_not_overwritten_on_rerun(self) -> None:
        self.prepare()
        config = load_config(self.run / "continuation_config.yaml")
        write_run(self.run, config, epoch=40, final=False)
        before = (self.run / "resume.pt").read_bytes()
        self.assertEqual(self.prepare().epoch, 40)
        self.assertEqual((self.run / "resume.pt").read_bytes(), before)

    def test_completed_110_run_accepts_original_best_metadata(self) -> None:
        self.prepare()
        config = load_config(self.run / "continuation_config.yaml")
        write_run(self.run, config, epoch=109)
        (self.run / "best.pt").write_bytes((self.source / "best.pt").read_bytes())
        self.assertEqual(self.prepare(), continuation.RunState(109, 4, True))

    def test_epoch109_without_final_is_ready_for_finalization(self) -> None:
        self.prepare()
        write_run(self.run, load_config(self.run / "continuation_config.yaml"), epoch=109, final=False)
        self.assertFalse(self.prepare().complete)

    def test_old_final_cannot_mark_destination_complete(self) -> None:
        self.prepare()
        (self.run / "final.pt").write_bytes((self.source / "final.pt").read_bytes())
        with self.assertRaisesRegex(ValueError, "stale final"):
            self.prepare()

    def test_duplicate_or_ahead_logs_are_rejected_without_trimming(self) -> None:
        for filename in ("epochs.csv", "loss_components.csv"):
            with self.subTest(filename=filename):
                path = self.source / filename
                original = path.read_text(encoding="utf-8")
                path.write_text(original + original.splitlines(keepends=True)[-1], encoding="utf-8")
                before = path.read_bytes()
                with self.assertRaisesRegex(ValueError, "no automatic trimming"):
                    self.prepare()
                self.assertEqual(path.read_bytes(), before)
                self.assertFalse(self.run.exists())
                path.write_text(original, encoding="utf-8")

    def test_mismatched_best_is_rejected(self) -> None:
        self.mutate_checkpoint("best.pt", lambda value: value.update(epoch=40))
        with self.assertRaisesRegex(ValueError, "best.pt disagrees"):
            self.prepare()

    def test_scheduler_and_rng_states_are_required(self) -> None:
        self.mutate_checkpoint("resume.pt", lambda value: value["scheduler"].update(last_epoch=0))
        with self.assertRaisesRegex(ValueError, "scheduler state"):
            self.prepare()
        write_run(self.source, self.expected)
        self.mutate_checkpoint("resume.pt", lambda value: value["rng"].pop("loader_generator"))
        with self.assertRaisesRegex(ValueError, "missing RNG"):
            self.prepare()

    def test_changed_recipe_is_rejected(self) -> None:
        self.mutate_checkpoint("resume.pt", lambda value: value["config"]["train"].update(lr=0.01))
        with self.assertRaisesRegex(ValueError, "recipe changes"):
            self.prepare()

    def test_unknown_destination_is_not_overwritten(self) -> None:
        self.run.mkdir(parents=True)
        marker = self.run / "user_file"
        marker.write_text("preserve", encoding="utf-8")
        with self.assertRaises(FileNotFoundError):
            self.prepare()
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")

    def test_nonfinite_flag_or_loss_blocks_copy(self) -> None:
        flag = self.source / "nonfinite_diagnostic.json"
        flag.write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nonfinite diagnostic"):
            self.prepare()
        flag.unlink()
        path = self.source / "loss_components.csv"
        path.write_text(path.read_text(encoding="utf-8").replace("2.1", "nan", 1), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.prepare()

    def test_mixed_generation_staging_is_not_published(self) -> None:
        original_copytree = continuation.shutil.copytree

        def copy_with_changed_csv(source, destination, **kwargs):
            result = original_copytree(source, destination, **kwargs)
            path = Path(destination) / "epochs.csv"
            text = path.read_text(encoding="utf-8")
            path.write_text(text + text.splitlines(keepends=True)[-1], encoding="utf-8")
            return result

        with mock.patch.object(continuation.shutil, "copytree", side_effect=copy_with_changed_csv):
            with redirect_stderr(io.StringIO()), self.assertRaisesRegex(ValueError, "no automatic trimming"):
                self.prepare()
        self.assertFalse(self.run.exists())
        self.assertEqual(len(list(self.output_root.glob(".*.preparing-*"))), 1)
        self.assertEqual(self.inspect(self.source, source=True).epoch, 39)

    def test_already_exited_child_does_not_abort_signal_cleanup(self) -> None:
        process = SimpleNamespace(pid=12345)
        with mock.patch.object(continuation.os, "killpg", side_effect=ProcessLookupError):
            continuation._signal_group(process, continuation.signal.SIGTERM)

    def test_full110_specs_match_intended_three_configs(self) -> None:
        for spec, objective, weight in zip(
            continuation.RUNS, ("mep_baseline", "ours_fd", "ours_fd"), (0, 50, 200)
        ):
            config = load_config(continuation.ROOT / "configs/train" / f"{spec.source_name}.yaml")
            self.assertEqual(config["train"]["objective"], objective)
            self.assertEqual(config["train"]["feature_weight"], weight)
            self.assertEqual(config["train"]["mep_logit_weight"], 10)
            self.assertEqual(config["train"]["alpha"], 8)
            self.assertEqual(config["train"]["epsilon"], 32)

    def test_launcher_uses_approved_gpus_and_resume_without_evaluation(self) -> None:
        self.output_root.mkdir()
        args = SimpleNamespace(output_root=self.output_root, data_root=self.root / "data",
                               gpus=[1, 5, 6])
        children = [mock.Mock() for _ in continuation.RUNS]
        for process in children:
            process.poll.return_value = 0
            process.wait.return_value = 0
        with mock.patch.object(continuation.subprocess, "Popen", side_effect=children) as popen:
            with mock.patch.object(continuation, "inspect_run", return_value=continuation.RunState(109, 4, True)):
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(continuation.launch_runs(args, [continuation.RunState(39, 4, False)] * 3), 0)
        self.assertEqual(popen.call_count, 3)
        for call, spec, gpu in zip(popen.call_args_list, continuation.RUNS, (1, 5, 6)):
            command = call.args[0]
            self.assertEqual(command[command.index("co_blessing") + 1], "train")
            self.assertIn(str(self.output_root / spec.name / "resume.pt"), command)
            self.assertNotIn("evaluate", command)
            self.assertEqual(call.kwargs["env"]["CUDA_VISIBLE_DEVICES"], str(gpu))
            self.assertTrue(call.kwargs["start_new_session"])

    def test_completed_runs_do_not_launch_any_process(self) -> None:
        args = SimpleNamespace(gpus=[1, 5, 6])
        with mock.patch.object(continuation.subprocess, "Popen") as popen:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(continuation.launch_runs(args, [continuation.RunState(109, 4, True)] * 3), 0)
        popen.assert_not_called()

    def test_real_tensor_loader_rejects_nonfinite_checkpoint(self) -> None:
        try:
            import torch
        except ImportError:
            self.skipTest("PyTorch not installed in this local environment")
        path = self.root / "bad.pt"
        torch.save({"model": {"weight": torch.tensor([float("nan")])}}, path)
        with self.assertRaisesRegex(ValueError, "nonfinite tensor"):
            continuation.load_checkpoint(path)


if __name__ == "__main__":
    unittest.main()
