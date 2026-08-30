#!/usr/bin/env python3
"""Continue three trusted, completed 40-epoch pilots without changing their recipe."""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable

import yaml

from co_blessing.config import load_config


ROOT = Path(__file__).resolve().parent
TARGET_EPOCHS = 110
BATCHES_PER_EPOCH = math.ceil(50000 / 128)


@dataclass(frozen=True)
class RunSpec:
    source_name: str
    name: str
    source_group: str


RUNS = (
    RunSpec("pilot_mep_eps32_alpha8_logit10", "full_mep_eps32_alpha8_logit10", "pilot"),
    RunSpec("pilot_fd_eps32_alpha8_fw50", "full_fd_eps32_alpha8_fw50", "highfd"),
    RunSpec("pilot_fd_eps32_alpha8_fw200", "full_fd_eps32_alpha8_fw200", "highfd"),
)


@dataclass(frozen=True)
class RunState:
    epoch: int
    best_epoch: int
    complete: bool


def scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    """Only location, device, run name and total duration may change."""
    result = copy.deepcopy(config)
    for key in list(result):
        if key.startswith("_") or key in {"name", "device", "output"}:
            result.pop(key)
    result["data"].pop("root", None)
    result["train"].pop("epochs", None)
    return result


def load_checkpoint(path: Path) -> dict[str, Any]:
    # Only load the user's own trusted local checkpoints. Never download/unpickle
    # arbitrary third-party files. Lazy import keeps the preparation tests light.
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)

    def check(value: Any, label: str) -> None:
        if isinstance(value, torch.Tensor) and (value.is_floating_point() or value.is_complex()):
            # MEP buffers are ~600 MB each; bound temporary isfinite allocations.
            for chunk in value.detach().reshape(-1).split(1_000_000):
                if not torch.isfinite(chunk).all().item():
                    raise ValueError(f"{path}: nonfinite tensor {label}; refusing to resume")
        elif isinstance(value, dict):
            for key, child in value.items():
                check(child, f"{label}.{key}")
        elif isinstance(value, (list, tuple)):
            for index, child in enumerate(value):
                check(child, f"{label}[{index}]")

    check(checkpoint, "checkpoint")
    return checkpoint


def _same_config(config: dict[str, Any], expected: dict[str, Any], label: str) -> None:
    if scientific_config(config) != scientific_config(expected):
        raise ValueError(f"{label}: configuration differs from the pilot; refusing recipe changes")


def _read_rows(path: Path, epoch: int, fields: tuple[str, ...]) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if [int(row["epoch"]) for row in rows] != list(range(epoch + 1)):
        raise ValueError(
            f"{path}: expected exactly epochs 0..{epoch} to match resume.pt. "
            "Logs may be ahead of the checkpoint after interruption; no automatic trimming is done."
        )
    for row in rows:
        for field in fields:
            if not math.isfinite(float(row[field])):
                raise ValueError(f"{path}: nonfinite {field} at epoch {row['epoch']}")
    return rows


def _accuracy(value: Any, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"{label}: invalid accuracy {value!r}")
    return result


def inspect_run(
    run: Path,
    expected: dict[str, Any],
    *,
    source: bool = False,
    checkpoint_loader: Callable[[Path], dict[str, Any]] = load_checkpoint,
) -> RunState:
    if (run / "nonfinite_diagnostic.json").exists():
        raise ValueError(f"{run}: nonfinite diagnostic exists; inspect it before continuing")
    _same_config(load_config(run / "config.yaml"), expected, str(run / "config.yaml"))
    saved = checkpoint_loader(run / "resume.pt")
    _same_config(saved["config"], expected, str(run / "resume.pt"))
    epoch = int(saved["epoch"])
    if (source and epoch != 39) or not 39 <= epoch < TARGET_EPOCHS:
        raise ValueError(f"{run}: expected {'source epoch 39' if source else 'epoch 39..109'}, got {epoch}")
    if source and int(saved["config"]["train"]["epochs"]) != 40:
        raise ValueError(f"{run}: source is not a 40-epoch pilot")
    for key in ("model", "optimizer", "scheduler", "mep", "rng"):
        if not saved.get(key):
            raise ValueError(f"{run}/resume.pt: missing {key} state")
    if expected["data"]["num_workers"] != 0:
        raise ValueError("Exact continuation requires num_workers=0 (worker RNGs are not saved)")
    if not {"python", "numpy", "torch", "loader_generator", "cuda"} <= saved["rng"].keys():
        raise ValueError(f"{run}/resume.pt: missing RNG state")
    if len(saved["rng"]["cuda"]) != 1:
        raise ValueError(f"{run}: expected a single-visible-GPU pilot RNG state")

    scheduler = saved["scheduler"]
    steps = (epoch + 1) * BATCHES_PER_EPOCH
    milestones = {100 * BATCHES_PER_EPOCH: 1, 105 * BATCHES_PER_EPOCH: 1}
    actual = {int(key): value for key, value in scheduler["milestones"].items()}
    if int(scheduler["last_epoch"]) != steps or actual != milestones or scheduler["gamma"] != 0.1:
        raise ValueError(f"{run}: scheduler state does not match absolute milestones 100/105")
    expected_lr = 0.1 * 0.1 ** sum(steps >= milestone for milestone in milestones)
    for group in saved["optimizer"]["param_groups"]:
        if not math.isclose(float(group["lr"]), expected_lr, rel_tol=1e-8):
            raise ValueError(f"{run}: optimizer LR does not match the saved epoch")
    mep = saved["mep"]
    if (
        mep["sample_count"] != 50000
        or tuple(mep["image_shape"]) != (3, 32, 32)
        or not math.isclose(mep["epsilon"], 32 / 255)
        or not math.isclose(mep["alpha"], 8 / 255)
        or mep["last_reset_epoch"] != (epoch // 40) * 40
        or tuple(mep["delta"].shape) != (50000, 3, 32, 32)
        or tuple(mep["momentum"].shape) != (50000, 3, 32, 32)
    ):
        raise ValueError(f"{run}: MEP state does not match the selected pilot")

    rows = _read_rows(
        run / "epochs.csv", epoch,
        ("lr", "train_loss", "train_accuracy", "monitor_clean_accuracy", "monitor_pgd10_accuracy"),
    )
    _read_rows(
        run / "loss_components.csv", epoch,
        ("train_ce_loss", "train_logit_mse", "train_feature_mse"),
    )
    best_row = max(rows, key=lambda row: (float(row["monitor_pgd10_accuracy"]), int(row["epoch"])))
    best_epoch = int(best_row["epoch"])
    best_score = _accuracy(best_row["monitor_pgd10_accuracy"], str(run / "epochs.csv"))
    if not math.isclose(float(saved["best_pgd"]), best_score, abs_tol=1e-8):
        raise ValueError(f"{run}: resume.best_pgd disagrees with epochs.csv")
    del saved, mep, scheduler  # Do not retain the large MEP buffers for the next load.

    best = checkpoint_loader(run / "best.pt")
    _same_config(best["config"], expected, str(run / "best.pt"))
    if int(best["epoch"]) != best_epoch or not math.isclose(
        _accuracy(best["metrics"]["pgd10_accuracy"], "best PGD"), best_score, abs_tol=1e-8
    ):
        raise ValueError(f"{run}: best.pt disagrees with committed CSV/resume state")
    del best

    final_path = run / "final.pt"
    metrics_path = run / "final_metrics.json"
    if not source and epoch < TARGET_EPOCHS - 1 and (final_path.exists() or metrics_path.exists()):
        raise ValueError(f"{run}: stale final artifacts before epoch 109; refusing to call it complete")
    if final_path.exists():
        final = checkpoint_loader(final_path)
        _same_config(final["config"], expected, str(final_path))
        if int(final["epoch"]) != epoch:
            raise ValueError(f"{final_path}: epoch does not match resume.pt")
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            for key in ("clean_accuracy", "pgd10_accuracy"):
                value = _accuracy(metrics[key], str(metrics_path))
                if not math.isclose(value, float(final["metrics"][key]), abs_tol=1e-8):
                    raise ValueError(f"{metrics_path}: metrics disagree with final.pt")
        del final
    if source and not (final_path.exists() and metrics_path.exists()):
        raise ValueError(f"{run}: source pilot must have completed final artifacts")
    return RunState(epoch, best_epoch, epoch == 109 and final_path.exists() and metrics_path.exists())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_run(
    spec: RunSpec,
    source_root: Path,
    output_root: Path,
    data_root: Path,
    *,
    checkpoint_loader: Callable[[Path], dict[str, Any]] = load_checkpoint,
) -> RunState:
    source = (source_root / spec.source_name).resolve()
    run = (output_root / spec.name).resolve()
    if source == run or source in run.parents or run in source.parents:
        raise ValueError("Source and destination run directories must not overlap")
    expected = load_config(ROOT / "configs" / "train" / f"{spec.source_name}.yaml")
    if run.exists():
        provenance = json.loads((run / "continuation.json").read_text(encoding="utf-8"))
        if provenance["source"] != str(source) or provenance["target_epochs"] != TARGET_EPOCHS:
            raise ValueError(f"{run}: belongs to a different continuation; use a new output directory")
        config = load_config(run / "continuation_config.yaml")
        _same_config(config, expected, str(run / "continuation_config.yaml"))
        if config["name"] != spec.name or int(config["train"]["epochs"]) != TARGET_EPOCHS:
            raise ValueError(f"{run}: invalid continuation name/epoch count")
        return inspect_run(run, expected, checkpoint_loader=checkpoint_loader)

    inspect_run(source, expected, source=True, checkpoint_loader=checkpoint_loader)
    config = load_config(source / "config.yaml")
    config = {key: value for key, value in config.items() if not key.startswith("_")}
    config["name"] = spec.name
    config["train"]["epochs"] = TARGET_EPOCHS
    config["data"]["root"] = str(data_root)
    config["output"]["root"] = str(output_root)
    config["device"] = "cuda:0"
    fingerprints = {name: _sha256(source / name) for name in ("resume.pt", "best.pt", "config.yaml")}
    output_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{spec.name}.preparing-", dir=output_root))
    try:
        shutil.copytree(source, staging, dirs_exist_ok=True)
        archive = staging / "source_pilot"
        archive.mkdir()  # A pilot that is already a continuation is not an accepted source.
        for name in ("config.yaml", "environment.json", "final.pt", "final_metrics.json"):
            if name.startswith("final"):
                (staging / name).rename(archive / name)
            else:
                shutil.copy2(staging / name, archive / name)
        for name in fingerprints:
            if _sha256(staging / name) != fingerprints[name]:
                raise ValueError(f"{source / name}: changed while copying; refusing this snapshot")
        (staging / "continuation_config.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
        )
        (staging / "continuation.json").write_text(
            json.dumps(
                {"source": str(source), "source_epoch": 39, "target_epochs": TARGET_EPOCHS,
                 "source_sha256": fingerprints, "created_at": datetime.now(timezone.utc).isoformat(),
                 "note": "Exploratory alpha=8 extension; test-selected matched-epsilon monitor. "
                         "Preserved best.pt may retain its original pilot name/config."},
                indent=2,
            ) + "\n", encoding="utf-8",
        )
        # Validate the copied generation, not just the pre-copy source. A source
        # accidentally resumed during copying must not publish mixed CSV/state.
        state = inspect_run(staging, expected, checkpoint_loader=checkpoint_loader)
        if run.exists():
            raise FileExistsError(f"Destination appeared while preparing: {run}")
        staging.rename(run)
    except BaseException:
        print(f"Preparation did not finish. Original pilot is untouched; staging retained: {staging}",
              file=sys.stderr, flush=True)
        raise
    return state


def _signal_group(process: subprocess.Popen, signum: int) -> None:
    try:
        os.killpg(process.pid, signum)
    except ProcessLookupError:
        pass  # It exited after poll(); still clean up the other owned children.


def launch_runs(args: argparse.Namespace, states: list[RunState]) -> int:
    pending = [(spec, gpu) for spec, gpu, state in zip(RUNS, args.gpus, states) if not state.complete]
    if not pending:
        print("All three 110-epoch runs are already complete; nothing launched.", flush=True)
        return 0
    logs = args.output_root / "logs"
    logs.mkdir(exist_ok=True)
    processes: list[tuple[RunSpec, subprocess.Popen]] = []
    try:
        for spec, gpu in pending:
            run = args.output_root / spec.name
            log = logs / f"{spec.name}.log"
            command = [sys.executable, "-u", "-m", "co_blessing", "train", "--config",
                       str(run / "continuation_config.yaml"), "--resume", str(run / "resume.pt"),
                       "--data-root", str(args.data_root), "--output-root", str(args.output_root),
                       "--device", "cuda:0"]
            print(f"Launch {spec.name} on physical GPU {gpu}; log={log}", flush=True)
            with log.open("a", encoding="utf-8") as handle:
                handle.write(f"\nContinuation launch {datetime.now(timezone.utc).isoformat()} GPU={gpu}\n")
                handle.flush()
                process = subprocess.Popen(
                    command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                    stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                )
            processes.append((spec, process))
        status = 0
        remaining = list(processes)
        while remaining:
            for spec, process in remaining[:]:
                code = process.poll()
                if code is not None:
                    print(f"Finished {spec.name}: exit={code}", flush=True)
                    status |= int(code != 0)
                    remaining.remove((spec, process))
            if remaining:
                time.sleep(0.5)
        if status:
            print("A run failed. Inspect its log/diagnostic; no automatic retry or evaluation.",
                  file=sys.stderr)
            return status
    finally:
        # Only our own child process groups are stopped if this launcher is interrupted.
        for _, process in processes:
            if process.poll() is None:
                _signal_group(process, signal.SIGTERM)
        for _, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _signal_group(process, signal.SIGKILL)
                process.wait()
    for spec in RUNS:
        print(f"Verify completed run: {spec.name}", flush=True)
        expected = load_config(ROOT / "configs" / "train" / f"{spec.source_name}.yaml")
        if not inspect_run(args.output_root / spec.name, expected).complete:
            raise ValueError(f"{spec.name}: process exited without a complete epoch-109 checkpoint")
    print("All three runs completed 110 epochs. No AutoAttack/evaluation was launched.", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/cjk/cifar-data"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("/data/cjk/FGSM-MEP-cifar10-eps32-alpha8-full110"))
    parser.add_argument("--pilot-root", type=Path,
                        default=Path("/data/cjk/FGSM-MEP-cifar10-eps32-alpha8-pilots"))
    parser.add_argument("--highfd-root", type=Path,
                        default=Path("/data/cjk/FGSM-MEP-cifar10-eps32-alpha8-highfd"))
    parser.add_argument("--gpus", type=int, nargs=3, default=[1, 5, 6], metavar=("MEP", "FD50", "FD200"))
    parser.add_argument("--prepare-only", action="store_true", help="Validate/copy runs without using GPUs")
    args = parser.parse_args(argv)
    if len(set(args.gpus)) != 3 or min(args.gpus) < 0:
        parser.error("--gpus requires three distinct nonnegative physical GPU IDs")
    for key in ("data_root", "output_root", "pilot_root", "highfd_root"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    for source_root in (args.pilot_root, args.highfd_root):
        if (args.output_root == source_root or args.output_root in source_root.parents
                or source_root in args.output_root.parents):
            parser.error("Use a separate output root outside both original pilot roots")
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root / ".continuation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another continuation launcher holds this output directory")
        states = []
        for spec in RUNS:
            print(f"Check/prepare {spec.name} (CPU checkpoint validation and file copy)", flush=True)
            source_root = args.pilot_root if spec.source_group == "pilot" else args.highfd_root
            state = prepare_run(spec, source_root, args.output_root, args.data_root)
            states.append(state)
            print(f"  epoch={state.epoch}, best={state.best_epoch}, "
                  f"{'COMPLETE: skip' if state.complete else 'READY: resume to 110 epochs'}", flush=True)
        if args.prepare_only:
            print("Preparation complete; no GPU processes launched.", flush=True)
            return 0
        return launch_runs(args, states)


if __name__ == "__main__":
    def _interrupted(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _interrupted)
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted; original pilots remain untouched. Re-run to check/resume saved epochs.",
              file=sys.stderr)
        raise SystemExit(130)
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Cannot continue safely: {error}", file=sys.stderr)
        raise SystemExit(1)
