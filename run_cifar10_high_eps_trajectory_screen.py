#!/usr/bin/env python3
"""Run 40-epoch high-epsilon Ours-FD trajectories selected by a finite-prefix screen.

This is deliberately a diagnostic track.  The frozen original Ours-FD baseline remains
N/A at epsilon 32/48/64 when its native alpha=epsilon, lr=0.1 recipe diverges.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, dataclass
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any

import yaml

from co_blessing.config import load_config, validate_config


ROOT = Path(__file__).resolve().parent
SCREEN_EPOCHS = 40


@dataclass(frozen=True)
class ScreenSpec:
    epsilon: int
    alpha: int
    lr: float
    rationale: str


# One learning-oriented and one larger-step candidate per epsilon.  All six were
# finite in the deterministic one-epoch screen; the latter candidate is closer to
# the finite/nonfinite boundary and is useful for observing whether CO occurs.
SPECS = (
    ScreenSpec(32, 4, 0.10, "learning-oriented"),
    ScreenSpec(32, 8, 0.10, "larger finite step"),
    ScreenSpec(48, 6, 0.10, "learning-oriented"),
    ScreenSpec(48, 12, 0.03, "larger finite step"),
    ScreenSpec(64, 8, 0.10, "learning-oriented"),
    ScreenSpec(64, 16, 0.03, "larger finite step"),
)


def _number_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def run_name(spec: ScreenSpec) -> str:
    return f"trajectory_ours_fd_eps{spec.epsilon}_alpha{spec.alpha}_lr{_number_token(spec.lr)}"


def build_config(
    spec: ScreenSpec, *, data_root: Path, output_root: Path
) -> dict[str, Any]:
    """Keep Ours-FD fixed except named diagnostic axes and duration."""
    config = copy.deepcopy(
        load_config(ROOT / "configs" / "train" / f"ours_fd_eps{spec.epsilon}.yaml")
    )
    config.pop("_config_path", None)
    config["name"] = run_name(spec)
    config["deterministic"] = True
    config["device"] = "cuda:0"
    config["data"]["root"] = str(data_root)
    config["output"]["root"] = str(output_root)
    config["train"].update(
        {
            "epochs": SCREEN_EPOCHS,
            "epsilon": spec.epsilon,
            "alpha": spec.alpha,
            "lr": spec.lr,
            "monitor_pgd_steps": 10,
            "monitor_pgd_step_size": spec.epsilon / 4,
            "track_features": True,
            "abort_on_nonfinite": True,
        }
    )
    validate_config(config)
    return config


def _write_exact(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Existing generated file differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _recorded(run_dir: Path) -> bool:
    return (run_dir / "final.pt").exists() or (run_dir / "nonfinite_diagnostic.json").exists()


def _stop_active(active: dict[int, tuple[dict[str, Any], subprocess.Popen[Any]]]) -> None:
    for _, (_, process) in active.items():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/cjk/cifar-data"))
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("/data/cjk/FGSM-MEP-cifar10-high-eps-trajectory-screen"),
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=[5, 6, 7], metavar="GPU")
    args = parser.parse_args(argv)
    if len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("--gpus requires distinct nonnegative physical GPU IDs")
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    with (args.output_root / ".trajectory.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another trajectory-screen launcher holds this output directory")

        jobs: list[dict[str, Any]] = []
        for spec in SPECS:
            name = run_name(spec)
            config_path = args.output_root / "trajectory_configs" / f"{name}.yaml"
            _write_exact(config_path, yaml.safe_dump(
                build_config(spec, data_root=args.data_root, output_root=args.output_root), sort_keys=False
            ))
            jobs.append({
                **asdict(spec), "name": name, "config": str(config_path),
                "run_dir": str(args.output_root / name),
            })
        _write_exact(
            args.output_root / "trajectory_manifest.json",
            json.dumps(
                {
                    "purpose": "diagnostic CO trajectory screen; not frozen formal baseline",
                    "screen_epochs": SCREEN_EPOCHS,
                    "fixed": {
                        "objective": "ours_fd", "backend": "mep", "mep_logit_weight": 10,
                        "feature_node": "B", "feature_weight": 200, "seed": 0,
                        "deterministic": True, "abort_on_nonfinite": True,
                    },
                    "jobs": jobs,
                }, indent=2, sort_keys=True,
            ) + "\n",
        )

        logs = args.output_root / "logs"
        logs.mkdir(exist_ok=True)
        pending = [job for job in jobs if not _recorded(Path(job["run_dir"]))]
        print(f"{len(jobs)} trajectories prepared; {len(jobs) - len(pending)} already recorded; {len(pending)} to run.", flush=True)
        active: dict[int, tuple[dict[str, Any], subprocess.Popen[Any]]] = {}
        failures: list[str] = []
        try:
            while pending or active:
                for gpu in args.gpus:
                    if gpu in active or not pending:
                        continue
                    job = pending.pop(0)
                    log_path = logs / f"{job['name']}.log"
                    command = [
                        sys.executable, "-u", "-m", "co_blessing", "train", "--config", job["config"],
                        "--data-root", str(args.data_root), "--output-root", str(args.output_root),
                        "--device", "cuda:0",
                    ]
                    print(f"launch {job['name']} on physical GPU {gpu}; log={log_path}", flush=True)
                    with log_path.open("a", encoding="utf-8") as handle:
                        process = subprocess.Popen(
                            command, cwd=ROOT, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                            stdout=handle, stderr=subprocess.STDOUT, start_new_session=True,
                        )
                    active[gpu] = (job, process)

                for gpu, (job, process) in list(active.items()):
                    status = process.poll()
                    if status is None:
                        continue
                    expected = _recorded(Path(job["run_dir"]))
                    outcome = "recorded" if expected else f"FAILED exit={status} without a diagnostic/final artifact"
                    print(f"finish {job['name']} on GPU {gpu}: {outcome}", flush=True)
                    if not expected:
                        failures.append(job["name"])
                    del active[gpu]
                if active:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            _stop_active(active)
            print("Interrupted: sent SIGTERM to active trajectory jobs.", file=sys.stderr)
            return 130

    if failures:
        print("Unexpected trajectory failures: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("Trajectory screen finished. Inspect the CO curves before any 110-epoch tuned run.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
