#!/usr/bin/env python3
"""Run the frozen, screened Ours-FD candidates to their final 110th epoch.

The candidate selection was made by the independent 40-epoch stability
screen.  This launcher writes its exact merged configurations and a manifest
into the output directory, trains only those configurations, and uses
``final.pt`` as the sole formal checkpoint.  It never resumes or evaluates a
recorded non-finite run automatically.
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
SEEDS = (0, 1, 2)


@dataclass(frozen=True)
class SelectedSpec:
    epsilon: int
    alpha: int
    feature_weight: float
    lr: float
    rationale: str


# Frozen after the 40-epoch, three-seed stability screen.  The epsilon=64
# candidate is explicitly a finite degraded/CO-like regime: no learned
# candidate was both non-random and free of early PGD degradation.
SELECTED_SPECS = (
    SelectedSpec(8, 8, 1.0, 0.03, "highest stable final monitor PGD-10"),
    SelectedSpec(12, 12, 5.0, 0.01, "highest stable final monitor PGD-10"),
    SelectedSpec(16, 16, 10.0, 0.01, "highest stable final monitor PGD-10"),
    SelectedSpec(32, 8, 25.0, 0.01, "highest stable final monitor PGD-10"),
    SelectedSpec(48, 12, 5.0, 0.01, "highest stable final monitor PGD-10"),
    SelectedSpec(
        64,
        16,
        10.0,
        0.01,
        "best final PGD-10 among non-random finite candidates; early degradation retained",
    ),
)


def _token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def run_name(spec: SelectedSpec, seed: int) -> str:
    return (
        f"aaer_selected_ours_fd_cifar10_eps{spec.epsilon}_alpha{spec.alpha}"
        f"_fw{_token(spec.feature_weight)}_lr{_token(spec.lr)}_seed{seed}"
    )


def build_config(
    spec: SelectedSpec,
    seed: int,
    *,
    data_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build a method-faithful final-training config for one frozen candidate."""

    config = copy.deepcopy(load_config(ROOT / "configs" / "aaer" / "ours_fd_cifar10_base.yaml"))
    config.pop("_config_path", None)
    config["name"] = run_name(spec, seed)
    config["seed"] = seed
    config["deterministic"] = False
    config["device"] = "cuda:0"
    config["data"].update({"root": str(data_root), "train_subset": None, "test_subset": None})
    config["output"]["root"] = str(output_root)
    config["train"].update(
        {
            "epochs": 110,
            "epsilon": spec.epsilon,
            "alpha": spec.alpha,
            "lr": spec.lr,
            "feature_weight": spec.feature_weight,
            # This monitor is diagnostic-only and never selects the formal
            # checkpoint.  Keeping it small prevents monitoring from
            # dominating the 110-epoch training cost.
            "monitor_pgd_steps": 10,
            "monitor_pgd_step_size": spec.epsilon / 4,
            "monitor_subset": 1000,
            "track_features": False,
            "abort_on_nonfinite": True,
            "save_resume": True,
        }
    )
    validate_config(config)
    return config


def _write_exact(path: Path, text: str) -> None:
    if path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Existing generated file differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _outcome(run_dir: Path) -> str:
    if (run_dir / "final.pt").is_file():
        return "complete"
    if (run_dir / "nonfinite_diagnostic.json").is_file():
        return "nonfinite"
    return "pending"


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
        "--output-root",
        type=Path,
        default=Path("/data/cjk/FGSM-MEP-aaer-ours-fd-cifar10-selected"),
    )
    parser.add_argument(
        "--screen-root",
        type=Path,
        default=Path("/data/cjk/FGSM-MEP-aaer-ours-fd-cifar10-stability-screen"),
        help="Recorded only as selection provenance; it is never used for resume state.",
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)), metavar="GPU")
    args = parser.parse_args(argv)
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("--gpus requires one or more distinct nonnegative physical GPU IDs")
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.screen_root = args.screen_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    with (args.output_root / ".selected-training.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another selected Ours-FD training run holds this output directory")

        jobs: list[dict[str, Any]] = []
        for spec in SELECTED_SPECS:
            for seed in SEEDS:
                name = run_name(spec, seed)
                config_path = args.output_root / "selected_configs" / f"{name}.yaml"
                _write_exact(
                    config_path,
                    yaml.safe_dump(
                        build_config(spec, seed, data_root=args.data_root, output_root=args.output_root),
                        sort_keys=False,
                    ),
                )
                jobs.append(
                    {
                        **asdict(spec),
                        "seed": seed,
                        "name": name,
                        "config": str(config_path),
                        "run_dir": str(args.output_root / name),
                    }
                )
        _write_exact(
            args.output_root / "selected_manifest.json",
            json.dumps(
                {
                    "purpose": "AAER Table-2 Ours-FD final-checkpoint runs after stability screening",
                    "selection_source": str(args.screen_root),
                    "selection_protocol": "40 epochs; three seeds; finite trajectory and final matched-epsilon PGD-10 monitor",
                    "formal_checkpoint": "final.pt at epoch 109; never best.pt",
                    "training_schedule": "original Ours-FD 110 epochs, milestones 100/105, MEP reset every 40 epochs",
                    "evaluation_protocol": "matched-epsilon AAER PGD-50, epsilon/4 step, 10 restarts",
                    "selected_specs": [asdict(spec) for spec in SELECTED_SPECS],
                    "jobs": jobs,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

        logs = args.output_root / "logs"
        logs.mkdir(exist_ok=True)
        nonfinite = [job for job in jobs if _outcome(Path(job["run_dir"])) == "nonfinite"]
        pending = [job for job in jobs if _outcome(Path(job["run_dir"])) == "pending"]
        complete = len(jobs) - len(pending) - len(nonfinite)
        print(
            f"{len(jobs)} frozen final-training jobs; {complete} completed; "
            f"{len(nonfinite)} nonfinite (not retried); {len(pending)} pending.",
            flush=True,
        )

        active: dict[int, tuple[dict[str, Any], subprocess.Popen[Any]]] = {}
        failed_after_launch: list[str] = []
        try:
            while pending or active:
                for gpu in args.gpus:
                    if gpu in active or not pending:
                        continue
                    job = pending.pop(0)
                    run_dir = Path(job["run_dir"])
                    resume = run_dir / "resume.pt"
                    command = [
                        sys.executable,
                        "-u",
                        "-m",
                        "co_blessing",
                        "train",
                        "--config",
                        str(job["config"]),
                        "--data-root",
                        str(args.data_root),
                        "--output-root",
                        str(args.output_root),
                        "--device",
                        "cuda:0",
                    ]
                    if resume.is_file():
                        command += ["--resume", str(resume)]
                    log_path = logs / f"{job['name']}.log"
                    print(f"launch {job['name']} on physical GPU {gpu}; log={log_path}", flush=True)
                    with log_path.open("a", encoding="utf-8") as handle:
                        process = subprocess.Popen(
                            command,
                            cwd=ROOT,
                            env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    active[gpu] = (job, process)

                for gpu, (job, process) in list(active.items()):
                    exit_code = process.poll()
                    if exit_code is None:
                        continue
                    outcome = _outcome(Path(job["run_dir"]))
                    print(f"finish {job['name']} on GPU {gpu}: {outcome}", flush=True)
                    if outcome != "complete":
                        failed_after_launch.append(str(job["name"]))
                    del active[gpu]
                if active:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            _stop_active(active)
            print("Interrupted: sent SIGTERM to active final-training jobs.", file=sys.stderr)
            return 130

    if nonfinite or failed_after_launch:
        print(
            "Non-complete selected runs: "
            + ", ".join([str(job["name"]) for job in nonfinite] + failed_after_launch),
            file=sys.stderr,
        )
        return 1
    print("All selected Ours-FD 110-epoch runs completed; launch evaluation separately.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
