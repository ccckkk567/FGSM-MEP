#!/usr/bin/env python3
"""Screen finite Ours-FD settings on AAER's CIFAR-10 PreActResNet-18 protocol.

This is deliberately a configuration screen, not a result-table run.  It uses
the AAER-aligned data/model/PGD monitor but omits large MEP resume checkpoints:
completed candidates retain final.pt and CSVs, while an interrupted candidate
is restarted from epoch 0 on the next invocation.
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
SEEDS = (0, 1, 2)
FEATURE_WEIGHTS = (1.0, 5.0, 10.0, 25.0)
LEARNING_RATES = (0.01, 0.03)
# The low-epsilon entries retain Ours-FD's alpha=epsilon convention.  The high
# entries are the finite alpha candidates found in the earlier raw-input audit;
# this screen verifies them afresh under AAER's PreAct/normalization protocol.
EPSILON_ALPHAS = ((8, 8), (12, 12), (16, 16), (32, 8), (48, 12), (64, 16))


@dataclass(frozen=True)
class ScreenSpec:
    epsilon: int
    alpha: int
    feature_weight: float
    lr: float
    seed: int


def _token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def candidates() -> list[ScreenSpec]:
    return [
        ScreenSpec(epsilon, alpha, feature_weight, lr, seed)
        for epsilon, alpha in EPSILON_ALPHAS
        for feature_weight in FEATURE_WEIGHTS
        for lr in LEARNING_RATES
        for seed in SEEDS
    ]


def run_name(spec: ScreenSpec) -> str:
    return (
        f"aaer_screen_ours_fd_cifar10_eps{spec.epsilon}_alpha{spec.alpha}"
        f"_fw{_token(spec.feature_weight)}_lr{_token(spec.lr)}_seed{spec.seed}"
    )


def build_config(
    spec: ScreenSpec, *, data_root: Path, output_root: Path
) -> dict[str, Any]:
    config = copy.deepcopy(load_config(ROOT / "configs" / "aaer" / "ours_fd_cifar10_base.yaml"))
    config.pop("_config_path", None)
    config["name"] = run_name(spec)
    config["seed"] = spec.seed
    config["deterministic"] = False
    config["device"] = "cuda:0"
    config["data"].update({"root": str(data_root), "train_subset": None, "test_subset": None})
    config["output"]["root"] = str(output_root)
    config["train"].update(
        {
            "epochs": SCREEN_EPOCHS,
            "epsilon": spec.epsilon,
            "alpha": spec.alpha,
            "lr": spec.lr,
            "feature_weight": spec.feature_weight,
            # The monitor is diagnostic only.  Reducing it to a deterministic
            # 1,000-example prefix makes a 144-run multi-seed screen practical.
            "monitor_pgd_steps": 10,
            "monitor_pgd_step_size": spec.epsilon / 4,
            "monitor_subset": 1000,
            "track_features": False,
            "abort_on_nonfinite": True,
            "save_resume": False,
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
    return (run_dir / "final.pt").is_file() or (run_dir / "nonfinite_diagnostic.json").is_file()


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
        default=Path("/data/cjk/FGSM-MEP-aaer-ours-fd-cifar10-stability-screen"),
    )
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)), metavar="GPU")
    args = parser.parse_args(argv)
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("--gpus requires one or more distinct nonnegative physical GPU IDs")
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    with (args.output_root / ".stability.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another AAER Ours-FD stability screen holds this output directory")

        jobs: list[dict[str, Any]] = []
        for spec in candidates():
            name = run_name(spec)
            config_path = args.output_root / "screen_configs" / f"{name}.yaml"
            _write_exact(
                config_path,
                yaml.safe_dump(
                    build_config(spec, data_root=args.data_root, output_root=args.output_root),
                    sort_keys=False,
                ),
            )
            jobs.append(
                {
                    **asdict(spec),
                    "name": name,
                    "config": str(config_path),
                    "run_dir": str(args.output_root / name),
                }
            )
        _write_exact(
            args.output_root / "screen_manifest.json",
            json.dumps(
                {
                    "purpose": "AAER-PreAct Ours-FD 40-epoch multi-seed stability screen; not a formal result",
                    "screen_epochs": SCREEN_EPOCHS,
                    "epsilons_alpha": [list(item) for item in EPSILON_ALPHAS],
                    "feature_weights": list(FEATURE_WEIGHTS),
                    "learning_rates": list(LEARNING_RATES),
                    "seeds": list(SEEDS),
                    "monitor": "PGD-10 on the first 1,000 test samples; epsilon matched, step epsilon/4",
                    "fixed": {
                        "objective": "ours_fd",
                        "backend": "mep",
                        "mep_logit_weight": 10.0,
                        "fd_include_mep_logit": True,
                        "feature_node": "B",
                        "label_true_probability": 0.6,
                        "epochs": SCREEN_EPOCHS,
                        "abort_on_nonfinite": True,
                        "save_resume": False,
                    },
                    "jobs": jobs,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )

        logs = args.output_root / "logs"
        logs.mkdir(exist_ok=True)
        pending = [job for job in jobs if not _recorded(Path(job["run_dir"]))]
        print(
            f"{len(jobs)} candidates prepared; {len(jobs) - len(pending)} already recorded; "
            f"{len(pending)} to run across {len(args.gpus)} GPUs.",
            flush=True,
        )
        active: dict[int, tuple[dict[str, Any], subprocess.Popen[Any]]] = {}
        unexpected: list[str] = []
        try:
            while pending or active:
                for gpu in args.gpus:
                    if gpu in active or not pending:
                        continue
                    job = pending.pop(0)
                    log_path = logs / f"{job['name']}.log"
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
                    recorded = _recorded(Path(job["run_dir"]))
                    outcome = "recorded" if recorded else f"UNEXPECTED exit={exit_code}"
                    print(f"finish {job['name']} on GPU {gpu}: {outcome}", flush=True)
                    if not recorded:
                        unexpected.append(str(job["name"]))
                    del active[gpu]
                if active:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            _stop_active(active)
            print("Interrupted: sent SIGTERM to active screen jobs.", file=sys.stderr)
            return 130

    if unexpected:
        print("Unexpected screen failures: " + ", ".join(unexpected), file=sys.stderr)
        return 1
    print("AAER Ours-FD stability screen completed; summarize before launching 110-epoch runs.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
