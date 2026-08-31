#!/usr/bin/env python3
"""Screen alpha/LR pairs for finite high-epsilon Ours-FD training prefixes."""
from __future__ import annotations

import argparse
import copy
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
EPSILONS = (32, 48, 64)
ALPHA_RATIOS = (1 / 8, 1 / 4, 1 / 2)
LEARNING_RATES = (0.01, 0.03, 0.1)


def _number_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def candidates() -> list[tuple[int, int, float]]:
    """Return 27 reproducible (epsilon, alpha, learning-rate) candidates."""
    result = []
    for epsilon in EPSILONS:
        for ratio in ALPHA_RATIOS:
            alpha = round(epsilon * ratio)
            for lr in LEARNING_RATES:
                result.append((epsilon, alpha, lr))
    return result


def run_name(epsilon: int, alpha: int, lr: float) -> str:
    return f"stability_ours_fd_eps{epsilon}_alpha{alpha}_lr{_number_token(lr)}"


def build_config(
    *, epsilon: int, alpha: int, lr: float, data_root: Path, output_root: Path
) -> dict[str, Any]:
    """Freeze all baseline fields except the documented diagnostic controls."""
    config = copy.deepcopy(load_config(ROOT / "configs" / "train" / f"ours_fd_eps{epsilon}.yaml"))
    config.pop("_config_path", None)
    config["name"] = run_name(epsilon, alpha, lr)
    config["deterministic"] = True
    config["device"] = "cuda:0"
    config["data"]["root"] = str(data_root)
    config["output"]["root"] = str(output_root)
    config["train"].update(
        {
            "epochs": 1,
            "epsilon": epsilon,
            "alpha": alpha,
            "lr": lr,
            "monitor_pgd_steps": 10,
            "monitor_pgd_step_size": epsilon / 4,
            "track_features": False,
            "abort_on_nonfinite": True,
        }
    )
    validate_config(config)
    return config


def _write_config(path: Path, config: dict[str, Any]) -> None:
    serialized = yaml.safe_dump(config, sort_keys=False)
    if path.exists() and path.read_text(encoding="utf-8") != serialized:
        raise ValueError(f"Existing grid config differs: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")


def _write_manifest(path: Path, jobs: list[dict[str, Any]]) -> None:
    payload = {
        "purpose": "diagnostic stability screen; not the frozen formal baseline",
        "axes": {"epsilon": list(EPSILONS), "alpha_ratios": list(ALPHA_RATIOS), "lr": list(LEARNING_RATES)},
        "fixed": {"objective": "ours_fd", "backend": "mep", "mep_logit_weight": 10,
                  "feature_node": "B", "feature_weight": 200, "seed": 0,
                  "deterministic": True, "epochs": 1, "abort_on_nonfinite": True},
        "jobs": jobs,
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _recorded(run_dir: Path) -> bool:
    return (run_dir / "final.pt").exists() or (run_dir / "nonfinite_diagnostic.json").exists()


def _stop_active(active: dict[int, tuple[dict[str, Any], subprocess.Popen[Any]]]) -> None:
    """Terminate only process groups launched by this scheduler."""
    for _, (_, process) in active.items():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/cjk/cifar-data"))
    parser.add_argument("--output-root", type=Path,
                        default=Path("/data/cjk/FGSM-MEP-cifar10-high-eps-stability-grid"))
    parser.add_argument("--gpus", type=int, nargs=3, default=[5, 6, 7], metavar=("GPU_A", "GPU_B", "GPU_C"))
    args = parser.parse_args(argv)
    if len(set(args.gpus)) != 3 or min(args.gpus) < 0:
        parser.error("--gpus requires three distinct nonnegative physical GPU IDs")
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    with (args.output_root / ".grid.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another stability-grid launcher holds this output directory")

        jobs: list[dict[str, Any]] = []
        for epsilon, alpha, lr in candidates():
            name = run_name(epsilon, alpha, lr)
            config = build_config(
                epsilon=epsilon, alpha=alpha, lr=lr,
                data_root=args.data_root, output_root=args.output_root,
            )
            config_path = args.output_root / "grid_configs" / f"{name}.yaml"
            _write_config(config_path, config)
            jobs.append({"name": name, "epsilon": epsilon, "alpha": alpha, "lr": lr,
                         "config": str(config_path), "run_dir": str(args.output_root / name)})
        _write_manifest(args.output_root / "grid_manifest.json", jobs)

        logs = args.output_root / "logs"
        logs.mkdir(exist_ok=True)
        pending = [job for job in jobs if not _recorded(Path(job["run_dir"]))]
        skipped = len(jobs) - len(pending)
        print(f"{len(jobs)} candidates prepared; {skipped} already recorded; {len(pending)} to run.", flush=True)

        active: dict[int, tuple[dict[str, Any], subprocess.Popen[Any]]] = {}
        failures: list[str] = []
        try:
            while pending or active:
                for gpu in args.gpus:
                    if gpu in active or not pending:
                        continue
                    job = pending.pop(0)
                    log_path = logs / f"{job['name']}.log"
                    command = [sys.executable, "-u", "-m", "co_blessing", "train", "--config", job["config"],
                               "--data-root", str(args.data_root), "--output-root", str(args.output_root),
                               "--device", "cuda:0"]
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
                    run_dir = Path(job["run_dir"])
                    expected = _recorded(run_dir)
                    outcome = "recorded" if expected else f"FAILED exit={status} without a diagnostic/final artifact"
                    print(f"finish {job['name']} on GPU {gpu}: {outcome}", flush=True)
                    if not expected:
                        failures.append(job["name"])
                    del active[gpu]
                if active:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            _stop_active(active)
            print("Interrupted: sent SIGTERM to active grid jobs.", file=sys.stderr)
            return 130

    if failures:
        print("Unexpected grid failures: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("Stability grid finished. Summarize finite prefixes before any multi-epoch tuning.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
