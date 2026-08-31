#!/usr/bin/env python3
"""Evaluate selected Ours-FD final checkpoints with AAER's PGD-50-10 protocol."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parent


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
    parser.add_argument("--gpus", type=int, nargs="+", default=list(range(8)), metavar="GPU")
    args = parser.parse_args(argv)
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or min(args.gpus) < 0:
        parser.error("--gpus requires one or more distinct nonnegative physical GPU IDs")
    args.data_root = args.data_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    manifest_path = args.output_root / "selected_manifest.json"
    if not manifest_path.is_file():
        parser.error(f"Missing selected training manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = list(manifest.get("jobs", []))
    if len(jobs) != 18:
        parser.error("Selected manifest must contain exactly 18 (six epsilon × three seed) jobs")

    missing = [job["name"] for job in jobs if not (Path(job["run_dir"]) / "final.pt").is_file()]
    if missing:
        parser.error("Missing final checkpoints; do not run formal evaluation: " + ", ".join(missing))

    with (args.output_root / ".selected-evaluation.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another selected Ours-FD evaluation holds this output directory")

        logs = args.output_root / "logs"
        logs.mkdir(exist_ok=True)
        pending = []
        for job in jobs:
            evaluation_name = f"eval_{job['name']}"
            result_path = args.output_root / evaluation_name / "evaluation.json"
            if result_path.is_file():
                print(f"skip completed evaluation {job['name']}", flush=True)
                continue
            pending.append({**job, "evaluation_name": evaluation_name, "result_path": str(result_path)})
        print(f"{len(jobs) - len(pending)} evaluations completed; {len(pending)} pending.", flush=True)

        active: dict[int, tuple[dict[str, Any], subprocess.Popen[Any]]] = {}
        unexpected: list[str] = []
        try:
            while pending or active:
                for gpu in args.gpus:
                    if gpu in active or not pending:
                        continue
                    job = pending.pop(0)
                    epsilon = int(job["epsilon"])
                    checkpoint = Path(job["run_dir"]) / "final.pt"
                    command = [
                        sys.executable,
                        "-u",
                        "-m",
                        "co_blessing",
                        "evaluate",
                        "--config",
                        str(ROOT / "configs" / "eval" / f"aaer_cifar10_eps{epsilon}.yaml"),
                        "--checkpoint",
                        str(checkpoint),
                        "--name",
                        str(job["evaluation_name"]),
                        "--data-root",
                        str(args.data_root),
                        "--output-root",
                        str(args.output_root),
                        "--device",
                        "cuda:0",
                    ]
                    log_path = logs / f"{job['evaluation_name']}.log"
                    print(f"evaluate {job['name']} on physical GPU {gpu}; log={log_path}", flush=True)
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
                    complete = Path(job["result_path"]).is_file()
                    print(f"finish evaluation {job['name']} on GPU {gpu}: {'complete' if complete else 'failed'}", flush=True)
                    if not complete:
                        unexpected.append(str(job["name"]))
                    del active[gpu]
                if active:
                    time.sleep(0.5)
        except KeyboardInterrupt:
            _stop_active(active)
            print("Interrupted: sent SIGTERM to active evaluation jobs.", file=sys.stderr)
            return 130

    if unexpected:
        print("Failed evaluations: " + ", ".join(unexpected), file=sys.stderr)
        return 1

    results = [str(args.output_root / f"eval_{job['name']}" / "evaluation.json") for job in jobs]
    summary_dir = args.output_root / "aaer_selected_ours_fd_table2"
    command = [
        sys.executable,
        "-m",
        "co_blessing",
        "aaer-summary",
        "--results",
        *results,
        "--output",
        str(summary_dir),
    ]
    subprocess.run(command, cwd=ROOT, check=True)
    print(f"AAER selected Ours-FD summary: {summary_dir / 'aaer_table2_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
