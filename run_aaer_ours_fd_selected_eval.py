#!/usr/bin/env python3
"""Evaluate selected Ours-FD checkpoints with AAER's PGD-50-10 protocol."""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parent
AAER_PROTOCOL = "aaer_official_pgd50_10"


def _mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def summarize_best_diagnostic(
    result_paths: list[Path], output_dir: Path, *, expected_epsilons: tuple[int, ...]
) -> tuple[Path, Path]:
    """Aggregate best.pt PGD-50 results without permitting Table-2 confusion."""

    groups: dict[int, dict[int, dict[str, float | int]]] = {}
    for result_path in result_paths:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        if payload.get("protocol") != AAER_PROTOCOL or payload.get("checkpoint_role") != "best":
            raise ValueError(f"{result_path}: expected an AAER best.pt evaluation")
        evaluation = payload.get("evaluation_config") or {}
        training = payload.get("training_config") or {}
        train = training.get("train") or {}
        epsilon = int(round(float(train["epsilon"])))
        if int(round(float(evaluation.get("epsilon", -1)))) != epsilon:
            raise ValueError(f"{result_path}: train/evaluation epsilon mismatch")
        if evaluation.get("attacks") != ["clean", "pgd50"] or int(evaluation.get("restarts", 0)) != 10:
            raise ValueError(f"{result_path}: not the AAER PGD-50-10 protocol")
        if abs(float(evaluation.get("step_size", 0.0)) - epsilon / 4.0) > 1e-12:
            raise ValueError(f"{result_path}: incorrect AAER PGD step size")
        seed = int(training["seed"])
        metrics = payload.get("metrics") or {}
        if not {"clean", "pgd50"}.issubset(metrics):
            raise ValueError(f"{result_path}: missing clean or pgd50")
        if seed in groups.setdefault(epsilon, {}):
            raise ValueError(f"Duplicate best.pt evaluation for epsilon={epsilon}, seed={seed}")
        groups[epsilon][seed] = {
            "epoch": int(payload["checkpoint_epoch"]),
            "clean": float(metrics["clean"]) * 100.0,
            "pgd50": float(metrics["pgd50"]) * 100.0,
        }

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "best_checkpoint_pgd50_summary.csv"
    markdown_path = output_dir / "best_checkpoint_pgd50_summary.md"
    fields = ["epsilon", "seeds", "best_epochs", "clean_mean", "clean_std", "pgd50_mean", "pgd50_std", "status"]
    rows: list[dict[str, object]] = []
    for epsilon in expected_epsilons:
        values = groups.get(epsilon, {})
        seeds = sorted(values)
        if seeds != [0, 1, 2]:
            rows.append({"epsilon": epsilon, "seeds": ",".join(map(str, seeds)), "status": "incomplete"})
            continue
        clean_mean, clean_std = _mean_std([float(values[seed]["clean"]) for seed in seeds])
        pgd_mean, pgd_std = _mean_std([float(values[seed]["pgd50"]) for seed in seeds])
        rows.append(
            {
                "epsilon": epsilon,
                "seeds": "0,1,2",
                "best_epochs": ",".join(str(values[seed]["epoch"]) for seed in seeds),
                "clean_mean": clean_mean,
                "clean_std": clean_std,
                "pgd50_mean": pgd_mean,
                "pgd50_std": pgd_std,
                "status": "complete",
            }
        )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# AAER Ours-FD best-checkpoint diagnostic",
        "",
        "These are **not** Table-2 results. Each checkpoint was selected during training by the 1,000-example PGD-10 monitor, then independently evaluated on the full test set with AAER matched-epsilon PGD-50-10. Formal baseline results use final.pt only.",
        "",
        "| Metric | " + " | ".join(f"{epsilon}/255" for epsilon in expected_epsilons) + " |",
        "|---|" + "---:|" * len(expected_epsilons),
    ]
    for metric, mean_key, std_key in (("best checkpoint epochs (seed 0/1/2)", "best_epochs", None), ("clean", "clean_mean", "clean_std"), ("PGD-50-10", "pgd50_mean", "pgd50_std")):
        cells = []
        for row in rows:
            if row["status"] != "complete":
                cells.append("MISSING")
            elif std_key is None:
                cells.append(str(row[mean_key]))
            else:
                cells.append(f"{float(row[mean_key]):.2f} ± {float(row[std_key]):.2f}")
        lines.append(f"| {metric} | " + " | ".join(cells) + " |")
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return csv_path, markdown_path


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
        "--checkpoint-role",
        choices=("final", "best"),
        default="final",
        help="final produces the formal Table-2 summary; best is an explicitly diagnostic evaluation.",
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

    checkpoint_role = str(args.checkpoint_role)
    checkpoint_filename = f"{checkpoint_role}.pt"
    missing = [job["name"] for job in jobs if not (Path(job["run_dir"]) / checkpoint_filename).is_file()]
    if missing:
        parser.error(f"Missing {checkpoint_filename} checkpoints: " + ", ".join(missing))

    with (args.output_root / f".selected-{checkpoint_role}-evaluation.lock").open("a", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another selected Ours-FD evaluation holds this output directory")

        logs = args.output_root / "logs"
        logs.mkdir(exist_ok=True)
        pending = []
        for job in jobs:
            evaluation_name = (
                f"eval_{job['name']}" if checkpoint_role == "final" else f"eval_best_{job['name']}"
            )
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
                    checkpoint = Path(job["run_dir"]) / checkpoint_filename
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

    evaluation_prefix = "eval_" if checkpoint_role == "final" else "eval_best_"
    results = [args.output_root / f"{evaluation_prefix}{job['name']}" / "evaluation.json" for job in jobs]
    if checkpoint_role == "final":
        summary_dir = args.output_root / "aaer_selected_ours_fd_table2"
        command = [
            sys.executable,
            "-m",
            "co_blessing",
            "aaer-summary",
            "--results",
            *map(str, results),
            "--output",
            str(summary_dir),
        ]
        subprocess.run(command, cwd=ROOT, check=True)
        print(f"AAER selected Ours-FD summary: {summary_dir / 'aaer_table2_summary.md'}")
    else:
        summary_dir = args.output_root / "aaer_selected_ours_fd_best_diagnostic"
        _, markdown_path = summarize_best_diagnostic(
            results, summary_dir, expected_epsilons=tuple(sorted({int(job["epsilon"]) for job in jobs}))
        )
        print(f"AAER selected Ours-FD best-checkpoint diagnostic: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
