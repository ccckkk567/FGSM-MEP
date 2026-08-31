#!/usr/bin/env python3
"""Summarize the alpha/LR high-epsilon Ours-FD stability screen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_ROOT = "/data/cjk/FGSM-MEP-cifar10-high-eps-stability-grid"


def _nonfinite_tensors(payload: dict[str, Any]) -> str:
    tensors = payload.get("tensors", {})
    if not isinstance(tensors, dict):
        return ""
    return ", ".join(
        name for name, stats in tensors.items()
        if isinstance(stats, dict) and float(stats.get("finite_fraction", 1.0)) < 1.0
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", nargs="?", default=DEFAULT_ROOT)
    root = Path(parser.parse_args(argv).output_root)
    manifest_path = root / "grid_manifest.json"
    if not manifest_path.exists():
        raise SystemExit(f"Missing grid manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs", [])
    complete = True
    print("# CIFAR-10 high-epsilon alpha/LR stability screen\n")
    print("This is a deterministic, one-epoch diagnostic screen with FD(B)=200 and "
          "logit weight=10. It does not replace the frozen formal baseline or select a "
          "final model. Finite prefixes require a separate multi-epoch CO trajectory check.\n")
    print("| εT | α | lr | Status | Stage | Batch | Clean | PGD-10 | Nonfinite tensors |")
    print("|---:|---:|---:|---|---|---:|---:|---:|---|")
    for job in jobs:
        run = Path(job["run_dir"])
        diagnostic = run / "nonfinite_diagnostic.json"
        final = run / "final_metrics.json"
        if diagnostic.exists():
            payload = json.loads(diagnostic.read_text(encoding="utf-8"))
            row = ("NONFINITE", str(payload.get("stage", "")), str(payload.get("batch", "")), "", "", _nonfinite_tensors(payload))
        elif final.exists():
            metrics = json.loads(final.read_text(encoding="utf-8"))
            row = ("FINITE_PREFIX", "completed", "", f"{100 * float(metrics['clean_accuracy']):.2f}",
                   f"{100 * float(metrics['pgd10_accuracy']):.2f}", "")
        else:
            complete = False
            row = ("MISSING", "", "", "", "", "")
        print(f"| {job['epsilon']}/255 | {job['alpha']}/255 | {job['lr']:g} | " + " | ".join(row) + " |")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
