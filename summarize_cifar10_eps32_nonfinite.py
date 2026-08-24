#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


RUN_NAMES = (
    "diagnostic_mep_eps32_alpha16_logit10_lr01",
    "diagnostic_mep_eps32_alpha8_logit10_lr01",
    "diagnostic_mep_ce_eps32_alpha8_lr01",
    "diagnostic_mep_eps32_alpha8_logit10_lr001",
)


def _nonfinite_names(section: object) -> str:
    if not isinstance(section, dict):
        return ""
    values = section.get("nonfinite", {})
    return ",".join(values) if isinstance(values, dict) else ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize epsilon=32 non-finite diagnostics")
    parser.add_argument(
        "output_root",
        nargs="?",
        default="/data/cjk/FGSM-MEP-cifar10-eps32-nonfinite",
    )
    root = Path(parser.parse_args().output_root)
    print(
        "| Run | Outcome | Stage | Epoch | Batch | Nonfinite tensors "
        "| Gradients | Parameters | Buffers |"
    )
    print("|---|---|---|---:|---:|---|---|---|---|")
    complete = True
    for run_name in RUN_NAMES:
        run = root / run_name
        diagnostic_path = run / "nonfinite_diagnostic.json"
        final_path = run / "final.pt"
        if diagnostic_path.exists():
            payload = json.loads(diagnostic_path.read_text(encoding="utf-8"))
            tensors = payload.get("tensors", {})
            tensor_names = ",".join(
                name
                for name, stats in tensors.items()
                if float(stats.get("finite_fraction", 1.0)) < 1.0
            )
            print(
                f"| {run_name} | NONFINITE | {payload['stage']} | {payload['epoch']} "
                f"| {payload['batch']} | {tensor_names} | "
                f"{_nonfinite_names(payload.get('gradients'))} | "
                f"{_nonfinite_names(payload.get('parameters'))} | "
                f"{_nonfinite_names(payload.get('buffers'))} |"
            )
        elif final_path.exists():
            print(f"| {run_name} | FINITE | completed | 0 | - | | | | |")
        else:
            complete = False
            print(f"| {run_name} | MISSING | | | | | | | |")
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
