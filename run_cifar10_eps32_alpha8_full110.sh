#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

# One process per physical GPU: 1=MEP, 5=FD50, 6=FD200. No tmux or AA.
exec "${PYTHON_BIN}" -u "${REPO_ROOT}/continue_cifar10_eps32_alpha8_full110.py" \
  --data-root "${1:-/data/cjk/cifar-data}" \
  --output-root "${2:-/data/cjk/FGSM-MEP-cifar10-eps32-alpha8-full110}" \
  --pilot-root "${3:-/data/cjk/FGSM-MEP-cifar10-eps32-alpha8-pilots}" \
  --highfd-root "${4:-/data/cjk/FGSM-MEP-cifar10-eps32-alpha8-highfd}" \
  --gpus 1 5 6
