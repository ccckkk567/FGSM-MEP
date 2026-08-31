#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:-/data/cjk/cifar-data}"
OUTPUT_ROOT="${2:-/data/cjk/FGSM-MEP-cifar10-native-high-eps-audit}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIGS=(
  "configs/train/native_audit_ours_fd_eps32.yaml"
  "configs/train/native_audit_ours_fd_eps48.yaml"
  "configs/train/native_audit_ours_fd_eps64.yaml"
)
RUN_NAMES=(
  "native_audit_ours_fd_eps32"
  "native_audit_ours_fd_eps48"
  "native_audit_ours_fd_eps64"
)
GPUS=(5 6 7)

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${REPO_ROOT}"

pids=()
# shellcheck disable=SC2317
stop_children() {
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap 'stop_children; exit 130' INT TERM

for index in "${!CONFIGS[@]}"; do
  run_dir="${OUTPUT_ROOT}/${RUN_NAMES[$index]}"
  log_path="${OUTPUT_ROOT}/logs/${RUN_NAMES[$index]}.log"
  if [[ -f "${run_dir}/final.pt" || -f "${run_dir}/nonfinite_diagnostic.json" ]]; then
    echo "skip recorded audit ${RUN_NAMES[$index]}"
    continue
  fi
  echo "launch ${RUN_NAMES[$index]} on physical GPU ${GPUS[$index]}; log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${GPUS[$index]}"
    exec "${PYTHON_BIN}" -u -m co_blessing train \
      --config "${CONFIGS[$index]}" \
      --data-root "${DATA_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --device cuda:0
  ) >"${log_path}" 2>&1 &
  pids+=("$!")
done

# A non-zero child status is the expected, recorded outcome for NaN/Inf.
for pid in "${pids[@]}"; do
  wait "${pid}" || true
done
trap - INT TERM

echo "native high-epsilon audit finished; summarize the recorded outcomes before any 110-epoch run"
