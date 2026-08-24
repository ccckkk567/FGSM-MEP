#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:-/data/cjk/cifar-data}"
OUTPUT_ROOT="${2:-/data/cjk/FGSM-MEP-cifar10-fd-pilots}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIGS=(
  "configs/train/pilot_fd_eps32_alpha32_fw25.yaml"
  "configs/train/pilot_fd_eps32_alpha16_fw25.yaml"
  "configs/train/pilot_fd_eps32_alpha16_fw10.yaml"
  "configs/train/pilot_mep_baseline_eps32_alpha16.yaml"
)
RUN_NAMES=(
  "pilot_fd_eps32_alpha32_fw25"
  "pilot_fd_eps32_alpha16_fw25"
  "pilot_fd_eps32_alpha16_fw10"
  "pilot_mep_baseline_eps32_alpha16"
)
GPUS=(0 1 2 3)

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${REPO_ROOT}"

pids=()
terminate_children() {
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap 'terminate_children; exit 130' INT TERM

for index in "${!CONFIGS[@]}"; do
  config="${CONFIGS[$index]}"
  run_name="${RUN_NAMES[$index]}"
  gpu="${GPUS[$index]}"
  log_path="${OUTPUT_ROOT}/logs/${run_name}.log"
  resume_path="${OUTPUT_ROOT}/${run_name}/resume.pt"
  final_path="${OUTPUT_ROOT}/${run_name}/final.pt"

  if [[ -f "${final_path}" ]]; then
    echo "skip completed ${run_name}"
    continue
  fi

  resume_args=()
  if [[ -f "${resume_path}" ]]; then
    resume_args=(--resume "${resume_path}")
  fi

  echo "launch ${run_name} on physical GPU ${gpu}; log=${log_path}"
  (
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u -m co_blessing train \
      --config "${config}" \
      --data-root "${DATA_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --device cuda:0 \
      "${resume_args[@]}"
  ) >"${log_path}" 2>&1 &
  pids+=("$!")
done

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "all pilots are already complete"
  exit 0
fi

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
trap - INT TERM

if [[ "${status}" -ne 0 ]]; then
  echo "one or more pilots failed; inspect ${OUTPUT_ROOT}/logs" >&2
  exit "${status}"
fi

echo "all epsilon=32 pilots completed"
