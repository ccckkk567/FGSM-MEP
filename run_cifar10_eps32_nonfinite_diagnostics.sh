#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:-/data/cjk/cifar-data}"
OUTPUT_ROOT="${2:-/data/cjk/FGSM-MEP-cifar10-eps32-nonfinite}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIGS=(
  "configs/train/diagnostic_mep_eps32_alpha16_logit10_lr01.yaml"
  "configs/train/diagnostic_mep_eps32_alpha8_logit10_lr01.yaml"
  "configs/train/diagnostic_mep_ce_eps32_alpha8_lr01.yaml"
  "configs/train/diagnostic_mep_eps32_alpha8_logit10_lr001.yaml"
)
RUN_NAMES=(
  "diagnostic_mep_eps32_alpha16_logit10_lr01"
  "diagnostic_mep_eps32_alpha8_logit10_lr01"
  "diagnostic_mep_ce_eps32_alpha8_lr01"
  "diagnostic_mep_eps32_alpha8_logit10_lr001"
)

mkdir -p "${OUTPUT_ROOT}/logs"
cd "${REPO_ROOT}"

pids=()
launched_pid=""
# shellcheck disable=SC2317
terminate_children() {
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap 'terminate_children; exit 130' INT TERM

launch_one() {
  local index="$1"
  local gpu="$2"
  local config="${CONFIGS[$index]}"
  local run_name="${RUN_NAMES[$index]}"
  local log_path="${OUTPUT_ROOT}/logs/${run_name}.log"
  local final_path="${OUTPUT_ROOT}/${run_name}/final.pt"

  launched_pid=""
  if [[ -f "${final_path}" ]]; then
    echo "skip completed ${run_name}"
    return
  fi

  echo "launch ${run_name} on physical GPU ${gpu}; log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${PYTHON_BIN}" -u -m co_blessing train \
      --config "${config}" \
      --data-root "${DATA_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --device cuda:0
  ) >"${log_path}" 2>&1 &
  launched_pid="$!"
  pids+=("${launched_pid}")
}

# GPU 1 runs the reference failure first and the low-LR case second.
launch_one 0 1
pid_gpu1_first="${launched_pid}"
launch_one 1 5
pid_gpu5="${launched_pid}"
launch_one 2 6
pid_gpu6="${launched_pid}"

# A non-zero status is expected when the non-finite guard identifies a failure.
if [[ -n "${pid_gpu1_first}" ]]; then
  wait "${pid_gpu1_first}" || true
fi
launch_one 3 1
pid_gpu1_second="${launched_pid}"

for pid in "${pid_gpu5}" "${pid_gpu6}" "${pid_gpu1_second}"; do
  if [[ -n "${pid}" ]]; then
    wait "${pid}" || true
  fi
done
trap - INT TERM

echo "diagnostics finished; non-zero child exits are expected when NaN is caught"
echo "inspect ${OUTPUT_ROOT}/*/nonfinite_diagnostic.json and ${OUTPUT_ROOT}/logs"
exit 0
