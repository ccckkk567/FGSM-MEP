#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DATA_ROOT="${1:-/data/cjk/cifar-data}"
OUTPUT_ROOT="${2:-/data/cjk/FGSM-MEP-cifar10-eps32-alpha8-highfd}"
PYTHON_BIN="${PYTHON_BIN:-python}"

CONFIGS=(
  "configs/train/pilot_fd_eps32_alpha8_fw50.yaml"
  "configs/train/pilot_fd_eps32_alpha8_fw100.yaml"
  "configs/train/pilot_fd_eps32_alpha8_fw200.yaml"
  "configs/train/pilot_fd_eps32_alpha8_fw400.yaml"
)
RUN_NAMES=(
  "pilot_fd_eps32_alpha8_fw50"
  "pilot_fd_eps32_alpha8_fw100"
  "pilot_fd_eps32_alpha8_fw200"
  "pilot_fd_eps32_alpha8_fw400"
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
  local run_dir="${OUTPUT_ROOT}/${run_name}"
  local log_path="${OUTPUT_ROOT}/logs/${run_name}.log"
  local final_path="${run_dir}/final.pt"
  local resume_path="${run_dir}/resume.pt"
  local resume_args=()

  launched_pid=""
  if [[ -f "${final_path}" ]]; then
    echo "skip completed ${run_name}"
    return
  fi
  if [[ -f "${resume_path}" ]]; then
    resume_args=(--resume "${resume_path}")
  fi

  echo "launch ${run_name} on physical GPU ${gpu}; log=${log_path}"
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    exec "${PYTHON_BIN}" -u -m co_blessing train \
      --config "${config}" \
      --data-root "${DATA_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --device cuda:0 \
      "${resume_args[@]}"
  ) >"${log_path}" 2>&1 &
  launched_pid="$!"
  pids+=("${launched_pid}")
}

# GPU 1 runs weights 50 and 400 sequentially; GPUs 5 and 6 run 100 and 200.
launch_one 0 1
pid_gpu1_first="${launched_pid}"
launch_one 1 5
pid_gpu5="${launched_pid}"
launch_one 2 6
pid_gpu6="${launched_pid}"

status=0
if [[ -n "${pid_gpu1_first}" ]] && ! wait "${pid_gpu1_first}"; then
  status=1
fi
launch_one 3 1
pid_gpu1_second="${launched_pid}"

for pid in "${pid_gpu5}" "${pid_gpu6}" "${pid_gpu1_second}"; do
  if [[ -n "${pid}" ]] && ! wait "${pid}"; then
    status=1
  fi
done
trap - INT TERM

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "all high-FD pilots are already complete"
  exit 0
fi
if [[ "${status}" -ne 0 ]]; then
  echo "one or more pilots failed; inspect ${OUTPUT_ROOT}/logs and diagnostics" >&2
  exit "${status}"
fi
echo "all epsilon=32 high-FD pilots completed"
