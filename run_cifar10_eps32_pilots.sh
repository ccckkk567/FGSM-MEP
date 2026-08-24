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
mkdir -p "${OUTPUT_ROOT}/logs"
cd "${REPO_ROOT}"

pids=()
launched_pid=""
terminate_children() {
  for pid in "${pids[@]}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap 'terminate_children; exit 130' INT TERM

launch_one() {
  local index="$1"
  local gpu="$2"
  config="${CONFIGS[$index]}"
  run_name="${RUN_NAMES[$index]}"
  log_path="${OUTPUT_ROOT}/logs/${run_name}.log"
  resume_path="${OUTPUT_ROOT}/${run_name}/resume.pt"
  final_path="${OUTPUT_ROOT}/${run_name}/final.pt"

  if [[ -f "${final_path}" ]]; then
    echo "skip completed ${run_name}"
    launched_pid=""
    return
  fi

  resume_args=()
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

# GPU 4 runs two jobs sequentially; GPUs 5 and 6 each run one job.
launch_one 0 4
pid_gpu4_first="${launched_pid}"
launch_one 1 5
pid_gpu5="${launched_pid}"
launch_one 2 6
pid_gpu6="${launched_pid}"

status=0
if [[ -n "${pid_gpu4_first}" ]] && ! wait "${pid_gpu4_first}"; then
  status=1
fi

# Start the pure-MEP control as soon as GPU 4 becomes free.
launch_one 3 4
pid_gpu4_second="${launched_pid}"

for pid in "${pid_gpu5}" "${pid_gpu6}" "${pid_gpu4_second}"; do
  if [[ -n "${pid}" ]] && ! wait "${pid}"; then
    status=1
  fi
done
trap - INT TERM

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "all pilots are already complete"
  exit 0
fi

if [[ "${status}" -ne 0 ]]; then
  echo "one or more pilots failed; inspect ${OUTPUT_ROOT}/logs" >&2
  exit "${status}"
fi

echo "all epsilon=32 pilots completed"
