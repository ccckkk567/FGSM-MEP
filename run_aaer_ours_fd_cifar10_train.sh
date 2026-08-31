#!/usr/bin/env bash
# Run the 18 AAER-compatible Ours-FD CIFAR-10 training jobs in bounded
# parallelism.  Each worker owns disjoint run directories and resumes safely.
set -uo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 DATA_ROOT OUTPUT_ROOT [GPU_IDS_COMMA_SEPARATED]" >&2
  exit 2
fi

DATA_ROOT="$1"
OUTPUT_ROOT="$2"
GPU_CSV="${3:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_CSV}"
if [[ ${#GPU_IDS[@]} -eq 0 || -z "${GPU_IDS[0]}" ]]; then
  echo "at least one GPU id is required" >&2
  exit 2
fi

CONFIGS=(
  configs/train/aaer_ours_fd_cifar10_eps8_seed0.yaml
  configs/train/aaer_ours_fd_cifar10_eps8_seed1.yaml
  configs/train/aaer_ours_fd_cifar10_eps8_seed2.yaml
  configs/train/aaer_ours_fd_cifar10_eps12_seed0.yaml
  configs/train/aaer_ours_fd_cifar10_eps12_seed1.yaml
  configs/train/aaer_ours_fd_cifar10_eps12_seed2.yaml
  configs/train/aaer_ours_fd_cifar10_eps16_seed0.yaml
  configs/train/aaer_ours_fd_cifar10_eps16_seed1.yaml
  configs/train/aaer_ours_fd_cifar10_eps16_seed2.yaml
  configs/train/aaer_ours_fd_cifar10_eps32_seed0.yaml
  configs/train/aaer_ours_fd_cifar10_eps32_seed1.yaml
  configs/train/aaer_ours_fd_cifar10_eps32_seed2.yaml
  configs/train/aaer_ours_fd_cifar10_eps48_seed0.yaml
  configs/train/aaer_ours_fd_cifar10_eps48_seed1.yaml
  configs/train/aaer_ours_fd_cifar10_eps48_seed2.yaml
  configs/train/aaer_ours_fd_cifar10_eps64_seed0.yaml
  configs/train/aaer_ours_fd_cifar10_eps64_seed1.yaml
  configs/train/aaer_ours_fd_cifar10_eps64_seed2.yaml
)

mkdir -p "${OUTPUT_ROOT}/logs"
QUEUES=()
for ((index = 0; index < ${#GPU_IDS[@]}; index++)); do
  QUEUES[index]=""
done
for ((index = 0; index < ${#CONFIGS[@]}; index++)); do
  slot=$((index % ${#GPU_IDS[@]}))
  QUEUES[slot]+=" ${CONFIGS[index]}"
done

worker() {
  local gpu="$1"
  shift
  local status=0
  local config run_name run_dir resume_path log_path
  for config in "$@"; do
    run_name="$(basename "${config}" .yaml)"
    run_dir="${OUTPUT_ROOT}/${run_name}"
    log_path="${OUTPUT_ROOT}/logs/${run_name}.log"
    if [[ -f "${run_dir}/final.pt" ]]; then
      echo "skip completed ${run_name}"
      continue
    fi
    resume_path=()
    if [[ -f "${run_dir}/resume.pt" ]]; then
      resume_path=(--resume "${run_dir}/resume.pt")
    fi
    echo "launch ${run_name} on physical GPU ${gpu}; log=${log_path}"
    CUDA_VISIBLE_DEVICES="${gpu}" python -m co_blessing train \
      --config "${config}" \
      "${resume_path[@]}" \
      --data-root "${DATA_ROOT}" \
      --output-root "${OUTPUT_ROOT}" \
      --device cuda:0 2>&1 | tee "${log_path}" || status=1
  done
  return "${status}"
}

PIDS=()
for ((index = 0; index < ${#GPU_IDS[@]}; index++)); do
  # shellcheck disable=SC2086
  worker "${GPU_IDS[index]}" ${QUEUES[index]} &
  PIDS+=("$!")
done

status=0
for pid in "${PIDS[@]}"; do
  wait "${pid}" || status=1
done
if [[ "${status}" -ne 0 ]]; then
  echo "one or more runs failed; inspect ${OUTPUT_ROOT}/logs and nonfinite_diagnostic.json" >&2
  exit "${status}"
fi
echo "all AAER-compatible CIFAR-10 Ours-FD training jobs finished"
