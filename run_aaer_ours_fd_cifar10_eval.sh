#!/usr/bin/env bash
# Evaluate completed final checkpoints with AAER Table-2 PGD-50-10.
set -uo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "usage: $0 DATA_ROOT OUTPUT_ROOT [GPU_IDS_COMMA_SEPARATED]" >&2
  exit 2
fi

DATA_ROOT="$1"
OUTPUT_ROOT="$2"
GPU_CSV="${3:-0,1,2,3}"
IFS=',' read -r -a GPU_IDS <<< "${GPU_CSV}"

SPECS=(
  '8:0' '8:1' '8:2' '12:0' '12:1' '12:2' '16:0' '16:1' '16:2'
  '32:0' '32:1' '32:2' '48:0' '48:1' '48:2' '64:0' '64:1' '64:2'
)
mkdir -p "${OUTPUT_ROOT}/logs"
QUEUES=()
for ((index = 0; index < ${#GPU_IDS[@]}; index++)); do QUEUES[index]=""; done
for ((index = 0; index < ${#SPECS[@]}; index++)); do
  slot=$((index % ${#GPU_IDS[@]}))
  QUEUES[slot]+=" ${SPECS[index]}"
done

worker() {
  local gpu="$1"
  shift
  local status=0 spec epsilon seed run_name checkpoint result_path log_path
  for spec in "$@"; do
    epsilon="${spec%%:*}"
    seed="${spec##*:}"
    run_name="aaer_ours_fd_cifar10_eps${epsilon}_seed${seed}"
    checkpoint="${OUTPUT_ROOT}/${run_name}/final.pt"
    result_path="${OUTPUT_ROOT}/eval_${run_name}/evaluation.json"
    log_path="${OUTPUT_ROOT}/logs/eval_${run_name}.log"
    if [[ ! -f "${checkpoint}" ]]; then
      echo "missing final checkpoint: ${checkpoint}" >&2
      status=1
      continue
    fi
    if [[ -f "${result_path}" ]]; then
      echo "skip completed evaluation ${run_name}"
      continue
    fi
    echo "evaluate ${run_name} on physical GPU ${gpu}; log=${log_path}"
    CUDA_VISIBLE_DEVICES="${gpu}" python -m co_blessing evaluate \
      --config "configs/eval/aaer_cifar10_eps${epsilon}.yaml" \
      --checkpoint "${checkpoint}" \
      --name "eval_${run_name}" \
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
for pid in "${PIDS[@]}"; do wait "${pid}" || status=1; done
if [[ "${status}" -ne 0 ]]; then
  echo "one or more evaluations failed; inspect ${OUTPUT_ROOT}/logs" >&2
  exit "${status}"
fi

RESULTS=()
for spec in "${SPECS[@]}"; do
  epsilon="${spec%%:*}"; seed="${spec##*:}"
  RESULTS+=("${OUTPUT_ROOT}/eval_aaer_ours_fd_cifar10_eps${epsilon}_seed${seed}/evaluation.json")
done
python -m co_blessing aaer-summary \
  --results "${RESULTS[@]}" \
  --output "${OUTPUT_ROOT}/aaer_ours_fd_cifar10_table2"
