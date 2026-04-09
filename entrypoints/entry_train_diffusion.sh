#!/bin/bash

set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

CONDA_ACTIVATE_SCRIPT="${CONDA_ACTIVATE_SCRIPT:-/mnt/bn/isp-traindata-lf3/liujinkun/miniforge3/bin/activate}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-/mnt/bn/embodied-lf3/liujinkun/envs/gld}"
if [[ -f "${CONDA_ACTIVATE_SCRIPT}" && -d "${CONDA_ENV_PATH}" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_ACTIVATE_SCRIPT}" "${CONDA_ENV_PATH}"
fi

export HTTP_PROXY="${HTTP_PROXY:-http://bj-rd-proxy.byted.org:8118}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://bj-rd-proxy.byted.org:8118}"
export http_proxy="${http_proxy:-${HTTP_PROXY}}"
export https_proxy="${https_proxy:-${HTTPS_PROXY}}"
unset no_proxy
unset NO_PROXY

export ENTITY="${ENTITY:-}"
export PROJECT="${PROJECT:-combustion-tsr}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY}}"

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONPATH="src:${PYTHONPATH:-}"

DEFAULT_EXP_NAME="combustion-diffusion-v0"
EXP_NAME="${EXP_NAME:-${DEFAULT_EXP_NAME}}"
CONFIG_PATH="${CONFIG_PATH:-configs/diffusion_base.yaml}"
RESULTS_DIR="${RESULTS_DIR:-results/diffusion}"
PRECISION="${PRECISION:-bf16}"
RESUME="${RESUME:-false}"
CKPT_PATH="${CKPT_PATH:-}"
AUTOENCODER_CKPT="${AUTOENCODER_CKPT:-outputs/autoencoder/checkpoints/best.pt}"
LOG_ROOT="${LOG_ROOT:-logs/diffusion}"
WANDB_ENABLED="${WANDB_ENABLED:-true}"
EXTRA_OVERRIDES=("$@")

mkdir -p "${LOG_ROOT}" "${RESULTS_DIR}"

num_gpus="${NPROC_PER_NODE:-${ARNOLD_WORKER_GPU:-1}}"
num_nodes="${NNODES:-${ARNOLD_WORKER_NUM:-1}}"
node_rank="${NODE_RANK:-${ARNOLD_ID:-0}}"
master_addr="${MASTER_ADDR:-${ARNOLD_WORKER_0_HOST:-127.0.0.1}}"
if [[ -n "${MASTER_PORT:-}" ]]; then
  master_port="${MASTER_PORT}"
elif [[ -n "${ARNOLD_WORKER_0_PORT:-}" ]]; then
  IFS=',' read -ra PORTS <<< "${ARNOLD_WORKER_0_PORT}"
  master_port="${PORTS[0]}"
else
  master_port=29500
fi

case "${RESUME,,}" in
  1|true|yes|y) RESUME_ENABLED="true" ;;
  0|false|no|n) RESUME_ENABLED="false" ;;
  *) echo "Invalid RESUME value: ${RESUME}" >&2; exit 1 ;;
esac

if [[ "${RESUME_ENABLED}" == "true" && -z "${CKPT_PATH}" ]]; then
  latest_ckpt="$(find "${RESULTS_DIR}" -path "*/checkpoints/*.pt" -type f | sort | tail -n 1 || true)"
  if [[ -n "${latest_ckpt}" ]]; then
    CKPT_PATH="${latest_ckpt}"
  fi
fi

if [[ "${WANDB_ENABLED,,}" =~ ^(1|true|yes|y)$ ]]; then
  WANDB_ENABLED_BOOL="true"
else
  WANDB_ENABLED_BOOL="false"
fi

echo "--- Combustion Diffusion Cluster Launch ---"
echo "Experiment Name: ${EXP_NAME}"
echo "Config Path: ${CONFIG_PATH}"
echo "Results Dir: ${RESULTS_DIR}"
echo "Log Root: ${LOG_ROOT}"
echo "Precision: ${PRECISION}"
echo "Autoencoder ckpt: ${AUTOENCODER_CKPT}"
echo "Num GPUs / node: ${num_gpus}"
echo "Num nodes: ${num_nodes}"
echo "Node rank: ${node_rank}"
echo "Master addr: ${master_addr}"
echo "Master port: ${master_port}"
echo "Resume enabled: ${RESUME_ENABLED}"
echo "Checkpoint path: ${CKPT_PATH:-<none>}"
echo "WandB enabled: ${WANDB_ENABLED_BOOL}"
echo "CLI overrides: ${EXTRA_OVERRIDES[*]:-<none>}"
echo "-------------------------------------------"

timestamp="$(date +%F_%H%M%S)"
log_path="${LOG_ROOT}/${EXP_NAME}_${timestamp}.log"
: > "${log_path}"

cmd=(
  torchrun
  --nproc_per_node "${num_gpus}"
  --nnodes "${num_nodes}"
  --node_rank "${node_rank}"
  --master_addr "${master_addr}"
  --master_port "${master_port}"
  scripts/train_diffusion.py
  --config "${CONFIG_PATH}"
  --override
  "experiment.name=${EXP_NAME}"
  "experiment.output_dir=${RESULTS_DIR}/${EXP_NAME}"
  "trainer.precision=${PRECISION}"
  "model.autoencoder.checkpoint=${AUTOENCODER_CKPT}"
  "wandb.enabled=${WANDB_ENABLED_BOOL}"
  "wandb.project=${PROJECT}"
  "wandb.entity=${ENTITY}"
  "wandb.run_name=${EXP_NAME}"
)

if [[ -n "${CKPT_PATH}" ]]; then
  cmd+=("trainer.resume=${CKPT_PATH}")
fi

if [[ ${#EXTRA_OVERRIDES[@]} -gt 0 ]]; then
  cmd+=("${EXTRA_OVERRIDES[@]}")
fi

nohup "${cmd[@]}" > >(tee -a "${log_path}") 2>&1 &

echo $! > "${LOG_ROOT}/${EXP_NAME}.pid"
echo "Started training. PID: $(cat "${LOG_ROOT}/${EXP_NAME}.pid")"
echo "Log file: ${log_path}"
sleep infinity
