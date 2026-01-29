#!/usr/bin/env bash
set -euo pipefail

# Start a pool of vLLM OpenAI-compatible servers (one instance per GPU by default).
# Each instance listens on a different port.
#
# Example:
#   conda activate mineru
#   ./service/scripts/start_vllm_pool.sh --gpus 4,5,6 --base-port 30000
#
# Then point the MinerU gateway at the pool:
#   python service/mineru_main.py server --backend vlm-http-client \
#     --server-urls http://127.0.0.1:30000,http://127.0.0.1:30001,http://127.0.0.1:30002

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_KEY="mineru2.5"
MODEL_PATH=""
MODEL_SOURCE="modelscope" # modelscope | hf
GPU_MEM_UTIL="0.8"
BASE_PORT="30000"
GPUS=""
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: start_vllm_pool.sh --gpus <ids> [options] [-- extra vLLM args...]

Options:
  --gpus <ids>             GPU id list, e.g. 4,6,7 (required)
  --base-port <port>       First port (default: 30000). Ports increment by 1.
  --model-key <key>        Built-in model key (default: mineru2.5)
  --model-path <path>      Explicit model path (overrides model-key/source)
  --model-source <src>     modelscope | hf (default: modelscope)
  --gpu-mem-util <r>       vLLM GPU memory utilization (default: 0.8)
  -h, --help               Show help

Daemonization:
  Each instance runs under nohup and writes:
    log: /tmp/vllm_<port>.log
    pid: /tmp/vllm_<port>.pid
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus) GPUS="${2:-}"; shift 2;;
    --base-port) BASE_PORT="${2:-}"; shift 2;;
    --model-key) MODEL_KEY="${2:-}"; shift 2;;
    --model-path) MODEL_PATH="${2:-}"; shift 2;;
    --model-source) MODEL_SOURCE="${2:-}"; shift 2;;
    --gpu-mem-util) GPU_MEM_UTIL="${2:-}"; shift 2;;
    -h|--help) usage; exit 0;;
    --) shift; EXTRA_ARGS+=("$@"); break;;
    *) EXTRA_ARGS+=("$1"); shift;;
  esac
done

if [[ -z "$GPUS" ]]; then
  echo "Missing --gpus" >&2
  usage
  exit 1
fi

MODEL_PATH_RESOLVED=""
if [[ -n "$MODEL_PATH" ]]; then
  MODEL_PATH_RESOLVED="$MODEL_PATH"
else
  if [[ "$MODEL_KEY" == "mineru2.5" || "$MODEL_KEY" == "mineru2.5-1.2b" ]]; then
    if [[ "$MODEL_SOURCE" == "modelscope" ]]; then
      MODEL_PATH_RESOLVED="${HOME}/.cache/modelscope/hub/models/OpenDataLab/MinerU2.5-2509-1.2B"
    elif [[ "$MODEL_SOURCE" == "hf" ]]; then
      MODEL_PATH_RESOLVED="${HOME}/.cache/huggingface/hub/models--opendatalab--MinerU2.5-2509-1.2B"
    else
      echo "Unsupported --model-source: $MODEL_SOURCE" >&2
      exit 1
    fi
  else
    echo "Unsupported --model-key for pool: $MODEL_KEY (use --model-path)" >&2
    exit 1
  fi
fi

IFS=',' read -ra GPU_LIST <<< "$GPUS"
PORT="$BASE_PORT"

echo "Starting vLLM pool:"
echo "  model_path: ${MODEL_PATH_RESOLVED}"
echo "  gpus: ${GPUS}"
echo "  base_port: ${BASE_PORT}"
echo "  gpu_mem_util: ${GPU_MEM_UTIL}"

for gpu in "${GPU_LIST[@]}"; do
  gpu="$(echo "$gpu" | xargs)"
  if [[ -z "$gpu" ]]; then
    continue
  fi
  log="/tmp/vllm_${PORT}.log"
  pidfile="/tmp/vllm_${PORT}.pid"

  echo "  - GPU ${gpu} -> port ${PORT} (log=${log})"
  CUDA_VISIBLE_DEVICES="${gpu}" \
  nohup vllm serve "${MODEL_PATH_RESOLVED}" \
    --served-model-name "$(basename "${MODEL_PATH_RESOLVED}")" \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    "${EXTRA_ARGS[@]}" \
    > "${log}" 2>&1 &

  echo $! > "${pidfile}"
  PORT=$((PORT + 1))
done

echo "Done. Example server-urls:"
python - <<PY
base = int("${BASE_PORT}")
gpus = [g.strip() for g in "${GPUS}".split(",") if g.strip()]
urls = [f"http://127.0.0.1:{base+i}" for i in range(len(gpus))]
print("  " + ",".join(urls))
PY
