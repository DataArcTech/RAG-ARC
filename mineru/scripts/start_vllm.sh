#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODEL_KEY="mineru2.5"
MODEL_PATH=""
MODEL_SOURCE="modelscope" # modelscope | hf
PORT="30000"
GPU_MEM_UTIL="0.8"
CUDA_DEVICES=""
TP_SIZE=""
SERVED_MODEL_NAME=""
DRY_RUN="0"
EXTRA_ARGS=()

usage() {
  cat <<'EOF'
Usage: start_vllm.sh [options] [-- extra vLLM args...]

Options:
  --model-key <key>        Built-in model key (default: mineru2.5)
  --model-path <path>      Explicit model path (overrides model-key/source)
  --model-source <src>     modelscope | hf (default: modelscope)
  --served-model-name <s>  OpenAI served model name (default: derived)
  --port <port>            vLLM port (default: 30000)
  --gpus <ids>             CUDA_VISIBLE_DEVICES list, e.g. 0,1 (default: unset)
  --tp <n>                 tensor parallel size (default: derived from --gpus or 1)
  --gpu-mem-util <r>       vLLM GPU memory utilization (default: 0.8)
  --dry-run                Print command and exit
  -h, --help               Show this help

Notes:
  - vLLM only accelerates VLM models (e.g. MinerU2.5 VLM). OCR models such as
    PaddleOCR are not served by vLLM.
  - If --gpus is provided and --tp is not, TP size is derived from GPU count.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-key)
      MODEL_KEY="${2:-}"; shift 2;;
    --model-path)
      MODEL_PATH="${2:-}"; shift 2;;
    --model-source)
      MODEL_SOURCE="${2:-}"; shift 2;;
    --served-model-name)
      SERVED_MODEL_NAME="${2:-}"; shift 2;;
    --port)
      PORT="${2:-}"; shift 2;;
    --gpus)
      CUDA_DEVICES="${2:-}"; shift 2;;
    --tp)
      TP_SIZE="${2:-}"; shift 2;;
    --gpu-mem-util)
      GPU_MEM_UTIL="${2:-}"; shift 2;;
    --dry-run)
      DRY_RUN="1"; shift;;
    -h|--help)
      usage; exit 0;;
    --)
      shift; EXTRA_ARGS+=("$@"); break;;
    *)
      EXTRA_ARGS+=("$1"); shift;;
  esac
done

resolve_model_path() {
  if [[ -n "$MODEL_PATH" ]]; then
    echo "$MODEL_PATH"
    return
  fi

  case "$MODEL_KEY" in
    mineru2.5|mineru2.5-1.2b)
      if [[ "$MODEL_SOURCE" == "modelscope" ]]; then
        echo "${HOME}/.cache/modelscope/hub/models/OpenDataLab/MinerU2.5-2509-1.2B"
        return
      fi
      if [[ "$MODEL_SOURCE" == "hf" ]]; then
        local base="${HOME}/.cache/huggingface/hub/models--opendatalab--MinerU2.5-2509-1.2B/snapshots"
        if [[ -d "$base" ]]; then
          ls -dt "$base"/* 2>/dev/null | head -n1
          return
        fi
        echo "${HOME}/.cache/huggingface/hub/models--opendatalab--MinerU2.5-2509-1.2B"
        return
      fi
      ;;
    *)
      echo "Unsupported model-key for vLLM: ${MODEL_KEY}" >&2
      echo "Hint: use --model-path for custom VLM models." >&2
      exit 1
      ;;
  esac
}

MODEL_PATH="$(resolve_model_path)"
if [[ -z "$MODEL_PATH" ]]; then
  echo "Failed to resolve model path. Use --model-path." >&2
  exit 1
fi

if [[ -n "$CUDA_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
  if [[ -z "$TP_SIZE" ]]; then
    IFS=',' read -ra _devs <<< "$CUDA_DEVICES"
    TP_SIZE="${#_devs[@]}"
  fi
fi

if [[ -z "$TP_SIZE" ]]; then
  TP_SIZE="1"
fi

if [[ -z "$SERVED_MODEL_NAME" ]]; then
  SERVED_MODEL_NAME="$(basename "$MODEL_PATH")"
fi

CMD=(vllm serve "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --port "$PORT"
  --tensor-parallel-size "$TP_SIZE"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
)

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY RUN:"
  printf ' %q' "${CMD[@]}"
  echo
  exit 0
fi

echo "Starting vLLM server..."
echo "  model_path: ${MODEL_PATH}"
echo "  served_name: ${SERVED_MODEL_NAME}"
echo "  port: ${PORT}"
echo "  tp_size: ${TP_SIZE}"
echo "  gpu_mem_util: ${GPU_MEM_UTIL}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
fi

exec "${CMD[@]}"
