#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"

# vLLM LLM server
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_LLM_PORT="${VLLM_LLM_PORT:-8080}"
LLM_MODEL="${LLM_MODEL:-BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM}"
VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-awq_marlin}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.70}"
# Device for LLM server: 'gpu' (default) or 'cpu'
LLM_DEVICE="${LLM_DEVICE:-gpu}"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

LLM_LOG="$LOG_DIR/vllm_llm.log"

echo "Starting vLLM LLM server on port $VLLM_LLM_PORT (device: $LLM_DEVICE) ..."
if [[ "$LLM_DEVICE" == "cpu" ]]; then
  CUDA_VISIBLE_DEVICES="" python -m vllm.entrypoints.openai.api_server \
    --model "$LLM_MODEL" \
    --host "$VLLM_HOST" \
    --port "$VLLM_LLM_PORT" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --device cpu \
    --dtype half \
    --trust-remote-code \
    2>&1 | tee "$LLM_LOG"
else
  python -m vllm.entrypoints.openai.api_server \
    --model "$LLM_MODEL" \
    --quantization "$VLLM_QUANTIZATION" \
    --host "$VLLM_HOST" \
    --port "$VLLM_LLM_PORT" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
    --trust-remote-code \
    2>&1 | tee "$LLM_LOG"
fi
