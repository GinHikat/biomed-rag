#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"

# vLLM embedding server
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_EMBED_PORT="${VLLM_EMBED_PORT:-8081}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-ai/nomic-embed-text-v1.5}"
VLLM_EMBED_GPU_MEM_UTIL="${VLLM_EMBED_GPU_MEM_UTIL:-0.15}"
# Device for embedding server: 'gpu' (default) or 'cpu'
EMBED_DEVICE="${EMBED_DEVICE:-gpu}"

mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

EMBED_LOG="$LOG_DIR/vllm_embed.log"

echo "Starting vLLM embedding server on port $VLLM_EMBED_PORT (device: $EMBED_DEVICE) ..."
if [[ "$EMBED_DEVICE" == "cpu" ]]; then
  CUDA_VISIBLE_DEVICES="" python -m vllm.entrypoints.openai.api_server \
    --model "$EMBEDDING_MODEL" \
    --runner pooling \
    --host "$VLLM_HOST" \
    --port "$VLLM_EMBED_PORT" \
    --dtype float32 \
    2>&1 | tee "$EMBED_LOG"
else
  python -m vllm.entrypoints.openai.api_server \
    --model "$EMBEDDING_MODEL" \
    --runner pooling \
    --host "$VLLM_HOST" \
    --port "$VLLM_EMBED_PORT" \
    --gpu-memory-utilization "$VLLM_EMBED_GPU_MEM_UTIL" \
    2>&1 | tee "$EMBED_LOG"
fi
