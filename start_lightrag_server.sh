#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"

# LightRAG server
LIGHTRAG_HOST="${LIGHTRAG_HOST:-0.0.0.0}"
LIGHTRAG_PORT="${LIGHTRAG_PORT:-9621}"
LIGHTRAG_WORKING_DIR="${LIGHTRAG_WORKING_DIR:-$REPO_ROOT/rag_storage}"
LIGHTRAG_INPUT_DIR="${LIGHTRAG_INPUT_DIR:-$REPO_ROOT/inputs}"

VLLM_LLM_PORT="${VLLM_LLM_PORT:-8080}"
VLLM_EMBED_PORT="${VLLM_EMBED_PORT:-8081}"

LLM_MODEL="${LLM_MODEL:-BioMistral/BioMistral-7B-AWQ-QGS128-W4-GEMM}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-nomic-ai/nomic-embed-text-v1.5}"

export LLM_BINDING="openai"
export EMBEDDING_BINDING="openai"
export LLM_BINDING_HOST="http://127.0.0.1:${VLLM_LLM_PORT}/v1"
export LLM_BINDING_API_KEY="none"
export LLM_MODEL="$LLM_MODEL"

export EMBEDDING_BINDING_HOST="http://127.0.0.1:${VLLM_EMBED_PORT}/v1"
export EMBEDDING_BINDING_API_KEY="none"
export EMBEDDING_MODEL="$EMBEDDING_MODEL"

mkdir -p "$LOG_DIR" "$LIGHTRAG_WORKING_DIR" "$LIGHTRAG_INPUT_DIR"
cd "$REPO_ROOT"

if [[ ! -f ".env" ]]; then
  touch .env
fi

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Missing required command: $1" >&2
    exit 1
  fi
}
require_cmd lightrag-server
require_cmd curl

wait_for_http() {
  local url="$1"
  local name="$2"
  local retries="${3:-120}"
  local sleep_sec="${4:-1}"
  local i

  for ((i = 1; i <= retries; i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$name is ready: $url"
      return 0
    fi
    sleep "$sleep_sec"
  done

  echo "Timed out waiting for $name at $url" >&2
  return 1
}

# Wait for both servers to be ready before starting LightRAG
echo "Waiting for vLLM servers to be ready..."
wait_for_http "http://127.0.0.1:${VLLM_LLM_PORT}/v1/models" "vLLM LLM"
wait_for_http "http://127.0.0.1:${VLLM_EMBED_PORT}/v1/models" "vLLM embedding"

LIGHTRAG_LOG="$LOG_DIR/lightrag.log"
echo "Starting LightRAG server on port $LIGHTRAG_PORT ..."
lightrag-server \
  --host "$LIGHTRAG_HOST" \
  --port "$LIGHTRAG_PORT" \
  --working-dir "$LIGHTRAG_WORKING_DIR" \
  --input-dir "$LIGHTRAG_INPUT_DIR" \
  --llm-binding openai \
  --embedding-binding openai \
  2>&1 | tee "$LIGHTRAG_LOG"
