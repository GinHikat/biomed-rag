#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$SCRIPT_DIR"

cd "$REPO_ROOT"

echo "Starting vLLM LLM server..."
./start_llm_server.sh &
LLM_PID=$!

if [[ "${START_EMBED_SERVER:-1}" == "1" ]]; then
  echo "Starting vLLM embedding server..."
  ./start_embed_server.sh &
  EMBED_PID=$!
fi

echo "Starting LightRAG server..."
./start_lightrag_server.sh &
LIGHTRAG_PID=$!

cleanup() {
  set +e
  echo
  echo "Stopping services..."

  if [[ -n "${LIGHTRAG_PID:-}" ]] && kill -0 "$LIGHTRAG_PID" >/dev/null 2>&1; then
    kill "$LIGHTRAG_PID" >/dev/null 2>&1 || true
    wait "$LIGHTRAG_PID" 2>/dev/null || true
  fi

  if [[ -n "${EMBED_PID:-}" ]] && kill -0 "$EMBED_PID" >/dev/null 2>&1; then
    kill "$EMBED_PID" >/dev/null 2>&1 || true
    wait "$EMBED_PID" 2>/dev/null || true
  fi

  if [[ -n "${LLM_PID:-}" ]] && kill -0 "$LLM_PID" >/dev/null 2>&1; then
    kill "$LLM_PID" >/dev/null 2>&1 || true
    wait "$LLM_PID" 2>/dev/null || true
  fi

  echo "All services stopped."
}

trap cleanup EXIT INT TERM

echo "All 3 servers are starting up in the background."
echo "Press Ctrl+C to stop all services."
echo "To view logs, you can run:"
echo "  tail -f logs/vllm_llm.log"
echo "  tail -f logs/vllm_embed.log"
echo "  tail -f logs/lightrag.log"

# Wait for LightRAG server to finish (or Ctrl+C to trigger cleanup)
wait "$LIGHTRAG_PID"
