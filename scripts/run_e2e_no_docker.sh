#!/usr/bin/env bash

set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8000}"
API_PORT="${API_PORT:-8000}"
START_SERVER="${START_SERVER:-1}"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="local/e2e/no_docker_${RUN_ID}"
mkdir -p "$RUN_DIR"

E2E_POSTGRES_DB="${E2E_POSTGRES_DB:-rag_arc_e2e_${RUN_ID}}"
E2E_REDIS_DB="${E2E_REDIS_DB:-15}"

SERVER_PID=""

function cleanup() {
  if [ -n "${SERVER_PID}" ]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

echo "E2E run dir: $RUN_DIR"
echo "API_BASE=$API_BASE"

if [ "$START_SERVER" = "1" ]; then
  if curl -sS -m 2 "$API_BASE/" >/dev/null 2>&1; then
    echo "API already running; not starting uvicorn."
  else
    echo "Starting uvicorn (no docker) ..."
    mkdir -p "$RUN_DIR/runtime" "$RUN_DIR/files" "$RUN_DIR/parsed_files" "$RUN_DIR/deepsearch_runs" "$RUN_DIR/deepsearch_artifacts"
    POSTGRES_DB="$E2E_POSTGRES_DB" \
    REDIS_DB="$E2E_REDIS_DB" \
    RAGARC_RUNTIME_DIR="$RUN_DIR/runtime" \
    LOCAL_FILE_STORAGE_PATH="$RUN_DIR/files" \
    PARSER_OUTPUT_DIR="$RUN_DIR/parsed_files" \
    BM25_INDEX_PATH="$RUN_DIR/bm25_index" \
    FAISS_INDEX_PATH="$RUN_DIR/faiss_index" \
    GRAPH_STORAGE_PATH="$RUN_DIR/graph_index" \
    GRAPH_INDEX_NAME="e2e_${RUN_ID}" \
    DEEPSEARCH_PLAN_OUTPUT_DIR="$RUN_DIR/deepsearch_runs" \
    DEEPSEARCH_TOOL_ARTIFACT_DIR="$RUN_DIR/deepsearch_artifacts" \
    MODEL_PROFILE="${MODEL_PROFILE:-api}" \
    nohup uv run uvicorn main:app --host 0.0.0.0 --port "$API_PORT" >"$RUN_DIR/uvicorn.log" 2>&1 &
    SERVER_PID="$!"
    echo "uvicorn pid: $SERVER_PID"
    READY=0
    for i in $(seq 1 90); do
      if curl -sS -m 2 "$API_BASE/" | grep -q "ok"; then
        echo "API is ready."
        READY=1
        break
      fi
      sleep 1
    done
    if [ "$READY" != "1" ]; then
      echo "❌ API failed to become ready; see $RUN_DIR/uvicorn.log"
      exit 1
    fi
  fi
fi

export API_BASE

echo "Running session comprehensive..."
bash test/api/session_comprehensive_test.sh >"$RUN_DIR/session.log" 2>&1

echo "Running knowledge comprehensive..."
bash test/api/knowledge_comprehensive_test.sh >"$RUN_DIR/knowledge.log" 2>&1

echo "Running user isolation comprehensive..."
bash test/api/user_isolation_comprehensive_test.sh >"$RUN_DIR/user_isolation.log" 2>&1

echo "Running stream_chat SSE comprehensive..."
bash test/api/stream_chat_comprehensive_test.sh >"$RUN_DIR/stream_chat_sse.log" 2>&1

echo "Running stream_chat WS comprehensive..."
uv run python test/api/stream_chat_ws_comprehensive_test.py >"$RUN_DIR/stream_chat_ws.log" 2>&1

echo "Running MCP comprehensive..."
bash test/api/mcp_comprehensive_test.sh >"$RUN_DIR/mcp.log" 2>&1

echo "Running DeepSearch comprehensive..."
bash test/api/deepsearch_comprehensive_test.sh >"$RUN_DIR/deepsearch.log" 2>&1

echo "Running concurrency probe..."
uv run python scripts/run_concurrency_e2e.py >"$RUN_DIR/concurrency.log" 2>&1

echo "E2E suite completed successfully."
