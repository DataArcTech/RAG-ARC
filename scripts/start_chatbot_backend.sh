#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Chatbot MVP defaults (override via env).
export MODEL_PROFILE="${MODEL_PROFILE:-api}"

# Disable DeepSearch for this chatbot backend profile.
export ENABLE_DEEPSEARCH="${ENABLE_DEEPSEARCH:-0}"
export DEEPSEARCH_WEB_PROVIDER="${DEEPSEARCH_WEB_PROVIDER:-}"
export DEEPSEARCH_DEFAULT_ADAPTER="${DEEPSEARCH_DEFAULT_ADAPTER:-}"

# Identity / shared document library.
export CHATBOT_SHARED_DOCUMENT_OWNER_ID="${CHATBOT_SHARED_DOCUMENT_OWNER_ID:-00000000-0000-0000-0000-000000000001}"

# Chatbot concurrency & context window.
export CHATBOT_MAX_CONCURRENCY="${CHATBOT_MAX_CONCURRENCY:-16}"
export CHATBOT_CONTEXT_TURNS="${CHATBOT_CONTEXT_TURNS:-5}"
export CHATBOT_MAX_CONTEXT_TOKENS="${CHATBOT_MAX_CONTEXT_TOKENS:-8192}"
export CHATBOT_CONVERSATION_LOCK_TIMEOUT_S="${CHATBOT_CONVERSATION_LOCK_TIMEOUT_S:-30}"
export CHATBOT_GLOBAL_SEMAPHORE_TIMEOUT_S="${CHATBOT_GLOBAL_SEMAPHORE_TIMEOUT_S:-30}"

# Config selection:
# - Default to chatbot_test configs so the service can run without external LLM keys (uses echo_chat).
# - You can override to point at production configs if desired.
export CHATBOT_RAG_INFERENCE_CONFIG_PATH="${CHATBOT_RAG_INFERENCE_CONFIG_PATH:-config/json_configs/chatbot_test/rag_inference.json}"
export CHATBOT_KNOWLEDGE_CONFIG_PATH="${CHATBOT_KNOWLEDGE_CONFIG_PATH:-config/json_configs/chatbot_test/knowledge.json}"
export RAG_INFERENCE_CONFIG_PATH="${RAG_INFERENCE_CONFIG_PATH:-$CHATBOT_RAG_INFERENCE_CONFIG_PATH}"
export KNOWLEDGE_CONFIG_PATH="${KNOWLEDGE_CONFIG_PATH:-$CHATBOT_KNOWLEDGE_CONFIG_PATH}"

# Knowledge storage paths (only needed when using local file stores).
export LOCAL_FILE_STORAGE_PATH="${LOCAL_FILE_STORAGE_PATH:-./test_output/chatbot_local_files}"
mkdir -p "$LOCAL_FILE_STORAGE_PATH" \
  test_output/chatbot_file_store \
  test_output/chatbot_parsed_content_store \
  test_output/chatbot_chunk_store \
  test_output/chatbot_parsed_files \
  test_output/chatbot_bm25_index >/dev/null 2>&1 || true

# PostgreSQL defaults for chatbot_test knowledge config (override via env / .env).
export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
export POSTGRES_PORT="${POSTGRES_PORT:-5555}"
export POSTGRES_USER="${POSTGRES_USER:-postgres}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-123}"
export POSTGRES_DB="${POSTGRES_DB:-rag_arc}"

# Uvicorn
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
LOG_LEVEL="${UVICORN_LOG_LEVEL:-info}"

exec uv run uvicorn main:app --host "$HOST" --port "$PORT" --log-level "$LOG_LEVEL"

