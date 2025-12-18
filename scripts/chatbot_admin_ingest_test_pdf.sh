#!/usr/bin/env bash
set -euo pipefail

# Admin ingestion helper for the chatbot backend profile.
# Ingests `test/test_pdf.pdf` into the shared document library owner used by Chatbot SSE v2.
#
# Usage:
#   ./scripts/chatbot_admin_ingest_test_pdf.sh
#
# Optional overrides (must match the backend process env to ensure the same storage/index is used):
#   CHATBOT_SHARED_DOCUMENT_OWNER_ID=... ./scripts/chatbot_admin_ingest_test_pdf.sh
#   CHATBOT_KNOWLEDGE_CONFIG_PATH=... ./scripts/chatbot_admin_ingest_test_pdf.sh
#   POSTGRES_HOST=... POSTGRES_PORT=... ./scripts/chatbot_admin_ingest_test_pdf.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Avoid uv using a host-level cache that may have permission issues.
export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/.uv-cache}"

# Keep defaults aligned with `scripts/start_chatbot_backend.sh` (override via env if needed).
export CHATBOT_SHARED_DOCUMENT_OWNER_ID="${CHATBOT_SHARED_DOCUMENT_OWNER_ID:-00000000-0000-0000-0000-000000000001}"
CHATBOT_PROFILE="${CHATBOT_PROFILE:-full}"
CHATBOT_LLM_PROFILE="${CHATBOT_LLM_PROFILE:-echo}"
if [ "$CHATBOT_PROFILE" = "full" ]; then
  export CHATBOT_KNOWLEDGE_CONFIG_PATH="${CHATBOT_KNOWLEDGE_CONFIG_PATH:-config/json_configs/chatbot_test_full/knowledge.json}"
  if [ "$CHATBOT_LLM_PROFILE" = "openai" ]; then
    export CHATBOT_RAG_INFERENCE_CONFIG_PATH="${CHATBOT_RAG_INFERENCE_CONFIG_PATH:-config/json_configs/chatbot_test_full/rag_inference_openai.json}"
  else
    export CHATBOT_RAG_INFERENCE_CONFIG_PATH="${CHATBOT_RAG_INFERENCE_CONFIG_PATH:-config/json_configs/chatbot_test_full/rag_inference_echo.json}"
  fi
elif [ "$CHATBOT_PROFILE" = "test_pdf" ]; then
  export CHATBOT_KNOWLEDGE_CONFIG_PATH="${CHATBOT_KNOWLEDGE_CONFIG_PATH:-config/json_configs/chatbot_test_pdf/knowledge.json}"
  if [ "$CHATBOT_LLM_PROFILE" = "openai" ]; then
    export CHATBOT_RAG_INFERENCE_CONFIG_PATH="${CHATBOT_RAG_INFERENCE_CONFIG_PATH:-config/json_configs/chatbot_test_pdf/rag_inference_openai.json}"
  else
    export CHATBOT_RAG_INFERENCE_CONFIG_PATH="${CHATBOT_RAG_INFERENCE_CONFIG_PATH:-config/json_configs/chatbot_test_pdf/rag_inference.json}"
  fi
else
  export CHATBOT_RAG_INFERENCE_CONFIG_PATH="${CHATBOT_RAG_INFERENCE_CONFIG_PATH:-config/json_configs/chatbot_test/rag_inference.json}"
  export CHATBOT_KNOWLEDGE_CONFIG_PATH="${CHATBOT_KNOWLEDGE_CONFIG_PATH:-config/json_configs/chatbot_test/knowledge.json}"
fi
export RAG_INFERENCE_CONFIG_PATH="${RAG_INFERENCE_CONFIG_PATH:-$CHATBOT_RAG_INFERENCE_CONFIG_PATH}"
export KNOWLEDGE_CONFIG_PATH="${KNOWLEDGE_CONFIG_PATH:-$CHATBOT_KNOWLEDGE_CONFIG_PATH}"

# PostgreSQL defaults used by the chatbot_test configs (override via env / .env).
export POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
export POSTGRES_PORT="${POSTGRES_PORT:-5555}"
export POSTGRES_USER="${POSTGRES_USER:-postgres}"
export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-123}"
export POSTGRES_DB="${POSTGRES_DB:-rag_arc}"

pdf_path="test/test_pdf.pdf"
if [ ! -f "$pdf_path" ]; then
  echo "missing file: $pdf_path" >&2
  exit 1
fi

python - <<'PY'
import json
import os
from pathlib import Path

cfg = os.getenv("CHATBOT_KNOWLEDGE_CONFIG_PATH") or ""
path = Path(cfg)
if not path.exists():
    raise SystemExit(f"CHATBOT_KNOWLEDGE_CONFIG_PATH not found: {cfg}")

doc = json.loads(path.read_text(encoding="utf-8"))
parser = (doc.get("index_manager_config") or {}).get("parser_config") or {}
has_ocr = "ocr_parser" in parser and parser["ocr_parser"] is not None
if not has_ocr:
    raise SystemExit(
        "Current knowledge config does not configure an OCR parser, so PDF ingest will fail.\n"
        f"- CHATBOT_KNOWLEDGE_CONFIG_PATH={cfg}\n"
        "Fix: restart backend + run this script with a PDF-capable config, e.g.:\n"
        "  CHATBOT_KNOWLEDGE_CONFIG_PATH=config/json_configs/knowledge.json \\\n"
        "  CHATBOT_RAG_INFERENCE_CONFIG_PATH=config/json_configs/rag_inference.json \\\n"
        "  ./scripts/start_chatbot_backend.sh\n"
    )
PY

echo "Ingesting PDF into shared owner: $CHATBOT_SHARED_DOCUMENT_OWNER_ID"
echo "Using config:"
echo "  CHATBOT_KNOWLEDGE_CONFIG_PATH=$CHATBOT_KNOWLEDGE_CONFIG_PATH"
echo "  POSTGRES_HOST=$POSTGRES_HOST POSTGRES_PORT=$POSTGRES_PORT POSTGRES_DB=$POSTGRES_DB"

uv run rag-arc ingest-file "$pdf_path" --owner-id "$CHATBOT_SHARED_DOCUMENT_OWNER_ID"
