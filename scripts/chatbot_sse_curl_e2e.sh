#!/usr/bin/env bash
set -euo pipefail

# End-to-end (CLI ingest -> SSE chat -> file open) using curl.
#
# Prereqs:
# - DB services running (PostgreSQL/Redis/Neo4j if enabled by your config).
# - API server running, e.g.:
#     uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
#
# Usage:
#   BASE_URL=http://localhost:8000 ./scripts/chatbot_sse_curl_e2e.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Optional: load local env so CLI + server use consistent config.
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Align CLI config with the chatbot backend if provided.
if [ -n "${CHATBOT_RAG_INFERENCE_CONFIG_PATH:-}" ]; then
  export RAG_INFERENCE_CONFIG_PATH="${RAG_INFERENCE_CONFIG_PATH:-$CHATBOT_RAG_INFERENCE_CONFIG_PATH}"
fi
if [ -n "${CHATBOT_KNOWLEDGE_CONFIG_PATH:-}" ]; then
  export KNOWLEDGE_CONFIG_PATH="${KNOWLEDGE_CONFIG_PATH:-$CHATBOT_KNOWLEDGE_CONFIG_PATH}"
fi

BASE_URL="${BASE_URL:-http://localhost:8000}"
OWNER_ID="${CHATBOT_SHARED_DOCUMENT_OWNER_ID:-00000000-0000-0000-0000-000000000001}"

tmp_dir="$(mktemp -d)"
cookie_jar="$(mktemp)"
sse_log="$(mktemp)"
sse_log_nonsense="$(mktemp)"
cleanup() {
  rm -rf "$tmp_dir" "$cookie_jar" "$sse_log" "$sse_log_nonsense"
}
trap cleanup EXIT

echo "[1/4] Prepare shared docs via CLI (owner_id=$OWNER_ID)"
cat >"$tmp_dir/facts.txt" <<'EOF'
Paris is the capital of France.
EOF
uv run rag-arc ingest-folder "$tmp_dir" --pattern '*.txt' --no-recursive --owner-id "$OWNER_ID"

echo "[2/4] Bootstrap cookie"
curl -sS -c "$cookie_jar" -b "$cookie_jar" "$BASE_URL/chatbot/bootstrap" >/dev/null

conversation_id="$(python - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"

echo "[3/5] SSE chat (conversation_id=$conversation_id)"
curl -sS -N \
  -c "$cookie_jar" -b "$cookie_jar" \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  --data "$(python - <<PY
import json
print(json.dumps({
  "id": "$conversation_id",
  "content": "What is the capital of France?",
  "messages": [],
  "stream": True,
}))
PY
)" \
  "$BASE_URL/api/messages" | tee "$sse_log" >/dev/null

conversation_id_nonsense="$(python - <<'PY'
import uuid
print(uuid.uuid4())
PY
)"

echo "[4/5] SSE chat (nonsense query sources must be citation-driven) (conversation_id=$conversation_id_nonsense)"
curl -sS -N \
  -c "$cookie_jar" -b "$cookie_jar" \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  --data "$(python - <<PY
import json
print(json.dumps({
  "id": "$conversation_id_nonsense",
  "content": "111111111111",
  "messages": [],
  "stream": True,
}))
PY
)" \
  "$BASE_URL/api/messages" | tee "$sse_log_nonsense" >/dev/null

python - <<'PY' "$sse_log_nonsense"
import json
import sys

path = sys.argv[1]
chunks = []
sources = None
with open(path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):].strip()
        if payload == "[DONE]":
            break
        obj = json.loads(payload)
        if obj.get("type") == "chunk":
            chunks.append(obj.get("content") or "")
        if obj.get("type") == "sources":
            sources = obj.get("sources")

joined = "".join(chunks)
if sources is None:
    raise SystemExit("missing sources event in nonsense response")
has_sup = "<sup>" in joined
if has_sup and not sources:
    raise SystemExit("answer contains <sup> but sources is empty")
if (not has_sup) and sources:
    raise SystemExit(f"answer contains no <sup> but sources is non-empty: {len(sources)}")
print("ok: nonsense query sources are citation-driven")
PY

echo "[5/5] Extract first sources + open file link"
python - <<'PY' "$BASE_URL" "$sse_log"
import json
import sys

base_url = sys.argv[1].rstrip("/")
path = sys.argv[2]

sources_event = None
title_event = None
chunks = []
with open(path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):].strip()
        if payload == "[DONE]":
            break
        obj = json.loads(payload)
        if obj.get("type") == "chunk":
            chunks.append(obj.get("content") or "")
        if obj.get("type") == "sources" and sources_event is None:
            sources_event = obj
        if obj.get("type") == "title" and title_event is None:
            title_event = obj

if title_event:
    print("title:", title_event.get("title"))
else:
    print("title: (missing)")

if not sources_event:
    raise SystemExit("sources: (missing)")

answer = "".join(chunks)
has_sup = "<sup>" in answer
sources = sources_event.get("sources") or []
if has_sup and not sources:
    raise SystemExit("answer contains <sup> but sources is empty")
if (not has_sup) and sources:
    raise SystemExit(f"answer contains no <sup> but sources is non-empty: {len(sources)}")
if not sources:
    print("sources: (empty, citation-driven)")
    raise SystemExit(0)

first = sources[0]
print("chunk_id:", first.get("chunk_id"))
print("file_id:", first.get("file_id"))
file_path = first.get("file")
print("file:", file_path)
if file_path:
    print("curl:", f"curl -sS {base_url}{file_path} | head")
PY
