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

BASE_URL="${BASE_URL:-http://localhost:8000}"
OWNER_ID="${CHATBOT_SHARED_DOCUMENT_OWNER_ID:-00000000-0000-0000-0000-000000000001}"

tmp_dir="$(mktemp -d)"
cookie_jar="$(mktemp)"
sse_log="$(mktemp)"
cleanup() {
  rm -rf "$tmp_dir" "$cookie_jar" "$sse_log"
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

echo "[3/4] SSE chat (conversation_id=$conversation_id)"
curl -sS -N \
  -c "$cookie_jar" -b "$cookie_jar" \
  -H 'Content-Type: application/json' \
  -H 'Accept: text/event-stream' \
  --data "$(python - <<PY
import json
print(json.dumps({
  \"id\": \"$conversation_id\",
  \"content\": \"What is the capital of France?\",
  \"messages\": [],
  \"stream\": True,
}))
PY
)" \
  "$BASE_URL/api/messages" | tee "$sse_log" >/dev/null

echo "[4/4] Extract first sources + open file link"
python - <<'PY' "$BASE_URL" "$sse_log"
import json
import sys

base_url = sys.argv[1].rstrip("/")
path = sys.argv[2]

sources_event = None
title_event = None
with open(path, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if not line.startswith("data: "):
            continue
        payload = line[len("data: "):].strip()
        if payload == "[DONE]":
            break
        obj = json.loads(payload)
        if obj.get("type") == "sources" and sources_event is None:
            sources_event = obj
        if obj.get("type") == "title" and title_event is None:
            title_event = obj

if title_event:
    print("title:", title_event.get("title"))
else:
    print("title: (missing)")

if not sources_event or not (sources_event.get("sources") or []):
    raise SystemExit("sources: (missing)")

first = sources_event["sources"][0]
print("chunk_id:", first.get("chunk_id"))
print("file_id:", first.get("file_id"))
file_path = first.get("file")
print("file:", file_path)
if file_path:
    print("curl:", f"curl -sS {base_url}{file_path} | head")
PY

