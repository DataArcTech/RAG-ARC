#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export LOG_LEVEL="${LOG_LEVEL:-WARNING}"
export RAG_INFERENCE_CONFIG_PATH="config/json_configs/chatbot_test/rag_inference.json"
export KNOWLEDGE_CONFIG_PATH="config/json_configs/chatbot_test/knowledge.json"
export CHATBOT_RAG_INFERENCE_CONFIG_PATH="config/json_configs/chatbot_test/rag_inference.json"
export CHATBOT_KNOWLEDGE_CONFIG_PATH="config/json_configs/chatbot_test/knowledge.json"
export LOCAL_FILE_STORAGE_PATH="./test_output/chatbot_local_files"
export CHATBOT_MAX_CONCURRENCY="${CHATBOT_MAX_CONCURRENCY:-16}"
export CHATBOT_MAX_CONTEXT_TOKENS="${CHATBOT_MAX_CONTEXT_TOKENS:-512}"
export CHATBOT_MAX_CONTEXT_FRACTION="${CHATBOT_MAX_CONTEXT_FRACTION:-0.9}"
export CHATBOT_CONTEXT_TURNS="${CHATBOT_CONTEXT_TURNS:-5}"
export CHATBOT_TOP_SOURCES="${CHATBOT_TOP_SOURCES:-5}"

# Make deepsearch registration fail fast (not part of this MVP e2e).
export DEEPSEARCH_WEB_PROVIDER=""
export DEEPSEARCH_DEFAULT_ADAPTER=""
export ENABLE_DEEPSEARCH="0"

rm -rf test_output/chatbot_file_store \
  test_output/chatbot_parsed_content_store \
  test_output/chatbot_chunk_store \
  test_output/chatbot_parsed_files \
  test_output/chatbot_bm25_index \
  test_output/chatbot_local_files || true
mkdir -p test_output \
  test_output/chatbot_file_store \
  test_output/chatbot_parsed_content_store \
  test_output/chatbot_chunk_store \
  test_output/chatbot_parsed_files \
  test_output/chatbot_bm25_index \
  test_output/chatbot_local_files

OWNER_A="$(python -c 'import uuid; print(uuid.uuid4())')"
OWNER_B="$(python -c 'import uuid; print(uuid.uuid4())')"
OWNER_C="$(python -c 'import uuid; print(uuid.uuid4())')"
ADMIN_OWNER="${CHATBOT_SHARED_DOCUMENT_OWNER_ID:-00000000-0000-0000-0000-000000000001}"

python - <<PY
import uuid
from datetime import datetime
from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
from encapsulation.data_model.orm_models import User

db = PostgreSQLConfig().build()
now = datetime.now(tz=datetime.now().astimezone().tzinfo)
owners = [uuid.UUID("$OWNER_A"), uuid.UUID("$OWNER_B"), uuid.UUID("$OWNER_C")]
owners.append(uuid.UUID("$ADMIN_OWNER"))
with db.SessionMaker() as session:
    for owner in owners:
        existing = session.query(User).filter_by(id=owner).first()
        if existing is None:
            session.add(User(id=owner, user_name=f"chatbot_curl_{str(owner)[:8]}", hashed_password="x", created_at=now, updated_at=now))
    session.commit()
PY

DOCS_DIR="test_output/chatbot_curl_docs"
rm -rf "$DOCS_DIR"
mkdir -p "$DOCS_DIR/a" "$DOCS_DIR/b" "$DOCS_DIR/c"

cat > "$DOCS_DIR/a/doc_a.txt" <<'EOF'
alphaA only.
Paris is the capital of France.
EOF

cat > "$DOCS_DIR/b/doc_b.txt" <<'EOF'
betaB only.
Berlin is the capital of Germany.
EOF

cat > "$DOCS_DIR/c/doc_c.txt" <<'EOF'
alphaA and betaB together.
EOF

uv run rag-arc ingest-folder "$DOCS_DIR/a" --owner-id "$ADMIN_OWNER" >/dev/null
uv run rag-arc ingest-folder "$DOCS_DIR/b" --owner-id "$ADMIN_OWNER" >/dev/null
uv run rag-arc ingest-folder "$DOCS_DIR/c" --owner-id "$ADMIN_OWNER" >/dev/null

PORT="${CHATBOT_E2E_PORT:-8099}"
UVICORN_PID=""
cleanup() {
  if [[ -n "${UVICORN_PID}" ]]; then
    kill "${UVICORN_PID}" >/dev/null 2>&1 || true
    wait "${UVICORN_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

uv run uvicorn main:app --host 127.0.0.1 --port "$PORT" --log-level warning >/tmp/chatbot_uvicorn.log 2>&1 &
UVICORN_PID="$!"

for _ in $(seq 1 60); do
  if curl -sS "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
    break
  fi
  sleep 0.2
done

echo "[curl-e2e] server up: http://127.0.0.1:$PORT"

REQ_DIR="$(mktemp -d)"
_post_sse() {
  local owner="$1"
  local conv="$2"
  local content="$3"
  local messages_json="$4"
  local out="$5"

  python - "$conv" "$content" "$messages_json" > "$REQ_DIR/request.json" <<'PY'
import json,sys
conv=sys.argv[1]
content=sys.argv[2]
messages=json.loads(sys.argv[3])
req={"id":conv,"content":content,"messages":messages,"stream":True}
print(json.dumps(req, ensure_ascii=False))
PY

  curl -sS -N -X POST "http://127.0.0.1:$PORT/api/messages" \
    -H 'Content-Type: application/json' \
    -H 'Accept: text/event-stream' \
    -H "X-Owner-Id: $owner" \
    -d @"$REQ_DIR/request.json" \
    > "$out"
}

_assert_sse_success() {
  local resp="$1"
  python - "$resp" >/dev/null <<'PY'
import json,sys
events=[]
for raw in open(sys.argv[1],encoding="utf-8",errors="replace"):
    raw=raw.strip()
    if not raw.startswith("data: "):
        continue
    payload=raw[len("data: "):].strip()
    if payload=="[DONE]":
        break
    events.append(json.loads(payload))
assert any(e.get("type")=="chunk" for e in events), "missing chunk"
assert any(e.get("type")=="sources" for e in events), "missing sources"
done=[e for e in events if e.get("type")=="done"]
assert done and done[-1].get("status")=="success", f"done={done[-1] if done else None}"
PY
}

_extract_sse_answer_and_sources() {
  local resp="$1"
  local out_answer="$2"
  local out_sources="$3"
  python - "$resp" "$out_answer" "$out_sources" <<'PY'
import json,sys
events=[]
for raw in open(sys.argv[1],encoding="utf-8",errors="replace"):
    raw=raw.strip()
    if not raw.startswith("data: "):
        continue
    payload=raw[len("data: "):].strip()
    if payload=="[DONE]":
        break
    events.append(json.loads(payload))
chunks=[e.get("content","") for e in events if e.get("type")=="chunk"]
answer="".join(chunks)
sources=[e for e in events if e.get("type")=="sources"]
src = sources[0]["sources"] if sources else []
open(sys.argv[2],"w",encoding="utf-8").write(answer)
open(sys.argv[3],"w",encoding="utf-8").write(json.dumps(src,ensure_ascii=False))
PY
}

RESP_A1="$REQ_DIR/a1.sse.txt"
RESP_A2="$REQ_DIR/a2.sse.txt"
RESP_B1="$REQ_DIR/b1.sse.txt"
ANSWER_A1="$REQ_DIR/a1.answer.txt"
SOURCES_A1="$REQ_DIR/a1.sources.json"

CONV_A="$(python -c 'import uuid; print(uuid.uuid4())')"
_post_sse "$OWNER_A" "$CONV_A" "alphaA" "[]" "$RESP_A1"
_assert_sse_success "$RESP_A1"
_extract_sse_answer_and_sources "$RESP_A1" "$ANSWER_A1" "$SOURCES_A1"

python - "$ANSWER_A1" "$SOURCES_A1" >/dev/null <<'PY'
import json,sys
answer=open(sys.argv[1],encoding="utf-8").read()
sources=json.load(open(sys.argv[2],encoding="utf-8"))
assert sources and sources[0].get("key")==1
assert "<sup>1</sup>" in answer, "expected <sup> marker"
assert sources[0].get("description"), "expected chunk text"
PY

python - "$SOURCES_A1" > "$REQ_DIR/a1.file_url.txt" <<'PY'
import json,sys
sources=json.load(open(sys.argv[1],encoding="utf-8"))
print(sources[0].get("file") or "")
PY

FILE_URL="$(cat "$REQ_DIR/a1.file_url.txt")"
if [[ -n "$FILE_URL" ]]; then
  if [[ "$FILE_URL" == /* ]]; then
    curl -sS "http://127.0.0.1:$PORT$FILE_URL" >/dev/null
  elif [[ "$FILE_URL" == file://* ]]; then
    FILE_PATH="${FILE_URL#file://}"
    test -f "$FILE_PATH"
    test -s "$FILE_PATH"
  else
    curl -sS "$FILE_URL" >/dev/null
  fi
fi

python - "$RESP_A1" >/dev/null <<'PY'
import json,sys
events=[]
for raw in open(sys.argv[1],encoding="utf-8",errors="replace"):
    raw=raw.strip()
    if not raw.startswith("data: "):
        continue
    payload=raw[len("data: "):].strip()
    if payload=="[DONE]":
        break
    events.append(json.loads(payload))
assert any(e.get("type")=="title" and e.get("title") for e in events), "expected title on first turn"
PY

python - "$ANSWER_A1" > "$REQ_DIR/a1.messages.json" <<'PY'
import json,sys
ans=open(sys.argv[1],encoding="utf-8").read()
msgs=[{"role":"user","content":"alphaA"},{"role":"assistant","content":ans}]
print(json.dumps(msgs,ensure_ascii=False))
PY

RESP_A2_ANSWER="$REQ_DIR/a2.answer.txt"
RESP_A2_SOURCES="$REQ_DIR/a2.sources.json"
_post_sse "$OWNER_A" "$CONV_A" "Answer again in one sentence." "$(cat "$REQ_DIR/a1.messages.json")" "$RESP_A2"
_assert_sse_success "$RESP_A2"
_extract_sse_answer_and_sources "$RESP_A2" "$RESP_A2_ANSWER" "$RESP_A2_SOURCES"

CONV_B="$(python -c 'import uuid; print(uuid.uuid4())')"
_post_sse "$OWNER_B" "$CONV_B" "betaB" "[]" "$RESP_B1"
_assert_sse_success "$RESP_B1"

START_MS="$(python -c 'import time; print(int(time.time()*1000))')"
_post_sse "$OWNER_A" "$(python -c 'import uuid; print(uuid.uuid4())')" "alphaA" "[]" "$REQ_DIR/p1.sse.txt" &
P1=$!
_post_sse "$OWNER_B" "$(python -c 'import uuid; print(uuid.uuid4())')" "betaB" "[]" "$REQ_DIR/p2.sse.txt" &
P2=$!
wait "$P1" "$P2"
END_MS="$(python -c 'import time; print(int(time.time()*1000))')"
ELAPSED_MS="$((END_MS-START_MS))"
if [[ "$ELAPSED_MS" -gt 700 ]]; then
  echo "expected concurrent chats (<700ms), took ${ELAPSED_MS}ms"
  exit 1
fi

# Long conversation under one conversation_id should keep working with front-end full history
# while backend only uses the last 5 turns.
LONG_CONV="$(python -c 'import uuid; print(uuid.uuid4())')"
HIST="$REQ_DIR/long.messages.json"
echo '[]' > "$HIST"
for i in $(seq 1 16); do
  MSG="${i}:$(python -c 'print("m"*60)')"
  RESP="$REQ_DIR/long.$i.sse.txt"
  _post_sse "$OWNER_A" "$LONG_CONV" "$MSG" "$(cat "$HIST")" "$RESP"
  _assert_sse_success "$RESP"
  python - "$HIST" "$MSG" "$RESP" > "$REQ_DIR/long.messages.next.json" <<'PY'
import json,sys
hist=json.load(open(sys.argv[1],encoding="utf-8"))
user=sys.argv[2]
events=[]
for raw in open(sys.argv[3],encoding="utf-8",errors="replace"):
    raw=raw.strip()
    if not raw.startswith("data: "):
        continue
    payload=raw[len("data: "):].strip()
    if payload=="[DONE]":
        break
    events.append(json.loads(payload))
chunks=[e.get("content","") for e in events if e.get("type")=="chunk"]
assistant="".join(chunks)
hist.append({"role":"user","content":user})
hist.append({"role":"assistant","content":assistant})
print(json.dumps(hist,ensure_ascii=False))
PY
  mv "$REQ_DIR/long.messages.next.json" "$HIST"
done

# Context-too-long should return SSE error without killing the server.
TOO_LONG_CONV="$(python -c 'import uuid; print(uuid.uuid4())')"
python - <<'PY' > "$REQ_DIR/too_long.messages.json"
import json
huge="x"*2000
messages=[{"role":"user","content":huge} for _ in range(10)]
print(json.dumps(messages))
PY
_post_sse "$OWNER_A" "$TOO_LONG_CONV" "hi" "$(cat "$REQ_DIR/too_long.messages.json")" "$REQ_DIR/too_long.sse.txt" || true
python - "$REQ_DIR/too_long.sse.txt" >/dev/null <<'PY'
import json,sys
events=[]
for raw in open(sys.argv[1],encoding="utf-8",errors="replace"):
    raw=raw.strip()
    if not raw.startswith("data: "):
        continue
    payload=raw[len("data: "):].strip()
    if payload=="[DONE]":
        break
    events.append(json.loads(payload))
err=[e for e in events if e.get("type")=="error"]
assert err, f"expected error, got {events[:3]}"
assert err[0].get("code")==413, err[0]
PY

echo "[curl-e2e] SSE streaming OK"

echo "[curl-e2e] OK (owners: a=$OWNER_A b=$OWNER_B c=$OWNER_C admin=$ADMIN_OWNER)"
