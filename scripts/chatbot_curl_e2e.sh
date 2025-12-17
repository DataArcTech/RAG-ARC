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
export CHATBOT_CONTEXT_TURNS="${CHATBOT_CONTEXT_TURNS:-5}"

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
RESP_A1="$REQ_DIR/a1.json"
RESP_A2="$REQ_DIR/a2.json"
RESP_B1="$REQ_DIR/b1.json"
REQ_A2="$REQ_DIR/a2.req.json"

curl -sS -X POST "http://127.0.0.1:$PORT/chatbot/chat" \
  -H 'Content-Type: application/json' \
  -H "X-Owner-Id: $OWNER_A" \
  -d "{\"conversation_id\":\"$(python -c 'import uuid; print(uuid.uuid4())')\",\"message\":{\"role\":\"user\",\"content\":\"alphaA\"},\"memory\":{\"version\":0,\"summary\":\"\",\"recent_messages\":[]},\"options\":{\"include_evidence\":true,\"top_k\":3,\"return_subgraph\":false,\"max_context_fraction\":0.9}}" \
  > "$RESP_A1"

python - "$RESP_A1" >/dev/null <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert "Sources: [1]" in d["assistant"]["content"]
PY

python - "$RESP_A1" > "$REQ_DIR/a_chunk_url.txt" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d["citations"], "expected citations"
print(d["citations"][0]["chunk_url"])
PY

python - "$RESP_A1" > "$REQ_DIR/a_file_url.txt" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(d["citations"][0]["file_url"])
PY

CHUNK_URL="$(cat "$REQ_DIR/a_chunk_url.txt")"
FILE_URL="$(cat "$REQ_DIR/a_file_url.txt")"

curl -sS -o /dev/null -w "%{http_code}" -H "X-Owner-Id: $OWNER_A" "http://127.0.0.1:$PORT$CHUNK_URL" | rg -q "^200$"
curl -sS -o /dev/null -w "%{http_code}" -H "X-Owner-Id: $OWNER_A" "http://127.0.0.1:$PORT$FILE_URL" | rg -q "^200$"
curl -sS -o /dev/null -w "%{http_code}" -H "X-Owner-Id: $OWNER_B" "http://127.0.0.1:$PORT$CHUNK_URL" | rg -q "^200$"
curl -sS -o /dev/null -w "%{http_code}" -H "X-Owner-Id: $OWNER_B" "http://127.0.0.1:$PORT$FILE_URL" | rg -q "^200$"
curl -sS -o /dev/null -w "%{http_code}" "http://127.0.0.1:$PORT$CHUNK_URL" | rg -q "^401$"

python - <<'PY' "$RESP_A1" > "$REQ_DIR/title.req.json"
import json,sys
d=json.load(open(sys.argv[1]))
req = {
  "conversation_id": d["conversation_id"],
  "user": "alphaA",
  "assistant": d["assistant"]["content"],
}
print(json.dumps(req, ensure_ascii=False))
PY

curl -sS -X POST "http://127.0.0.1:$PORT/chatbot/title" \
  -H 'Content-Type: application/json' \
  -H "X-Owner-Id: $OWNER_A" \
  -d @"$REQ_DIR/title.req.json" \
  | python -c 'import json,sys; d=json.load(sys.stdin); assert d.get("title")'

python - <<'PY' "$RESP_A1" "$REQ_A2"
import json,sys
d=json.load(open(sys.argv[1]))
req = {
  "conversation_id": d["conversation_id"],
  "message": {"role": "user", "content": "Answer again in one sentence."},
  "memory": d["memory"],
  "options": {"include_evidence": False, "top_k": 1, "return_subgraph": False, "max_context_fraction": 0.9},
}
open(sys.argv[2],"w",encoding="utf-8").write(json.dumps(req,ensure_ascii=False))
PY

curl -sS -X POST "http://127.0.0.1:$PORT/chatbot/chat" \
  -H 'Content-Type: application/json' \
  -H "X-Owner-Id: $OWNER_A" \
  -d @"$REQ_A2" \
  > "$RESP_A2"

python - <<'PY' "$RESP_A1" "$RESP_A2"
import json,sys
d1=json.load(open(sys.argv[1])); d2=json.load(open(sys.argv[2]))
assert d2["memory"]["version"] > d1["memory"]["version"]
PY

curl -sS -X POST "http://127.0.0.1:$PORT/chatbot/chat" \
  -H 'Content-Type: application/json' \
  -H "X-Owner-Id: $OWNER_B" \
  -d "{\"conversation_id\":\"$(python -c 'import uuid; print(uuid.uuid4())')\",\"message\":{\"role\":\"user\",\"content\":\"betaB\"},\"memory\":{\"version\":0,\"summary\":\"\",\"recent_messages\":[]},\"options\":{\"include_evidence\":true,\"top_k\":3,\"return_subgraph\":false,\"max_context_fraction\":0.9}}" \
  > "$RESP_B1"

python - "$RESP_B1" >/dev/null <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
assert d["citations"], "expected citations"
print("ok")
PY

START_MS="$(python -c 'import time; print(int(time.time()*1000))')"
curl -sS -X POST "http://127.0.0.1:$PORT/chatbot/chat" -H 'Content-Type: application/json' -H "X-Owner-Id: $OWNER_A" \
  -d "{\"conversation_id\":\"$(python -c 'import uuid; print(uuid.uuid4())')\",\"message\":{\"role\":\"user\",\"content\":\"alphaA\"},\"memory\":{\"version\":0,\"summary\":\"\",\"recent_messages\":[]},\"options\":{\"include_evidence\":false,\"top_k\":1,\"return_subgraph\":false,\"max_context_fraction\":0.9}}" \
  >/dev/null &
P1=$!
curl -sS -X POST "http://127.0.0.1:$PORT/chatbot/chat" -H 'Content-Type: application/json' -H "X-Owner-Id: $OWNER_B" \
  -d "{\"conversation_id\":\"$(python -c 'import uuid; print(uuid.uuid4())')\",\"message\":{\"role\":\"user\",\"content\":\"betaB\"},\"memory\":{\"version\":0,\"summary\":\"\",\"recent_messages\":[]},\"options\":{\"include_evidence\":false,\"top_k\":1,\"return_subgraph\":false,\"max_context_fraction\":0.9}}" \
  >/dev/null &
P2=$!
wait "$P1" "$P2"
END_MS="$(python -c 'import time; print(int(time.time()*1000))')"
ELAPSED_MS="$((END_MS-START_MS))"
if [[ "$ELAPSED_MS" -gt 700 ]]; then
  echo "expected concurrent chats (<700ms), took ${ELAPSED_MS}ms"
  exit 1
fi

# Long conversation under one conversation_id should keep working via last-5-turn window.
LONG_CONV="$(python -c 'import uuid; print(uuid.uuid4())')"
MEM_FILE="$REQ_DIR/long.mem.json"
echo '{"version":0,"summary":"","recent_messages":[]}' > "$MEM_FILE"
for i in $(seq 1 16); do
  python - "$MEM_FILE" "$LONG_CONV" "$i" > "$REQ_DIR/long.req.json" <<'PY'
import json,sys
mem=json.load(open(sys.argv[1]))
conv=sys.argv[2]
i=int(sys.argv[3])
msg=str(i)+":" + ("m"*60)
req={"conversation_id":conv,"message":{"role":"user","content":msg},"memory":mem,"options":{"include_evidence":False,"top_k":1,"return_subgraph":False,"max_context_fraction":0.9}}
print(json.dumps(req))
PY
  HTTP_CODE="$(curl -sS -o "$REQ_DIR/long.resp.json" -w "%{http_code}" -X POST "http://127.0.0.1:$PORT/chatbot/chat" -H 'Content-Type: application/json' -H "X-Owner-Id: $OWNER_A" -d @"$REQ_DIR/long.req.json")"
  if [[ "$HTTP_CODE" != "200" ]]; then
    echo "long conversation failed at turn $i (http=$HTTP_CODE):"
    cat "$REQ_DIR/long.resp.json" || true
    exit 1
  fi
  python - "$REQ_DIR/long.resp.json" "$MEM_FILE" <<'PY'
import json,sys
resp=json.load(open(sys.argv[1]))
open(sys.argv[2],'w',encoding='utf-8').write(json.dumps(resp['memory'], ensure_ascii=False))
PY
done
python - <<'PY' "$MEM_FILE" >/dev/null
import json,sys
mem=json.load(open(sys.argv[1]))
assert len(mem.get("recent_messages") or []) <= 10
PY

python - <<PY
import asyncio, json, websockets, uuid

async def main():
    conv = str(uuid.uuid4())
    uri = f"ws://127.0.0.1:$PORT/chatbot/ws?conversation_id={conv}&owner_id=$OWNER_C"
    async with websockets.connect(uri, additional_headers=[("X-Owner-Id", "$OWNER_C")]) as ws:
        await ws.send(json.dumps({
          "message": {"role": "user", "content": "stream betaB"},
          "memory": {"version": 0, "summary": "", "recent_messages": []},
          "options": {"include_evidence": False, "top_k": 1, "return_subgraph": False, "max_context_fraction": 0.9}
        }))
        start = json.loads(await ws.recv())
        assert start["type"] == "start"
        saw_delta = False
        saw_title = False
        for _ in range(200):
            frame = json.loads(await ws.recv())
            if frame["type"] == "delta":
                saw_delta = True
            if frame["type"] == "final":
                assert saw_delta
                assert frame["assistant"]["content"]
                break
        for _ in range(200):
            frame = json.loads(await ws.recv())
            if frame["type"] == "title":
                assert frame["title"]
                saw_title = True
                break
        assert saw_title, "expected title frame after first final"
        return
    raise RuntimeError("no final frame")

asyncio.run(main())
print("[curl-e2e] ws streaming OK")
PY

echo "[curl-e2e] OK (owners: a=$OWNER_A b=$OWNER_B c=$OWNER_C admin=$ADMIN_OWNER)"
