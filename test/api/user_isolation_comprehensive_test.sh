#!/usr/bin/env bash

set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8000}"
AUTH_ENDPOINT="$API_BASE/auth"
KNOWLEDGE_ENDPOINT="$API_BASE/knowledge"
RAG_ENDPOINT="$API_BASE/rag_inference"

echo "Testing User Isolation Comprehensive Flow at $API_BASE"
echo "======================================================"

# 0) Health check
curl -sS "$API_BASE/" | grep -q "ok" && echo "✅ Health check PASS" || { echo "❌ Health check failed"; exit 1; }

SUFFIX="$(date +%s)"
USER_A="isolation_a_${SUFFIX}"
USER_B="isolation_b_${SUFFIX}"
PASS="isolation_password"

function ensure_user_and_token() {
  local USERNAME="$1"
  local PASSWORD="$2"
  local REGISTER_STATUS

  REGISTER_STATUS=$(curl -sS -o /dev/null -w "%{http_code}" -X POST "$AUTH_ENDPOINT/register" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"$USERNAME\",\"user_name\":\"$USERNAME\",\"password\":\"$PASSWORD\"}" || true)

  if [ "$REGISTER_STATUS" != "201" ] && [ "$REGISTER_STATUS" != "400" ]; then
    echo "❌ Register failed for $USERNAME (status=$REGISTER_STATUS)"
    exit 1
  fi

  curl -sS -X POST "$AUTH_ENDPOINT/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=$USERNAME&password=$PASSWORD" | python3 -c "import json,sys; print(json.load(sys.stdin)['access_token'])"
}

echo -e "\n1) Create/login two users"
TOKEN_A="$(ensure_user_and_token "$USER_A" "$PASS")"
TOKEN_B="$(ensure_user_and_token "$USER_B" "$PASS")"
echo "✅ Got tokens for user A and user B"

UNIQUE="USER_ISOLATION_${SUFFIX}_$(python3 -c 'import uuid; print(uuid.uuid4().hex[:8])')"
TMP_FILE="/tmp/user_isolation_${UNIQUE}.txt"
echo "UniquePhrase: $UNIQUE" > "$TMP_FILE"

echo -e "\n2) Upload file as user A"
UPLOAD_RESPONSE=$(curl -sS -w "\n%{http_code}" -F "file=@$TMP_FILE;type=text/plain" \
  -H "Authorization: Bearer $TOKEN_A" "$KNOWLEDGE_ENDPOINT")
UPLOAD_BODY=$(echo "$UPLOAD_RESPONSE" | sed '$d')
UPLOAD_STATUS=$(echo "$UPLOAD_RESPONSE" | tail -n1)
if [ "$UPLOAD_STATUS" != "201" ]; then
  echo "❌ Upload failed (expected 201, got $UPLOAD_STATUS): $UPLOAD_BODY"
  exit 1
fi
FILE_ID=$(echo "$UPLOAD_BODY" | tr -d '"')
echo "✅ Uploaded file_id: $FILE_ID"

echo -e "\n3) Trigger indexing (idempotent) and wait for INDEXED"
TRIGGER_STATUS=$(curl -sS -o /tmp/isolation_trigger.json -w "%{http_code}" -X POST "$KNOWLEDGE_ENDPOINT/trigger_indexing" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN_A" \
  -d "{\"file_ids\":[\"$FILE_ID\"]}")
if [ "$TRIGGER_STATUS" != "200" ]; then
  echo "❌ trigger_indexing failed (expected 200, got $TRIGGER_STATUS)"
  cat /tmp/isolation_trigger.json || true
  exit 1
fi

INDEXED=0
for i in $(seq 1 180); do
  STATUS=$(curl -sS -X GET "$KNOWLEDGE_ENDPOINT/list_files?limit=1000&offset=0" -H "Authorization: Bearer $TOKEN_A" | python3 -c "
import json,sys
data=json.load(sys.stdin)
files=data.get('files') or []
target='$FILE_ID'
for f in files:
    if (f or {}).get('file_id')==target:
        print((f or {}).get('status') or '')
        raise SystemExit(0)
print('')
")
  echo "Attempt $i status: $STATUS"
  if [ "$STATUS" = "INDEXED" ]; then
    INDEXED=1
    break
  fi
  if [ "$i" = "30" ] || [ "$i" = "90" ]; then
    curl -sS -o /dev/null -X POST "$KNOWLEDGE_ENDPOINT/trigger_indexing" \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer $TOKEN_A" \
      -d "{\"file_ids\":[\"$FILE_ID\"]}" >/dev/null || true
  fi
  sleep 1
done
if [ "$INDEXED" != "1" ]; then
  echo "❌ File did not reach INDEXED state in time"
  exit 1
fi
echo "✅ File INDEXED"

echo -e "\n4) Query as user A (must retrieve from uploaded file)"
CHAT_A=$(curl -sS -w "\n%{http_code}" -X POST "$RAG_ENDPOINT/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN_A" \
  -d "{\"query\":\"$UNIQUE\"}")
CHAT_A_BODY=$(echo "$CHAT_A" | sed '$d')
CHAT_A_STATUS=$(echo "$CHAT_A" | tail -n1)
if [ "$CHAT_A_STATUS" != "200" ]; then
  echo "❌ chat as user A failed (status=$CHAT_A_STATUS): $CHAT_A_BODY"
  exit 1
fi
echo "$CHAT_A_BODY" | python3 -c "
import json,sys
data=json.load(sys.stdin)
chunks=data.get('chunks') or []
target='$FILE_ID'
hits=0
for ch in chunks:
    meta=(ch or {}).get('metadata') or {}
    if (meta.get('source_file_id') or meta.get('sourceFileId'))==target:
        hits += 1
if hits < 1:
    raise SystemExit('❌ user A did not retrieve any chunk from uploaded file')
print(f'✅ user A retrieved {hits} chunk(s) from uploaded file')
"

echo -e "\n5) Query as user B (must NOT retrieve user A file)"
CHAT_B=$(curl -sS -w "\n%{http_code}" -X POST "$RAG_ENDPOINT/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $TOKEN_B" \
  -d "{\"query\":\"$UNIQUE\"}")
CHAT_B_BODY=$(echo "$CHAT_B" | sed '$d')
CHAT_B_STATUS=$(echo "$CHAT_B" | tail -n1)
if [ "$CHAT_B_STATUS" != "200" ]; then
  echo "❌ chat as user B failed (status=$CHAT_B_STATUS): $CHAT_B_BODY"
  exit 1
fi
echo "$CHAT_B_BODY" | python3 -c "
import json,sys
data=json.load(sys.stdin)
chunks=data.get('chunks') or []
target='$FILE_ID'
for ch in chunks:
    meta=(ch or {}).get('metadata') or {}
    if (meta.get('source_file_id') or meta.get('sourceFileId'))==target:
        raise SystemExit('❌ user B retrieved a chunk from user A file (isolation broken)')
print('✅ user B did not retrieve user A file chunks')
"

echo -e "\n6) Ensure user B cannot download user A file"
DL_STATUS=$(curl -sS -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $TOKEN_B" "$KNOWLEDGE_ENDPOINT/$FILE_ID/download" || true)
if [ "$DL_STATUS" != "403" ] && [ "$DL_STATUS" != "404" ]; then
  echo "❌ Expected 403/404 when user B downloads user A file, got $DL_STATUS"
  exit 1
fi
echo "✅ Download blocked for user B (status=$DL_STATUS)"

echo -e "\n7) Cleanup file as user A"
DEL_STATUS=$(curl -sS -o /dev/null -w "%{http_code}" -X DELETE -H "Authorization: Bearer $TOKEN_A" "$KNOWLEDGE_ENDPOINT/$FILE_ID" || true)
if [ "$DEL_STATUS" != "202" ] && [ "$DEL_STATUS" != "204" ] && [ "$DEL_STATUS" != "404" ]; then
  echo "❌ Cleanup delete failed (status=$DEL_STATUS)"
  exit 1
fi
echo "✅ Cleanup delete scheduled (status=$DEL_STATUS)"

rm -f "$TMP_FILE" /tmp/isolation_trigger.json || true

echo -e "\n🎉 User isolation comprehensive test passed!"
