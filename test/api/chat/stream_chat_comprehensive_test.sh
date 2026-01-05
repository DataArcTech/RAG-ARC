#!/usr/bin/env bash

set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8000}"
SESSION_ENDPOINT="$API_BASE/session"
AUTH_ENDPOINT="$API_BASE/auth"
KNOWLEDGE_ENDPOINT="$API_BASE/knowledge"
STREAM_CHAT_ENDPOINT="$API_BASE/rag_inference/stream_chat"

echo "Testing Stream Chat SSE API Comprehensive Flow"
echo "================================================"

# 0) Health check
echo "0) Health check:"
curl -sS "$API_BASE/" | grep -q "ok" && echo "✅ Health check PASS" || { echo "❌ Health check failed"; exit 1; }

# 1) Ensure test user exists
echo -e "\n1) Ensure test user exists:"

LOGIN_PRECHECK_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$AUTH_ENDPOINT/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test_user&password=test_password")

LOGIN_PRECHECK_STATUS=$(echo "$LOGIN_PRECHECK_RESPONSE" | tail -n1)

if [ "$LOGIN_PRECHECK_STATUS" = "200" ]; then
  echo "✅ User test_user already exists"
else
  echo "Registering test_user since login failed..."
  REGISTER_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$API_BASE/auth/register" \
    -H "Content-Type: application/json" \
    -d '{"name": "Test User", "user_name": "test_user", "password": "test_password"}')

  REGISTER_STATUS=$(echo "$REGISTER_RESPONSE" | tail -n1)
  if [ "$REGISTER_STATUS" != "201" ]; then
    echo "❌ User registration failed (expected 201, got $REGISTER_STATUS)"
    exit 1
  else
    echo "✅ test_user successfully registered"
  fi
fi

# 2) Login to get authentication token
echo -e "\n2) Login to get authentication token:"
LOGIN_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$AUTH_ENDPOINT/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test_user&password=test_password")

LOGIN_BODY=$(echo "$LOGIN_RESPONSE" | sed '$d')
LOGIN_STATUS=$(echo "$LOGIN_RESPONSE" | tail -n1)

if [ "$LOGIN_STATUS" != "200" ]; then
  echo "❌ Login failed (expected 200)"
  exit 1
fi

ACCESS_TOKEN=$(echo "$LOGIN_BODY" | grep -o '"access_token":"[^"]*"' | sed 's/"access_token":"//' | sed 's/"//')
if [ -z "$ACCESS_TOKEN" ]; then
  echo "❌ Did not receive an access token"
  exit 1
fi
echo "✅ Login PASS - access_token: ${ACCESS_TOKEN:0:20}..."

# 3) Create a session (required for stream chat)
echo -e "\n3) Create a session for stream chat:"
CREATE_SESSION_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$SESSION_ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

CREATE_BODY=$(echo "$CREATE_SESSION_RESPONSE" | sed '$d')
CREATE_STATUS=$(echo "$CREATE_SESSION_RESPONSE" | tail -n1)

if [ "$CREATE_STATUS" != "200" ]; then
  echo "❌ Create session failed (expected 200)"
  exit 1
fi

SESSION_ID=$(echo "$CREATE_BODY" | tr -d '"')
if [ -z "$SESSION_ID" ]; then
  echo "❌ Did not receive a session id"
  exit 1
fi
echo "✅ Create session PASS - session_id: $SESSION_ID"

# 4) Upload a test file for RAG functionality
echo -e "\n4) Upload test file for RAG functionality:"
TEST_FILE="./test/test2.html"
UPLOAD_RESPONSE=$(curl -sS -w "\n%{http_code}" -F "file=@$TEST_FILE;type=application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" "$KNOWLEDGE_ENDPOINT")

UPLOAD_STATUS=$(echo "$UPLOAD_RESPONSE" | tail -n1)
if [ "$UPLOAD_STATUS" != "201" ]; then
  echo "❌ Upload failed (expected 201, got $UPLOAD_STATUS)"
  exit 1
fi
echo "✅ Upload PASS"

echo -e "\n⏳ Waiting for indexing to complete (10 seconds)..."
sleep 10

# 5) Test SSE stream chat (single message)
echo -e "\n5) Test SSE stream chat:"
ENC_QUERY=$(python3 -c 'import urllib.parse; print(urllib.parse.quote("Hello, this is a test message for stream chat"))')
SSE_URL="$STREAM_CHAT_ENDPOINT/$SESSION_ID?query=$ENC_QUERY"
SSE_RESULT=$(uv run python test/api/stream_chat_sse_comprehensive_test.py "$SSE_URL" "$ACCESS_TOKEN")
echo "$SSE_RESULT" | grep -q '"http_status": 200' && echo "✅ SSE stream chat test PASS" || { echo "❌ SSE stream chat test FAILED"; echo "$SSE_RESULT"; exit 1; }

# 5.1) Test SSE evidence/subgraph payload tool-call
echo -e "\n5.1) Test SSE stream chat with evidence + subgraph payload:"
SSE_URL_EVID="$STREAM_CHAT_ENDPOINT/$SESSION_ID?query=$ENC_QUERY&include_evidence=true&return_subgraph=true"
SSE_RESULT_EVID=$(uv run python test/api/stream_chat_sse_comprehensive_test.py "$SSE_URL_EVID" "$ACCESS_TOKEN")
echo "$SSE_RESULT_EVID" | grep -q '"http_status": 200' || { echo "❌ SSE (evidence) request FAILED"; echo "$SSE_RESULT_EVID"; exit 1; }
echo "$SSE_RESULT_EVID" | python3 -c "
import json,sys
data=json.load(sys.stdin)
assert data.get('payload_calls', 0) >= 1, 'Expected at least one rag_arc_payload tool-call'
assert data.get('payload_has_evidence') is True, 'Expected evidence field in rag_arc_payload arguments'
if data.get('payload_has_subgraph') is True:
    print('✅ SSE rag_arc_payload includes evidence + subgraph')
else:
    print('⚠️  SSE rag_arc_payload missing subgraph (graph generation may be unavailable); evidence is present')
"

# 6) Unauthorized SSE request should be rejected
echo -e "\n6) Test unauthorized SSE request:"
UNAUTH_URL="$STREAM_CHAT_ENDPOINT/$SESSION_ID?query=test"
UNAUTH_OUT=$(uv run python test/api/stream_chat_sse_comprehensive_test.py "$UNAUTH_URL")
echo "$UNAUTH_OUT" | grep -q '"http_status": 401' && echo "✅ Unauthorized SSE request PASS" || { echo "❌ Unauthorized SSE request FAILED"; echo "$UNAUTH_OUT"; exit 1; }

echo -e "\n🎉 All Stream Chat SSE API comprehensive tests passed!"
