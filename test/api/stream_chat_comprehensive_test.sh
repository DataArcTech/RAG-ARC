#!/usr/bin/env bash

set -euo pipefail

API_BASE="http://localhost:8000"
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

cat > /tmp/test_sse_stream_chat.py << 'EOF'
#!/usr/bin/env python3
import json
import sys
import httpx

def read_stream_text(url: str, token: str | None):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    parts: list[str] = []
    with httpx.stream("GET", url, headers=headers, timeout=120.0) as r:
        if r.status_code != 200:
            return {"http_status": r.status_code, "body": r.text}
        for line in r.iter_lines():
            if not line:
                continue
            if not line.startswith("data:"):
                continue
            data = line.split(":", 1)[1].strip()
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = (choices[0] or {}).get("delta") or {}
            parts.append(delta.get("content") or "")
    return {"http_status": 200, "text": "".join(parts)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_sse_stream_chat.py <url> [token]")
        sys.exit(1)
    url = sys.argv[1]
    token = sys.argv[2] if len(sys.argv) >= 3 else None
    result = read_stream_text(url, token)
    print(json.dumps(result, ensure_ascii=False))
EOF

chmod +x /tmp/test_sse_stream_chat.py

# 5) Test SSE stream chat (single message)
echo -e "\n5) Test SSE stream chat:"
ENC_QUERY=$(python3 -c 'import urllib.parse; print(urllib.parse.quote("Hello, this is a test message for stream chat"))')
SSE_URL="$STREAM_CHAT_ENDPOINT/$SESSION_ID?query=$ENC_QUERY"
SSE_RESULT=$(python3 /tmp/test_sse_stream_chat.py "$SSE_URL" "$ACCESS_TOKEN")
echo "$SSE_RESULT" | grep -q '"http_status": 200' && echo "✅ SSE stream chat test PASS" || { echo "❌ SSE stream chat test FAILED"; echo "$SSE_RESULT"; exit 1; }

# 6) Unauthorized SSE request should be rejected
echo -e "\n6) Test unauthorized SSE request:"
UNAUTH_URL="$STREAM_CHAT_ENDPOINT/$SESSION_ID?query=test"
UNAUTH_OUT=$(python3 /tmp/test_sse_stream_chat.py "$UNAUTH_URL")
echo "$UNAUTH_OUT" | grep -q '"http_status": 401' && echo "✅ Unauthorized SSE request PASS" || { echo "❌ Unauthorized SSE request FAILED"; echo "$UNAUTH_OUT"; exit 1; }

rm -f /tmp/test_sse_stream_chat.py

echo -e "\n🎉 All Stream Chat SSE API comprehensive tests passed!"
