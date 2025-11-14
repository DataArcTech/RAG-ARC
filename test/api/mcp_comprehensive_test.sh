#!/usr/bin/env bash

set -euo pipefail

API_BASE="http://localhost:8000"
KNOWLEDGE_ENDPOINT="$API_BASE/knowledge"
MCP_ENDPOINT="$API_BASE/mcp/"

echo "Testing MCP Server Comprehensive Test Suite"
echo "==========================================="

# 0) Health check
curl -sS "$API_BASE/" | grep -q "ok" || { echo "❌ Health check failed"; exit 1; }
echo "✅ Health check PASS"

# 0.5) Authentication setup
echo -e "\n0.5. Setting up authentication:"
AUTH_ENDPOINT="$API_BASE/auth"

# Register test user
echo "Registering test user..."
REGISTER_RESPONSE=$(curl -sS -w "\n%{http_code}" "$AUTH_ENDPOINT/register" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Test User",
    "user_name": "testuser",
    "password": "testpass123"
  }')

REGISTER_STATUS=$(echo "$REGISTER_RESPONSE" | tail -n1)
if [ "$REGISTER_STATUS" = "201" ]; then
  echo "✅ User registered successfully"
elif [ "$REGISTER_STATUS" = "400" ]; then
  echo "ℹ️  User already exists, continuing..."
else
  echo "❌ User registration failed (status $REGISTER_STATUS)"
  echo "$REGISTER_RESPONSE"
  exit 1
fi

# Login to get auth token
echo "Logging in to get auth token..."
LOGIN_RESPONSE=$(curl -sS -w "\n%{http_code}" "$AUTH_ENDPOINT/token" \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d 'username=testuser&password=testpass123')

LOGIN_STATUS=$(echo "$LOGIN_RESPONSE" | tail -n1)
if [ "$LOGIN_STATUS" != "200" ]; then
  echo "❌ Login failed (status $LOGIN_STATUS)"
  echo "$LOGIN_RESPONSE"
  exit 1
fi

# Extract access token
AUTH_TOKEN=$(echo "$LOGIN_RESPONSE" | sed '$d' | grep -o '"access_token":"[^"]*"' | sed 's/"access_token":"//' | sed 's/"//')
if [ -z "$AUTH_TOKEN" ]; then
  echo "❌ Could not extract auth token from login response"
  echo "$LOGIN_RESPONSE"
  exit 1
fi

echo "✅ Authentication successful, token obtained"

# 1) Initialize MCP and capture session ID
echo -e "\n1. Initialize MCP connection:"
INIT_RESPONSE=$(curl -i -s "$MCP_ENDPOINT" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {
      "protocolVersion": "2025-06-18",
      "capabilities": {},
      "clientInfo": {"name": "curl", "version": "1.0"}
    }
  }')

echo "$INIT_RESPONSE" | grep -E "(mcp-session-id|data: |event: )"

SESSION_ID=$(echo "$INIT_RESPONSE" | grep -i mcp-session-id | awk -F': ' '{print $2}' | tr -d '\r\n')
if [ -z "$SESSION_ID" ]; then
  echo "❌ Could not extract MCP session id"
  echo "$INIT_RESPONSE"
  exit 1
fi
echo "✅ MCP session established: $SESSION_ID"

# 2) Send initialized notification (required by FastMCP protocol)
echo -e "\n2. Send initialized notification:"
curl -i -s "$MCP_ENDPOINT" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "method": "notifications/initialized"
  }' > /dev/null

echo "✅ Initialized notification sent."

# 3) Test hello_world tool
echo -e "\n3. Test hello_world tool:"
HELLO_RESPONSE=$(curl -i -s "$MCP_ENDPOINT" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "id": 2,
    "method": "tools/call",
    "params": {
      "name": "hello_world",
      "arguments": {}
    }
  }')

echo "$HELLO_RESPONSE" | grep -E "(data: |event: )"
echo
echo "Full hello_world response body:"
echo "$HELLO_RESPONSE" | sed -n '/^\r$/,$p' | tail -n +2

echo "$HELLO_RESPONSE" | grep -q 'Hello, world!' && echo "✅ [hello_world] PASS" || echo "❌ [hello_world] Did not find 'Hello, world!'"

# 4) Test create_chat tool
echo -e "\n4. Test create_chat tool:"
CREATE_CHAT_RESPONSE=$(curl -i -s "$MCP_ENDPOINT" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "mcp-session-id: $SESSION_ID" \
  -d "{
    \"jsonrpc\": \"2.0\",
    \"id\": 3,
    \"method\": \"tools/call\",
    \"params\": {
      \"name\": \"create_chat\",
      \"arguments\": {
        \"auth_token\": \"$AUTH_TOKEN\"
      }
    }
  }")

echo "$CREATE_CHAT_RESPONSE" | grep -E "(data: |event: )"
echo
echo "Full create_chat response body:"
echo "$CREATE_CHAT_RESPONSE" | sed -n '/^\r$/,$p' | tail -n +2

# Extract JSON payload from SSE if present
CREATE_CHAT_BODY=$(echo "$CREATE_CHAT_RESPONSE" | sed -n 's/^data: //p' | tail -n1)
if [ -z "$CREATE_CHAT_BODY" ]; then
  # Fallback: try to strip headers if any, otherwise use raw
  CREATE_CHAT_BODY=$(echo "$CREATE_CHAT_RESPONSE" | sed -n '/^$/,$p' | tail -n +2)
  if [ -z "$CREATE_CHAT_BODY" ]; then
    CREATE_CHAT_BODY="$CREATE_CHAT_RESPONSE"
  fi
fi

# Robust session_id extraction
CHAT_SESSION_ID=$(echo "$CREATE_CHAT_BODY" | sed -n 's/.*"session_id":"\([^"]*\)".*/\1/p')
if [ -z "$CHAT_SESSION_ID" ]; then
  # Fallback extraction method
  CHAT_SESSION_ID=$(echo "$CREATE_CHAT_RESPONSE" | grep -o '"session_id":"[^"]*"' | sed 's/"session_id":"//' | sed 's/"//')
fi

if [ -n "$CHAT_SESSION_ID" ]; then
  echo "✅ [create_chat] PASS - session_id: $CHAT_SESSION_ID"
else
  echo "❌ [create_chat] Did not find session_id in response"
  echo "--- Raw Response ---"
  echo "$CREATE_CHAT_RESPONSE"
  echo "--- Parsed Body ---"
  echo "$CREATE_CHAT_BODY"
  exit 1
fi

# 5) Test basic chat functionality
echo -e "\n5. Test basic chat functionality:"
CHAT_RESPONSE=$(curl -i -s "$MCP_ENDPOINT" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "mcp-session-id: $SESSION_ID" \
  -d "{
    \"jsonrpc\": \"2.0\",
    \"id\": 4,
    \"method\": \"tools/call\",
    \"params\": {
      \"name\": \"chat\",
      \"arguments\": {
        \"session_id\": \"$CHAT_SESSION_ID\",
        \"query\": \"Give me 10 best amazon deals?\",
        \"auth_token\": \"$AUTH_TOKEN\"
      }
    }
  }")

echo "$CHAT_RESPONSE" | grep -E "(data: |event: )"
echo
echo "Full chat response body:"
echo "$CHAT_RESPONSE" | sed -n '/^\r$/,$p' | tail -n +2

echo "$CHAT_RESPONSE" | grep -q '"response"' && echo "✅ [basic chat] PASS" || echo "❌ [basic chat] Did not find 'response' in response"

# 6) Test retrieval functionality by uploading content and querying it
echo -e "\n6. Testing MCP retrieval functionality:"
echo "========================================="

# Upload test HTML file via REST
TEST_FILE="./test/test2.html"

echo "Uploading test file: $TEST_FILE"
UPLOAD_RESPONSE=$(curl -sS -w "\n%{http_code}" \
  -H "Authorization: Bearer $AUTH_TOKEN" \
  -F "file=@$TEST_FILE" "$KNOWLEDGE_ENDPOINT")
UPLOAD_BODY=$(echo "$UPLOAD_RESPONSE" | sed '$d')
UPLOAD_STATUS=$(echo "$UPLOAD_RESPONSE" | tail -n1)

if [ "$UPLOAD_STATUS" != "201" ]; then
  echo "❌ Upload failed (status $UPLOAD_STATUS)"
  echo "$UPLOAD_BODY"
  exit 1
fi

FILE_ID=$(echo "$UPLOAD_BODY" | tr -d '"')
echo "✅ Uploaded file_id: $FILE_ID"

# 7) Test retrieval with uploaded content
echo -e "\nWaiting for indexing to complete (10 seconds)..."
sleep 10

echo -e "\n7. Testing retrieval with uploaded content:"
MAX_ATTEMPTS=6
SLEEP_SECONDS=5
ATTEMPT=1
SUCCESS=0

while [ $ATTEMPT -le $MAX_ATTEMPTS ]; do
  echo "Attempt $ATTEMPT/$MAX_ATTEMPTS querying MCP chat for uploaded content..."
  CHAT_PAYLOAD=$(cat <<EOF
{
  "jsonrpc": "2.0",
  "id": $((ATTEMPT+4)),
  "method": "tools/call",
  "params": {
    "name": "chat",
    "arguments": {
      "session_id": "$CHAT_SESSION_ID",
      "query": "who is the author of Venom: The Black Suit Saga?",
      "auth_token": "$AUTH_TOKEN"
    }
  }
}
EOF
)

  CHAT_RESPONSE=$(curl -i -s -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' -H "mcp-session-id: $SESSION_ID" -d "$CHAT_PAYLOAD" "$MCP_ENDPOINT") || true

  # Extract JSON payload from SSE if present
  BODY=$(echo "$CHAT_RESPONSE" | sed -n 's/^data: //p' | tail -n1)
  # Fallback if not SSE framed
  if [ -z "$BODY" ]; then
    BODY=$(echo "$CHAT_RESPONSE" | sed -n '/^$/,$p' | tail -n +2)
    if [ -z "$BODY" ]; then BODY="$CHAT_RESPONSE"; fi
  fi

  # Check for token presence safely
  if echo "$BODY" | grep -qi "Yuxuan Zhou"; then
    echo "✅ MCP reply includes expected author information"
    SUCCESS=1
    break
  fi

  echo "Author not found yet; sleeping $SLEEP_SECONDS s..."
  sleep $SLEEP_SECONDS
  ATTEMPT=$((ATTEMPT+1))
done

if [ "$SUCCESS" -ne 1 ]; then
  echo "❌ MCP did not surface uploaded content author within timeout"
  echo "Last response:"
  echo "$CHAT_RESPONSE"
  exit 1
fi

echo -e "\n🎉 All MCP comprehensive tests passed!"
echo "✅ Basic MCP service functionality"
echo "✅ Chat session creation and management"
echo "✅ Content upload and retrieval"
echo "✅ End-to-end RAG pipeline"
