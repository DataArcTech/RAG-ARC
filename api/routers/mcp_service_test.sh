ENDPOINT="http://192.168.80.1:8005/mcp/"

echo "Testing MCP Server at $ENDPOINT"
echo "=================================="

echo "1. Initialize connection (this will show the session ID in headers):"
INIT_RESPONSE=$(curl -i -s "$ENDPOINT" \
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

# Extract session ID from headers
SESSION_ID=$(echo "$INIT_RESPONSE" | grep -i mcp-session-id | awk -F': ' '{print $2}' | tr -d '\r\n')

if [ -z "$SESSION_ID" ]; then
  echo "Could not extract session ID. Exiting."
  exit 1
fi

echo -e "\n2. Send initialized notification:"
curl -i -s "$ENDPOINT" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "method": "notifications/initialized"
  }' > /dev/null

echo "Initialized notification sent."

echo -e "\n3. Test hello_world tool using the session ID:"
HELLO_RESPONSE=$(curl -i -s "$ENDPOINT" \
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
echo -e "\n4. Test create_chat tool using the session ID:"
CREATE_CHAT_RESPONSE=$(curl -i -s "$ENDPOINT" \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -H "mcp-session-id: $SESSION_ID" \
  -d '{
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {
      "name": "create_chat",
      "arguments": {}
    }
  }')

echo "$CREATE_CHAT_RESPONSE" | grep -E "(data: |event: )"
echo
echo "Full create_chat response body:"
echo "$CREATE_CHAT_RESPONSE" | sed -n '/^\r$/,$p' | tail -n +2

CHAT_SESSION_ID=$(echo "$CREATE_CHAT_RESPONSE" | grep -o '"session_id":"[^"]*"' | sed 's/"session_id":"//' | sed 's/"//')

if [ -n "$CHAT_SESSION_ID" ]; then
  echo "✅ [create_chat] PASS - session_id: $CHAT_SESSION_ID"
else
  echo "❌ [create_chat] Did not find session_id in response"
fi

echo -e "\n5. Test chat tool using the chat session ID:"
CHAT_RESPONSE=$(curl -i -s "$ENDPOINT" \
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
        \"query\": \"Give me 10 best amazon deals?\"
      }
    }
  }")

echo "$CHAT_RESPONSE" | grep -E "(data: |event: )"
echo
echo "Full chat response body:"
echo "$CHAT_RESPONSE" | sed -n '/^\r$/,$p' | tail -n +2

echo "$CHAT_RESPONSE" | grep -q 'reply' && echo "✅ [chat] PASS" || echo "❌ [chat] Did not find 'reply' in response"
