ENDPOINT="http://localhost:8000/mcp/"

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

echo "$HELLO_RESPONSE" | grep -q 'Hello, world!' && echo "✅ PASS" || echo "❌ Did not find 'Hello, world!'"
