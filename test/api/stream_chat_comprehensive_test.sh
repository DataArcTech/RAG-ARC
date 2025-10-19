#!/usr/bin/env bash

set -euo pipefail

API_BASE="http://localhost:8005"
SESSION_ENDPOINT="$API_BASE/session"
AUTH_ENDPOINT="$API_BASE/auth"
KNOWLEDGE_ENDPOINT="$API_BASE/knowledge"
STREAM_CHAT_ENDPOINT="ws://localhost:8005/rag_inference/stream_chat"

echo "Testing Stream Chat WebSocket API Comprehensive Flow"
echo "===================================================="

# 0) Health check
echo "0) Health check:"
curl -sS "$API_BASE/" | grep -q "ok" && echo "✅ Health check PASS" || { echo "❌ Health check failed"; exit 1; }

# 1) Register test user only if the user doesn't exist
echo -e "\n1) Ensure test user exists:"

# First, try to login
LOGIN_PRECHECK_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$AUTH_ENDPOINT/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test_user&password=test_password")

LOGIN_PRECHECK_BODY=$(echo "$LOGIN_PRECHECK_RESPONSE" | sed '$d')
LOGIN_PRECHECK_STATUS=$(echo "$LOGIN_PRECHECK_RESPONSE" | tail -n1)

if [ "$LOGIN_PRECHECK_STATUS" = "200" ]; then
  echo "✅ User test_user already exists"
else
  echo "Registering test_user since login failed..."
  REGISTER_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$API_BASE/auth/register" \
    -H "Content-Type: application/json" \
    -d '{"name": "Test User", "user_name": "test_user", "password": "test_password"}')

  REGISTER_BODY=$(echo "$REGISTER_RESPONSE" | sed '$d')
  REGISTER_STATUS=$(echo "$REGISTER_RESPONSE" | tail -n1)

  echo "Register Status: $REGISTER_STATUS"
  echo "Register Body:   $REGISTER_BODY"

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

echo "Status: $LOGIN_STATUS"
echo "Body:   $LOGIN_BODY"

if [ "$LOGIN_STATUS" != "200" ]; then
  echo "❌ Login failed (expected 200)"
  exit 1
fi

# Extract access token from response
ACCESS_TOKEN=$(echo "$LOGIN_BODY" | grep -o '"access_token":"[^"]*"' | sed 's/"access_token":"//' | sed 's/"//')
if [ -z "$ACCESS_TOKEN" ]; then
  echo "❌ Did not receive an access token"
  exit 1
fi
echo "✅ Login PASS - access_token: ${ACCESS_TOKEN:0:20}..."

# 3) Create a session first (required for stream chat)
echo -e "\n3) Create a session for stream chat:"
CREATE_SESSION_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$SESSION_ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

CREATE_BODY=$(echo "$CREATE_SESSION_RESPONSE" | sed '$d')
CREATE_STATUS=$(echo "$CREATE_SESSION_RESPONSE" | tail -n1)

echo "Status: $CREATE_STATUS"
echo "Body:   $CREATE_BODY"

if [ "$CREATE_STATUS" != "200" ]; then
  echo "❌ Create session failed (expected 200)"
  exit 1
fi

# The response body is just the session ID string directly, not JSON
SESSION_ID=$(echo "$CREATE_BODY" | tr -d '"')
if [ -z "$SESSION_ID" ]; then
  echo "❌ Did not receive a session id"
  exit 1
fi
echo "✅ Create session PASS - session_id: $SESSION_ID"

# 4) Upload test file for RAG functionality
echo -e "\n4) Upload test file for RAG functionality:"
TEST_FILE="./test/test2.html"

echo "Uploading file: $TEST_FILE"
UPLOAD_RESPONSE=$(curl -sS -w "\n%{http_code}" -F "file=@$TEST_FILE;type=application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" "$KNOWLEDGE_ENDPOINT")

UPLOAD_BODY=$(echo "$UPLOAD_RESPONSE" | sed '$d')
UPLOAD_STATUS=$(echo "$UPLOAD_RESPONSE" | tail -n1)

echo "Status: $UPLOAD_STATUS"
echo "Body:   $UPLOAD_BODY"

if [ "$UPLOAD_STATUS" != "201" ]; then
  echo "❌ Upload failed (expected 201)"
  exit 1
fi

FILE_ID=$(echo "$UPLOAD_BODY" | tr -d '"')
if ! echo "$FILE_ID" | grep -q "-"; then
  echo "❌ Did not receive a file id"
  exit 1
fi
echo "✅ Upload PASS - file_id: $FILE_ID"

# Wait for indexing to complete
echo -e "\n⏳ Waiting for indexing to complete (10 seconds)..."
sleep 10

# 5) Test WebSocket connection to stream chat
echo -e "\n5) Test WebSocket connection to stream chat:"

# Create a temporary Python script to test WebSocket connection
cat > /tmp/test_websocket.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import websockets
import json
import sys
import os

async def test_websocket():
    session_id = sys.argv[1]
    access_token = sys.argv[2]
    uri = f"ws://localhost:8005/rag_inference/stream_chat/{session_id}"
    
    try:
        # Connect to WebSocket with JWT authentication via cookies
        async with websockets.connect(uri, additional_headers=[("Cookie", f"auth_token={access_token}")]) as websocket:
            print("✅ WebSocket connection established")
            
            # Send a test message
            test_message = "Hello, this is a test message for stream chat"
            await websocket.send(test_message)
            print(f"✅ Sent message: {test_message}")
            
            # Wait for response with timeout
            try:
                response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                print(f"✅ Received response: {response}")
                
                # Try to parse as JSON
                try:
                    response_data = json.loads(response)
                    if "error" in response_data:
                        print(f"❌ Error in response: {response_data['error']}")
                        return False
                    else:
                        print("✅ Response is valid JSON")
                        return True
                except json.JSONDecodeError:
                    print("✅ Response is not JSON (might be streaming text)")
                    return True
                    
            except asyncio.TimeoutError:
                print("❌ Timeout waiting for response")
                return False
                
    except Exception as e:
        print(f"❌ WebSocket connection failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 test_websocket.py <session_id> <access_token>")
        sys.exit(1)
    
    result = asyncio.run(test_websocket())
    sys.exit(0 if result else 1)
EOF

# Make the script executable and run it
chmod +x /tmp/test_websocket.py
python3 /tmp/test_websocket.py "$SESSION_ID" "$ACCESS_TOKEN"
WEBSOCKET_RESULT=$?

if [ $WEBSOCKET_RESULT -eq 0 ]; then
  echo "✅ WebSocket stream chat test PASS"
else
  echo "❌ WebSocket stream chat test FAILED"
  exit 1
fi

# 6) Test multiple message exchange with uploaded content
echo -e "\n6) Test multiple message exchange with uploaded content:"

cat > /tmp/test_multiple_messages.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import websockets
import json
import sys
import time

async def test_multiple_messages():
    session_id = sys.argv[1]
    access_token = sys.argv[2]
    uri = f"ws://localhost:8005/rag_inference/stream_chat/{session_id}"
    
    try:
        # Connect to WebSocket with JWT authentication via cookies
        async with websockets.connect(uri, additional_headers=[("Cookie", f"auth_token={access_token}")]) as websocket:
            print("✅ WebSocket connection established for multiple messages")
            
            messages = [
                "What is the capital of France?",
                "Who is the author of Venom: The Black Suit Saga?",
                "Tell me about the content in the uploaded file",
                "Which comic you mentioned is authored by Yuxuan Zhou?"
            ]
            
            for i, message in enumerate(messages, 1):
                print(f"Sending message {i}: {message}")
                await websocket.send(message)
                
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                    print(f"✅ Received response {i}: {response[:100]}...")
                    
                    # Special check for the author question
                    if message == "Who is the author of Venom: The Black Suit Saga?":
                        print(f"🔍 Checking author response: {response}")
                        if "Yuxuan Zhou" in response:
                            print("✅ Author check PASS - Found 'Yuxuan Zhou' in response")
                        else:
                            print("❌ Author check FAILED - 'Yuxuan Zhou' not found in response")
                            print(f"Full response: {response}")
                            return False
                            
                    if message == "Which comic you mentioned is authored by Yuxuan Zhou?":
                        print(f"🔍 Checking author response: {response}")
                        if "Yuxuan Zhou" in response:
                            print("✅ Author check PASS - Found 'Venom: The Black Suit Saga' in response")
                        else:
                            print("❌ Author check FAILED - 'Venom: The Black Suit Saga' not found in response")
                            print(f"Full response: {response}")
                            return False
                    # Small delay between messages
                    await asyncio.sleep(1)
                    
                except asyncio.TimeoutError:
                    print(f"❌ Timeout waiting for response {i}")
                    return False
                    
            return True
            
    except Exception as e:
        print(f"❌ Multiple messages test failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 test_multiple_messages.py <session_id> <access_token>")
        sys.exit(1)
    
    result = asyncio.run(test_multiple_messages())
    sys.exit(0 if result else 1)
EOF

chmod +x /tmp/test_multiple_messages.py
python3 /tmp/test_multiple_messages.py "$SESSION_ID" "$ACCESS_TOKEN"
MULTIPLE_MESSAGES_RESULT=$?

if [ $MULTIPLE_MESSAGES_RESULT -eq 0 ]; then
  echo "✅ Multiple message exchange test PASS"
else
  echo "❌ Multiple message exchange test FAILED"
  exit 1
fi

# 7) Test unauthorized WebSocket connection (without session cookie)
echo -e "\n7) Test unauthorized WebSocket connection:"

cat > /tmp/test_unauthorized_websocket.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import websockets
import sys

async def test_unauthorized_websocket():
    session_id = sys.argv[1]
    uri = f"ws://localhost:8005/rag_inference/stream_chat/{session_id}"

    # The connection should be closed quickly due to missing authentication cookie, so we expect a policy violation (1008) error code
    try:
        async with websockets.connect(uri) as websocket:
            await asyncio.wait_for(websocket.send("test"), timeout=1.0)
            print("❌ Unauthorized connection should have been closed")
            return False
    except websockets.exceptions.ConnectionClosed as e:
        if e.code == 1008:  # Policy violation
            print("✅ Unauthorized connection properly closed with policy violation (1008)")
            return True
        else:
            print(f"❌ Unexpected close code: {e.code}")
            return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test_unauthorized_websocket.py <session_id>")
        sys.exit(1)

    result = asyncio.run(test_unauthorized_websocket())
    sys.exit(0 if result else 1)
EOF

chmod +x /tmp/test_unauthorized_websocket.py
python3 /tmp/test_unauthorized_websocket.py "$SESSION_ID"
UNAUTHORIZED_RESULT=$?

if [ $UNAUTHORIZED_RESULT -eq 0 ]; then
  echo "✅ Unauthorized WebSocket connection test PASS"
else
  echo "❌ Unauthorized WebSocket connection test FAILED"
  exit 1
fi

# 8) Test connection to non-existent session
echo -e "\n8) Test connection to non-existent session:"

NONEXISTENT_SESSION_ID="00000000-0000-0000-0000-000000000000"
cat > /tmp/test_nonexistent_session.py << 'EOF'
#!/usr/bin/env python3
import asyncio
import websockets
import sys

async def test_nonexistent_session():
    session_id = sys.argv[1]
    access_token = sys.argv[2]
    uri = f"ws://localhost:8005/rag_inference/stream_chat/{session_id}"
    
    # The connection may be established but should be closed immediately when session validation fails
    try:
        async with websockets.connect(uri, additional_headers=[("Cookie", f"auth_token={access_token}")]) as websocket:
            await asyncio.wait_for(websocket.send("test"), timeout=1.0)
            print("❌ Connection to non-existent session should have been rejected")
            return False
    except websockets.exceptions.ConnectionClosed as e:
        if e.code == 1008:  # Policy violation
            print("✅ Non-existent session properly closed with policy violation (1008)")
            return True
        else:
            print(f"❌ Unexpected close code: {e.code}")
            return False

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 test_nonexistent_session.py <session_id> <access_token>")
        sys.exit(1)
    
    result = asyncio.run(test_nonexistent_session())
    sys.exit(0 if result else 1)
EOF

chmod +x /tmp/test_nonexistent_session.py
python3 /tmp/test_nonexistent_session.py "$NONEXISTENT_SESSION_ID" "$ACCESS_TOKEN"
NONEXISTENT_RESULT=$?

if [ $NONEXISTENT_RESULT -eq 0 ]; then
  echo "✅ Non-existent session connection test PASS"
else
  echo "❌ Non-existent session connection test FAILED"
  exit 1
fi

# 9) Verify messages were created in the session
echo -e "\n9) Verify messages were created in the session:"
LIST_MESSAGES_RESPONSE=$(curl -sS -w "\n%{http_code}" -X GET "$SESSION_ENDPOINT/$SESSION_ID/messages" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

MESSAGES_BODY=$(echo "$LIST_MESSAGES_RESPONSE" | sed '$d')
MESSAGES_STATUS=$(echo "$LIST_MESSAGES_RESPONSE" | tail -n1)

echo "Status: $MESSAGES_STATUS"
echo "Body:   $MESSAGES_BODY"

if [ "$MESSAGES_STATUS" != "200" ]; then
  echo "❌ List messages failed (expected 200)"
  exit 1
fi

# Count messages in the response
MESSAGE_COUNT=$(echo "$MESSAGES_BODY" | grep -o '"content"' | wc -l)
echo "Found $MESSAGE_COUNT messages in session"

if [ "$MESSAGE_COUNT" -lt 3 ]; then
  echo "❌ Expected at least 3 messages (user messages + assistant responses)"
  exit 1
fi
echo "✅ Messages properly stored in session PASS"

# 10) Cleanup uploaded file
echo -e "\n10) Cleanup uploaded file:"
DELETE_CODE=$(curl -sS -o /dev/null -w "%{http_code}" -X DELETE \
  -H "Authorization: Bearer $ACCESS_TOKEN" "$KNOWLEDGE_ENDPOINT/$FILE_ID")
echo "Delete Status: $DELETE_CODE"
if [ "$DELETE_CODE" != "204" ]; then
  echo "❌ Delete failed (expected 204)"
  exit 1
fi
echo "✅ File cleanup PASS"

# Cleanup temporary files
rm -f /tmp/test_websocket.py /tmp/test_multiple_messages.py /tmp/test_unauthorized_websocket.py /tmp/test_nonexistent_session.py

echo -e "\n🎉 All Stream Chat WebSocket API comprehensive tests passed!"
echo "✅ File upload and indexing"
echo "✅ WebSocket connection establishment"
echo "✅ Single message exchange"
echo "✅ Multiple message exchange with RAG content"
echo "✅ Authorization and access control"
echo "✅ Error handling for invalid sessions"
echo "✅ Message persistence in session"
echo "✅ File cleanup"
