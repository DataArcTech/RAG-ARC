#!/usr/bin/env bash

set -euo pipefail

API_BASE="http://localhost:8000"
SESSION_ENDPOINT="$API_BASE/session"
AUTH_ENDPOINT="$API_BASE/auth"

echo "Testing Session REST API Comprehensive Flow at $SESSION_ENDPOINT"
echo "=================================================================="

# 0) Health check
echo "0) Health check:"
curl -sS "$API_BASE/" | grep -q "ok" && echo "✅ Health check PASS" || { echo "❌ Health check failed"; exit 1; }

# 1) Check if test user exists, register if not
echo -e "\n1) Check if test user exists and register if needed:"

# First, try to login to see if user already exists
echo "Checking if test user already exists..."
LOGIN_CHECK_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$AUTH_ENDPOINT/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test_user&password=test_password")

LOGIN_CHECK_BODY=$(echo "$LOGIN_CHECK_RESPONSE" | sed '$d')
LOGIN_CHECK_STATUS=$(echo "$LOGIN_CHECK_RESPONSE" | tail -n1)

if [ "$LOGIN_CHECK_STATUS" = "200" ]; then
  echo "✅ Test user already exists, skipping registration"
else
  echo "Test user does not exist, registering new user..."
  REGISTER_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$AUTH_ENDPOINT/register" \
    -H "Content-Type: application/json" \
    -d '{"name": "Test User", "user_name": "test_user", "password": "test_password"}')

  REGISTER_BODY=$(echo "$REGISTER_RESPONSE" | sed '$d')
  REGISTER_STATUS=$(echo "$REGISTER_RESPONSE" | tail -n1)

  echo "Registration Status: $REGISTER_STATUS"
  echo "Registration Body:   $REGISTER_BODY"

  if [ "$REGISTER_STATUS" != "201" ]; then
    echo "❌ User registration failed (expected 201)"
    exit 1
  fi
  echo "✅ Test user registered successfully"
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

# 3) Create a new session
echo -e "\n3) Create a new session:"
CREATE_SESSION_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$SESSION_ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

# Split body and status code
CREATE_BODY=$(echo "$CREATE_SESSION_RESPONSE" | sed '$d')
CREATE_STATUS=$(echo "$CREATE_SESSION_RESPONSE" | tail -n1)

echo "Status: $CREATE_STATUS"
echo "Body:   $CREATE_BODY"

if [ "$CREATE_STATUS" != "200" ]; then
  echo "❌ Create session failed (expected 200)"
  exit 1
fi

# Extract session ID from response (response is just the UUID string directly)
SESSION_ID=$(echo "$CREATE_BODY" | sed 's/"//g')
if [ -z "$SESSION_ID" ]; then
  echo "❌ Did not receive a session id"
  exit 1
fi
echo "✅ Create session PASS - session_id: $SESSION_ID"

# 4) List all sessions
echo -e "\n4) List all sessions:"
LIST_SESSIONS_RESPONSE=$(curl -sS -w "\n%{http_code}" -X GET "$SESSION_ENDPOINT" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

LIST_BODY=$(echo "$LIST_SESSIONS_RESPONSE" | sed '$d')
LIST_STATUS=$(echo "$LIST_SESSIONS_RESPONSE" | tail -n1)

echo "Status: $LIST_STATUS"
echo "Body:   $LIST_BODY"

if [ "$LIST_STATUS" != "200" ]; then
  echo "❌ List sessions failed (expected 200)"
  exit 1
fi

# Verify our created session appears in the list
if ! echo "$LIST_BODY" | grep -q "$SESSION_ID"; then
  echo "❌ Created session not found in session list"
  exit 1
fi
echo "✅ List sessions PASS"

# 5) Create a message in the session
echo -e "\n5) Create a message in the session:"
MESSAGE_CONTENT="Hello, this is a test message for session $SESSION_ID"
CREATE_MESSAGE_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$SESSION_ENDPOINT/$SESSION_ID/messages" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d "{\"content\": \"$MESSAGE_CONTENT\"}")

MESSAGE_BODY=$(echo "$CREATE_MESSAGE_RESPONSE" | sed '$d')
MESSAGE_STATUS=$(echo "$CREATE_MESSAGE_RESPONSE" | tail -n1)

echo "Status: $MESSAGE_STATUS"
echo "Body:   $MESSAGE_BODY"

if [ "$MESSAGE_STATUS" != "200" ]; then
  echo "❌ Create message failed (expected 200)"
  exit 1
fi

# Extract message ID from response (response is just the UUID string directly)
MESSAGE_ID=$(echo "$MESSAGE_BODY" | sed 's/"//g')
if [ -z "$MESSAGE_ID" ]; then
  echo "❌ Did not receive a message id"
  exit 1
fi
echo "✅ Create message PASS - message_id: $MESSAGE_ID"

# 6) List messages in the session
echo -e "\n6) List messages in the session:"
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

# Verify our created message appears in the list
if ! echo "$MESSAGES_BODY" | grep -q "$MESSAGE_CONTENT"; then
  echo "❌ Created message not found in message list"
  exit 1
fi
echo "✅ List messages PASS"

# 7) Create another message to test multiple messages
echo -e "\n7) Create another message in the session:"
MESSAGE_CONTENT_2="This is a second test message for session $SESSION_ID"
CREATE_MESSAGE_2_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$SESSION_ENDPOINT/$SESSION_ID/messages" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d "{\"content\": \"$MESSAGE_CONTENT_2\"}")

MESSAGE_2_BODY=$(echo "$CREATE_MESSAGE_2_RESPONSE" | sed '$d')
MESSAGE_2_STATUS=$(echo "$CREATE_MESSAGE_2_RESPONSE" | tail -n1)

echo "Status: $MESSAGE_2_STATUS"
echo "Body:   $MESSAGE_2_BODY"

if [ "$MESSAGE_2_STATUS" != "200" ]; then
  echo "❌ Create second message failed (expected 200)"
  exit 1
fi
echo "✅ Create second message PASS"

# 8) Verify both messages are in the session
echo -e "\n8) Verify both messages are in the session:"
LIST_MESSAGES_2_RESPONSE=$(curl -sS -w "\n%{http_code}" -X GET "$SESSION_ENDPOINT/$SESSION_ID/messages" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

MESSAGES_2_BODY=$(echo "$LIST_MESSAGES_2_RESPONSE" | sed '$d')
MESSAGES_2_STATUS=$(echo "$LIST_MESSAGES_2_RESPONSE" | tail -n1)

echo "Status: $MESSAGES_2_STATUS"
echo "Body:   $MESSAGES_2_BODY"

if [ "$MESSAGES_2_STATUS" != "200" ]; then
  echo "❌ List messages (second time) failed (expected 200)"
  exit 1
fi

# Verify both messages appear in the list
if ! echo "$MESSAGES_2_BODY" | grep -q "$MESSAGE_CONTENT"; then
  echo "❌ First message not found in message list"
  exit 1
fi

if ! echo "$MESSAGES_2_BODY" | grep -q "$MESSAGE_CONTENT_2"; then
  echo "❌ Second message not found in message list"
  exit 1
fi
echo "✅ Both messages found in session PASS"

# 9) Test unauthorized access (without authorization header)
echo -e "\n9) Test unauthorized access (without authorization header):"
UNAUTHORIZED_RESPONSE=$(curl -sS -w "\n%{http_code}" -X GET "$SESSION_ENDPOINT/$SESSION_ID/messages" \
  -H "Content-Type: application/json")

UNAUTHORIZED_BODY=$(echo "$UNAUTHORIZED_RESPONSE" | sed '$d')
UNAUTHORIZED_STATUS=$(echo "$UNAUTHORIZED_RESPONSE" | tail -n1)

echo "Status: $UNAUTHORIZED_STATUS"
echo "Body:   $UNAUTHORIZED_BODY"

if [ "$UNAUTHORIZED_STATUS" != "401" ]; then
  echo "❌ Unauthorized access should return 401 (got $UNAUTHORIZED_STATUS)"
  exit 1
fi
echo "✅ Unauthorized access properly rejected PASS"

# 10) Test access to non-existent session
echo -e "\n10) Test access to non-existent session:"
NONEXISTENT_SESSION_ID="00000000-0000-0000-0000-000000000000"
NONEXISTENT_RESPONSE=$(curl -sS -w "\n%{http_code}" -X GET "$SESSION_ENDPOINT/$NONEXISTENT_SESSION_ID/messages" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

NONEXISTENT_BODY=$(echo "$NONEXISTENT_RESPONSE" | sed '$d')
NONEXISTENT_STATUS=$(echo "$NONEXISTENT_RESPONSE" | tail -n1)

echo "Status: $NONEXISTENT_STATUS"
echo "Body:   $NONEXISTENT_BODY"

if [ "$NONEXISTENT_STATUS" != "401" ]; then
  echo "❌ Access to non-existent session should return 401 (got $NONEXISTENT_STATUS)"
  exit 1
fi
echo "✅ Non-existent session access properly rejected PASS"

echo -e "\n🎉 All Session API comprehensive tests passed!"
echo "✅ Session creation and management"
echo "✅ Message creation and listing"
echo "✅ Multiple messages in session"
echo "✅ Authorization and access control"
echo "✅ Error handling for invalid requests"
