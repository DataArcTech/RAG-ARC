#!/usr/bin/env bash

set -euo pipefail

API_BASE="http://localhost:8005"
KNOWLEDGE_ENDPOINT="$API_BASE/knowledge"

echo "Testing Knowledge REST API Comprehensive Flow at $KNOWLEDGE_ENDPOINT"
echo "=================================================================="

# 0) Health check
echo "0) Health check:"
curl -sS "$API_BASE/" | grep -q "ok" && echo "✅ Health check PASS" || { echo "❌ Health check failed"; exit 1; }

# 1) Use test JSON file for upload
TEST_FILE="./test/test2.html"

echo -e "\n1) Upload file: $TEST_FILE"
UPLOAD_RESPONSE=$(curl -sS -w "\n%{http_code}" -F "file=@$TEST_FILE;type=application/json" "$KNOWLEDGE_ENDPOINT")

# Split body and status code
UPLOAD_BODY=$(echo "$UPLOAD_RESPONSE" | sed '$d')
UPLOAD_STATUS=$(echo "$UPLOAD_RESPONSE" | tail -n1)

echo "Status: $UPLOAD_STATUS"
echo "Body:   $UPLOAD_BODY"

if [ "$UPLOAD_STATUS" != "201" ]; then
  echo "❌ Upload failed (expected 201)"
  exit 1
fi

# Expect response body to be a UUID-like string (loose check: contains '-')
FILE_ID=$(echo "$UPLOAD_BODY" | tr -d '"')
if ! echo "$FILE_ID" | grep -q "-"; then
  echo "❌ Did not receive a file id"
  exit 1
fi
echo "✅ Upload PASS - file_id: $FILE_ID"

# 2) Download the file
echo -e "\n2) Download file: $FILE_ID"
DOWNLOAD_HEADERS=$(mktemp)
DOWNLOAD_FILE="/tmp/downloaded.json"

HTTP_CODE=$(curl -sS -D "$DOWNLOAD_HEADERS" -o "$DOWNLOAD_FILE" -w "%{http_code}" "$KNOWLEDGE_ENDPOINT/$FILE_ID/download")

echo "Status: $HTTP_CODE"
grep -i "content-disposition" "$DOWNLOAD_HEADERS" || true
grep -i "content-type" "$DOWNLOAD_HEADERS" || true

if [ "$HTTP_CODE" != "200" ]; then
  echo "❌ Download failed (expected 200)"
  exit 1
fi

# Verify content matches
if ! diff -q "$TEST_FILE" "$DOWNLOAD_FILE" > /dev/null; then
  echo "❌ Downloaded content does not match uploaded content"
  exit 1
fi
echo "✅ Download PASS"

# Wait for indexing before testing search functionality
echo -e "\n⏳ Waiting for indexing to complete (10 seconds)..."
sleep 10

# 3) Test RAG inference chat functionality with uploaded content
echo -e "\n3) Test RAG inference chat with uploaded content"
SEARCH_QUERY="who is the author of Venom: The Black Suit Saga?"
echo "Searching for: '$SEARCH_QUERY'"

# Test chat endpoint to verify uploaded content is searchable
CHAT_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$API_BASE/rag_inference/chat" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$SEARCH_QUERY\"}")

CHAT_BODY=$(echo "$CHAT_RESPONSE" | sed '$d')
CHAT_STATUS=$(echo "$CHAT_RESPONSE" | tail -n1)

echo "Chat Status: $CHAT_STATUS"
echo "Chat Response: $CHAT_BODY"

if [ "$CHAT_STATUS" != "200" ]; then
  echo "❌ Chat request failed (expected 200)"
  exit 1
fi

# Verify we get a non-empty response
if [ -z "$CHAT_BODY" ]; then
  echo "❌ Chat response is empty"
  exit 1
fi

# Check if the uploaded content appears in search results
if ! echo "$CHAT_BODY" | grep -q "Yuxuan Zhou"; then
  echo "❌ Uploaded file content not found in search results"
  exit 1
fi
echo "✅ RAG inference chat with uploaded content PASS"

# 4) Delete the file
echo -e "\n4) Delete file: $FILE_ID"
DELETE_CODE=$(curl -sS -o /dev/null -w "%{http_code}" -X DELETE "$KNOWLEDGE_ENDPOINT/$FILE_ID")
echo "Status: $DELETE_CODE"
if [ "$DELETE_CODE" != "204" ]; then
  echo "❌ Delete failed (expected 204)"
  exit 1
fi
echo "✅ Delete PASS"

# 5) Ensure download now returns 404
echo -e "\n5) Verify downloading deleted file returns 404"
CODE_AFTER_DELETE=$(curl -sS -o /dev/null -w "%{http_code}" "$KNOWLEDGE_ENDPOINT/$FILE_ID/download")
echo "Status: $CODE_AFTER_DELETE"
if [ "$CODE_AFTER_DELETE" != "404" ]; then
  echo "❌ Expected 404 when downloading deleted file"
  exit 1
fi
echo "✅ 404 after delete PASS"

# 6) Verify second delete returns 404 (non-existent)
echo -e "\n6) Re-delete should return 404"
SECOND_DELETE_CODE=$(curl -sS -o /dev/null -w "%{http_code}" -X DELETE "$KNOWLEDGE_ENDPOINT/$FILE_ID")
echo "Status: $SECOND_DELETE_CODE"
if [ "$SECOND_DELETE_CODE" != "404" ]; then
  echo "❌ Expected 404 on second delete"
  exit 1
fi
echo "✅ 404 on second delete PASS"

# 7) Verify deleted file content is no longer searchable
echo -e "\n7) Verify deleted file content is no longer searchable"
# Extract some content from the test file to search for
SEARCH_QUERY="who is the author of Venom: The Black Suit Saga?"
echo "Searching for: '$SEARCH_QUERY'"

# Perform search to verify the deleted file's content is not retrievable
SEARCH_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$API_BASE/rag_inference/chat" \
  -H "Content-Type: application/json" \
  -d "{\"query\": \"$SEARCH_QUERY\"}")

SEARCH_BODY=$(echo "$SEARCH_RESPONSE" | sed '$d')
SEARCH_STATUS=$(echo "$SEARCH_RESPONSE" | tail -n1)

echo "Search Status: $SEARCH_STATUS"
echo "Search Response: $SEARCH_BODY"

if [ "$SEARCH_STATUS" != "200" ]; then
  echo "❌ Search request failed (expected 200)"
  exit 1
fi

# Check if the deleted contents appears in search results
if echo "$SEARCH_BODY" | grep -q "Yuxuan Zhou"; then
  echo "❌ Deleted file still appears in search results"
  exit 1
fi
echo "✅ Deleted content no longer searchable PASS"

# Cleanup temporary files
rm -f "$DOWNLOAD_HEADERS" "$DOWNLOAD_FILE"

echo -e "\n🎉 All Knowledge API comprehensive tests passed!"
