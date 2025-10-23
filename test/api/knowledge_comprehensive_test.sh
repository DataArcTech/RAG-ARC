#!/usr/bin/env bash

set -euo pipefail

API_BASE="http://localhost:8005"
KNOWLEDGE_ENDPOINT="$API_BASE/knowledge"
AUTH_ENDPOINT="$API_BASE/auth"

echo "Testing Knowledge REST API Comprehensive Flow at $KNOWLEDGE_ENDPOINT"
echo "=================================================================="

# 0) Health check
echo "0) Health check:"
curl -sS "$API_BASE/" | grep -q "ok" && echo "✅ Health check PASS" || { echo "❌ Health check failed"; exit 1; }

# 1) Ensure test user exists and login to get authentication token
echo -e "\n1) Ensure test user exists and login:"

# First, try to login
LOGIN_PRECHECK_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$AUTH_ENDPOINT/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=test_user&password=test_password")

LOGIN_PRECHECK_BODY=$(echo "$LOGIN_PRECHECK_RESPONSE" | sed '$d')
LOGIN_PRECHECK_STATUS=$(echo "$LOGIN_PRECHECK_RESPONSE" | tail -n1)

if [ "$LOGIN_PRECHECK_STATUS" = "200" ]; then
  echo "✅ User test_user already exists"
  ACCESS_TOKEN=$(echo "$LOGIN_PRECHECK_BODY" | grep -o '"access_token":"[^"]*"' | sed 's/"access_token":"//' | sed 's/"//')
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

  # Now login to get token
  LOGIN_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$AUTH_ENDPOINT/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=test_user&password=test_password")

  LOGIN_BODY=$(echo "$LOGIN_RESPONSE" | sed '$d')
  LOGIN_STATUS=$(echo "$LOGIN_RESPONSE" | tail -n1)

  echo "Login Status: $LOGIN_STATUS"
  echo "Login Body:   $LOGIN_BODY"

  if [ "$LOGIN_STATUS" != "200" ]; then
    echo "❌ Login failed (expected 200)"
    exit 1
  fi

  ACCESS_TOKEN=$(echo "$LOGIN_BODY" | grep -o '"access_token":"[^"]*"' | sed 's/"access_token":"//' | sed 's/"//')
fi

if [ -z "$ACCESS_TOKEN" ]; then
  echo "❌ Did not receive an access token"
  exit 1
fi
echo "✅ Authentication PASS - access_token: ${ACCESS_TOKEN:0:20}..."

# 2) Use test html file for upload
TEST_FILE="./test/test2.html"

echo -e "\n2) Upload file: $TEST_FILE"
UPLOAD_RESPONSE=$(curl -sS -w "\n%{http_code}" -F "file=@$TEST_FILE;type=application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" "$KNOWLEDGE_ENDPOINT")

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

# 3) Test list files functionality
echo -e "\n3) Test list files functionality"

# Upload a second test file for list files testing
TEST_FILE_2="./test/test.json"
echo "Uploading second test file: $TEST_FILE_2"
UPLOAD_RESPONSE_2=$(curl -sS -w "\n%{http_code}" -F "file=@$TEST_FILE_2;type=application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" "$KNOWLEDGE_ENDPOINT")

UPLOAD_BODY_2=$(echo "$UPLOAD_RESPONSE_2" | sed '$d')
UPLOAD_STATUS_2=$(echo "$UPLOAD_RESPONSE_2" | tail -n1)

echo "Upload 2 Status: $UPLOAD_STATUS_2"
echo "Upload 2 Body:   $UPLOAD_BODY_2"

if [ "$UPLOAD_STATUS_2" != "201" ]; then
  echo "❌ Second upload failed (expected 201)"
  exit 1
fi

FILE_ID_2=$(echo "$UPLOAD_BODY_2" | tr -d '"')
echo "✅ Second upload PASS - file_id: $FILE_ID_2"

# Wait for files to be processed
echo "⏳ Waiting for files to be processed (2 seconds)..."
sleep 2

# Test list all files
echo "Testing list all files..."
LIST_RESPONSE=$(curl -sS -w "\n%{http_code}" -X GET "$KNOWLEDGE_ENDPOINT/list_files" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

LIST_BODY=$(echo "$LIST_RESPONSE" | sed '$d')
LIST_STATUS=$(echo "$LIST_RESPONSE" | tail -n1)

echo "List Status: $LIST_STATUS"
echo "List Response:"
echo "$LIST_BODY" | python3 -m json.tool

if [ "$LIST_STATUS" != "200" ]; then
  echo "❌ List files failed (expected 200)"
  exit 1
fi

# Validate list response structure
echo "Validating list response structure..."
echo "$LIST_BODY" | python3 -c "
import sys
import json

try:
    data = json.load(sys.stdin)
    
    # Check if response has expected structure
    assert 'files' in data, 'Missing files field'
    assert 'total' in data, 'Missing total field'
    assert isinstance(data['files'], list), 'files should be a list'
    assert data['total'] >= 2, f'Expected at least 2 files, got {data[\"total\"]}'
    
    # Check first file structure
    if len(data['files']) > 0:
        file = data['files'][0]
        required_fields = ['file_id', 'filename', 'status', 'created_at', 'updated_at', 'file_size', 'content_type']
        for field in required_fields:
            assert field in file, f'Missing field: {field}'
        
        print('✅ Response structure is valid')
        print(f'✅ Total files: {data[\"total\"]}')
        print(f'✅ File status values: {[f[\"status\"] for f in data[\"files\"]]}')
    else:
        print('❌ No files returned')
        sys.exit(1)
        
except AssertionError as e:
    print(f'❌ Validation failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
  echo "❌ List files validation failed"
  exit 1
fi

# Test pagination
echo "Testing pagination (limit=1, offset=0)..."
LIST_RESPONSE_PAGINATED=$(curl -sS -w "\n%{http_code}" -X GET "$KNOWLEDGE_ENDPOINT/list_files?limit=1&offset=0" \
  -H "Authorization: Bearer $ACCESS_TOKEN")

LIST_PAGINATED_BODY=$(echo "$LIST_RESPONSE_PAGINATED" | sed '$d')
LIST_PAGINATED_STATUS=$(echo "$LIST_RESPONSE_PAGINATED" | tail -n1)

echo "Paginated Status: $LIST_PAGINATED_STATUS"
echo "Paginated Response:"
echo "$LIST_PAGINATED_BODY" | python3 -m json.tool

if [ "$LIST_PAGINATED_STATUS" != "200" ]; then
  echo "❌ Paginated list files failed (expected 200)"
  exit 1
fi

# Extract total counts and verify they match
echo "Verifying total counts match between paginated and non-paginated requests..."
TOTAL_FROM_PAGINATED=$(echo "$LIST_PAGINATED_BODY" | python3 -c "
import sys
import json
try:
    data = json.load(sys.stdin)
    print(data.get('total', 0))
except:
    print(0)
")

TOTAL_FROM_NON_PAGINATED=$(echo "$LIST_BODY" | python3 -c "
import sys
import json
try:
    data = json.load(sys.stdin)
    print(data.get('total', 0))
except:
    print(0)
")

echo "Total from paginated request (limit=1, offset=0): $TOTAL_FROM_PAGINATED"
echo "Total from non-paginated request: $TOTAL_FROM_NON_PAGINATED"

if [ "$TOTAL_FROM_PAGINATED" != "$TOTAL_FROM_NON_PAGINATED" ]; then
  echo "❌ Total counts do not match! Paginated: $TOTAL_FROM_PAGINATED, Non-paginated: $TOTAL_FROM_NON_PAGINATED"
  exit 1
fi

echo "✅ Total counts match between paginated and non-paginated requests"

echo "✅ List files functionality PASS"

# 4) Download the file
echo -e "\n4) Download file: $FILE_ID"
DOWNLOAD_HEADERS=$(mktemp)
DOWNLOAD_FILE="/tmp/downloaded.json"

HTTP_CODE=$(curl -sS -D "$DOWNLOAD_HEADERS" -o "$DOWNLOAD_FILE" -w "%{http_code}" \
  -H "Authorization: Bearer $ACCESS_TOKEN" "$KNOWLEDGE_ENDPOINT/$FILE_ID/download")

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
sleep 20

# 5) Test RAG inference chat functionality with uploaded content
echo -e "\n5) Test RAG inference chat with uploaded content"
SEARCH_QUERY="who is the author of Venom: The Black Suit Saga?"
echo "Searching for: '$SEARCH_QUERY'"

# Test chat endpoint to verify uploaded content is searchable
CHAT_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$API_BASE/rag_inference/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
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

# 5.1) Test full graph export functionality
echo -e "\n5.1) Test full graph export functionality"
echo "Endpoint: $KNOWLEDGE_ENDPOINT/graph/export"
echo "Token (first 20 chars): ${ACCESS_TOKEN:0:20}..."

FULL_GRAPH_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$KNOWLEDGE_ENDPOINT/graph/export" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{"max_nodes": 500, "max_edges": 2000}')

FULL_GRAPH_BODY=$(echo "$FULL_GRAPH_RESPONSE" | sed '$d')
FULL_GRAPH_STATUS=$(echo "$FULL_GRAPH_RESPONSE" | tail -n1)

echo "Full Graph Status: $FULL_GRAPH_STATUS"
if [ "$FULL_GRAPH_STATUS" != "200" ]; then
  echo "Response Body: $FULL_GRAPH_BODY"
fi

if [ "$FULL_GRAPH_STATUS" != "200" ]; then
  echo "❌ Full graph export failed (expected 200)"
  exit 1
fi

# Validate full graph response structure
echo "Validating full graph response structure..."
echo "$FULL_GRAPH_BODY" | python3 -c "
import sys
import json

try:
    data = json.load(sys.stdin)

    # Check if response has expected structure
    assert 'nodes' in data, 'Missing nodes field'
    assert 'edges' in data, 'Missing edges field'
    assert isinstance(data['nodes'], list), 'nodes should be a list'
    assert isinstance(data['edges'], list), 'edges should be a list'

    # Check node structure
    if len(data['nodes']) > 0:
        node = data['nodes'][0]
        assert 'id' in node, 'Node missing id field'
        assert 'type' in node, 'Node missing type field'
        print(f'✅ Full graph structure is valid')
        print(f'✅ Total nodes: {len(data[\"nodes\"])}')
        print(f'✅ Total edges: {len(data[\"edges\"])}')

        # Show node types distribution
        node_types = {}
        for node in data['nodes']:
            node_type = node.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        print(f'✅ Node types: {node_types}')
    else:
        print('⚠️  No nodes in full graph')

except AssertionError as e:
    print(f'❌ Validation failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
  echo "❌ Full graph validation failed"
  exit 1
fi

# Save full graph to file for inspection
FULL_GRAPH_FILE="/tmp/full_graph_export.json"
echo "$FULL_GRAPH_BODY" > "$FULL_GRAPH_FILE"
echo "✅ Full graph exported to: $FULL_GRAPH_FILE"

# 5.2) Test subgraph export functionality with RAG query
echo -e "\n5.2) Test subgraph export functionality with RAG query"
SUBGRAPH_QUERY="who is the author of Venom: The Black Suit Saga?"
echo "Querying for subgraph: '$SUBGRAPH_QUERY'"

SUBGRAPH_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$API_BASE/rag_inference/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d "{\"query\": \"$SUBGRAPH_QUERY\", \"return_subgraph\": true}")

SUBGRAPH_BODY=$(echo "$SUBGRAPH_RESPONSE" | sed '$d')
SUBGRAPH_STATUS=$(echo "$SUBGRAPH_RESPONSE" | tail -n1)

echo "Subgraph Status: $SUBGRAPH_STATUS"

if [ "$SUBGRAPH_STATUS" != "200" ]; then
  echo "❌ Subgraph export failed (expected 200)"
  exit 1
fi

# Validate subgraph response structure
echo "Validating subgraph response structure..."
echo "$SUBGRAPH_BODY" | python3 -c "
import sys
import json

try:
    data = json.load(sys.stdin)

    # Check if response has expected structure
    assert 'subgraph' in data, 'Missing subgraph field'
    subgraph = data['subgraph']

    assert 'nodes' in subgraph, 'Subgraph missing nodes field'
    assert 'edges' in subgraph, 'Subgraph missing edges field'
    assert isinstance(subgraph['nodes'], list), 'nodes should be a list'
    assert isinstance(subgraph['edges'], list), 'edges should be a list'

    # Check node structure
    if len(subgraph['nodes']) > 0:
        node = subgraph['nodes'][0]
        assert 'id' in node, 'Node missing id field'
        assert 'type' in node, 'Node missing type field'
        print(f'✅ Subgraph structure is valid')
        print(f'✅ Subgraph nodes: {len(subgraph[\"nodes\"])}')
        print(f'✅ Subgraph edges: {len(subgraph[\"edges\"])}')

        # Show node types distribution
        node_types = {}
        for node in subgraph['nodes']:
            node_type = node.get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        print(f'✅ Node types: {node_types}')

        # Check for seed entities if present
        seed_count = sum(1 for node in subgraph['nodes'] if node.get('is_seed', False))
        if seed_count > 0:
            print(f'✅ Seed entities: {seed_count}')

        # Check for PPR scores if present
        ppr_count = sum(1 for node in subgraph['nodes'] if 'ppr_score' in node)
        if ppr_count > 0:
            print(f'✅ Nodes with PPR scores: {ppr_count}')
    else:
        print('⚠️  No nodes in subgraph')

except AssertionError as e:
    print(f'❌ Validation failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'❌ Error: {e}')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
  echo "❌ Subgraph validation failed"
  exit 1
fi

# Save subgraph to file for inspection
SUBGRAPH_FILE="/tmp/subgraph_export.json"
echo "$SUBGRAPH_BODY" | python3 -c "
import sys
import json

data = json.load(sys.stdin)
if 'subgraph' in data:
    with open('$SUBGRAPH_FILE', 'w') as f:
        json.dump(data['subgraph'], f, indent=2, ensure_ascii=False)
"
echo "✅ Subgraph exported to: $SUBGRAPH_FILE"

# 6) Delete the file
echo -e "\n6) Delete file: $FILE_ID"
DELETE_CODE=$(curl -sS -o /dev/null -w "%{http_code}" -X DELETE \
  -H "Authorization: Bearer $ACCESS_TOKEN" "$KNOWLEDGE_ENDPOINT/$FILE_ID")
echo "Status: $DELETE_CODE"
if [ "$DELETE_CODE" != "204" ]; then
  echo "❌ Delete failed (expected 204)"
  exit 1
fi
echo "✅ Delete PASS"

# 7) Ensure download now returns 404
echo -e "\n7) Verify downloading deleted file returns 404"
CODE_AFTER_DELETE=$(curl -sS -o /dev/null -w "%{http_code}" \
  -H "Authorization: Bearer $ACCESS_TOKEN" "$KNOWLEDGE_ENDPOINT/$FILE_ID/download")
echo "Status: $CODE_AFTER_DELETE"
if [ "$CODE_AFTER_DELETE" != "404" ]; then
  echo "❌ Expected 404 when downloading deleted file"
  exit 1
fi
echo "✅ 404 after delete PASS"

# 8) Verify second delete returns 404 (non-existent)
echo -e "\n8) Re-delete should return 404"
SECOND_DELETE_CODE=$(curl -sS -o /dev/null -w "%{http_code}" -X DELETE \
  -H "Authorization: Bearer $ACCESS_TOKEN" "$KNOWLEDGE_ENDPOINT/$FILE_ID")
echo "Status: $SECOND_DELETE_CODE"
if [ "$SECOND_DELETE_CODE" != "404" ]; then
  echo "❌ Expected 404 on second delete"
  exit 1
fi
echo "✅ 404 on second delete PASS"

# 9) Verify deleted file content is no longer searchable
echo -e "\n9) Verify deleted file content is no longer searchable"
# Extract some content from the test file to search for
SEARCH_QUERY="who is the author of Venom: The Black Suit Saga?"
echo "Searching for: '$SEARCH_QUERY'"

# Perform search to verify the deleted file's content is not retrievable
SEARCH_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$API_BASE/rag_inference/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
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

# 10) Cleanup second test file
echo -e "\n10) Cleanup second test file: $FILE_ID_2"
DELETE_CODE_2=$(curl -sS -o /dev/null -w "%{http_code}" -X DELETE \
  -H "Authorization: Bearer $ACCESS_TOKEN" "$KNOWLEDGE_ENDPOINT/$FILE_ID_2")
echo "Delete 2 Status: $DELETE_CODE_2"
if [ "$DELETE_CODE_2" != "204" ]; then
  echo "❌ Second file delete failed (expected 204)"
  exit 1
fi
echo "✅ Second file cleanup PASS"

# Cleanup temporary files
rm -f "$DOWNLOAD_HEADERS" "$DOWNLOAD_FILE"

# Note: Graph export files are kept for inspection
echo -e "\n📊 Graph export files saved for inspection:"
echo "   - Full graph: $FULL_GRAPH_FILE"
echo "   - Subgraph: $SUBGRAPH_FILE"

echo -e "\n🎉 All Knowledge API comprehensive tests passed!"
