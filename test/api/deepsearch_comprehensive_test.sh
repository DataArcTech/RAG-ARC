#!/usr/bin/env bash

set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8000}"
AUTH_ENDPOINT="$API_BASE/auth"
DEEPSEARCH_ENDPOINT="$API_BASE/deepsearch"

echo "Testing DeepSearch API Comprehensive Flow at $DEEPSEARCH_ENDPOINT"
echo "==============================================================="

# 0) Health check
echo "0) Health check:"
curl -sS "$API_BASE/" | grep -q "ok" && echo "✅ Health check PASS" || { echo "❌ Health check failed"; exit 1; }

# 1) Ensure test user exists and login
echo -e "\n1) Ensure test user exists and login:"

LOGIN_PRECHECK_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$AUTH_ENDPOINT/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=deepsearch_user&password=deepsearch_password")

LOGIN_PRECHECK_BODY=$(echo "$LOGIN_PRECHECK_RESPONSE" | sed '$d')
LOGIN_PRECHECK_STATUS=$(echo "$LOGIN_PRECHECK_RESPONSE" | tail -n1)

if [ "$LOGIN_PRECHECK_STATUS" = "200" ]; then
  echo "✅ User deepsearch_user already exists"
  ACCESS_TOKEN=$(echo "$LOGIN_PRECHECK_BODY" | grep -o '"access_token":"[^"]*"' | sed 's/"access_token":"//' | sed 's/"//')
else
  echo "Registering deepsearch_user since login failed..."
  REGISTER_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$AUTH_ENDPOINT/register" \
    -H "Content-Type: application/json" \
    -d '{"name": "DeepSearch User", "user_name": "deepsearch_user", "password": "deepsearch_password"}')
  REGISTER_STATUS=$(echo "$REGISTER_RESPONSE" | tail -n1)
  if [ "$REGISTER_STATUS" != "201" ] && [ "$REGISTER_STATUS" != "400" ]; then
    echo "❌ User registration failed (expected 201/400, got $REGISTER_STATUS)"
    exit 1
  fi
  LOGIN_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$AUTH_ENDPOINT/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=deepsearch_user&password=deepsearch_password")
  LOGIN_BODY=$(echo "$LOGIN_RESPONSE" | sed '$d')
  LOGIN_STATUS=$(echo "$LOGIN_RESPONSE" | tail -n1)
  if [ "$LOGIN_STATUS" != "200" ]; then
    echo "❌ Login failed (expected 200)"
    exit 1
  fi
  ACCESS_TOKEN=$(echo "$LOGIN_BODY" | grep -o '"access_token":"[^"]*"' | sed 's/"access_token":"//' | sed 's/"//')
fi

if [ -z "${ACCESS_TOKEN:-}" ]; then
  echo "❌ Did not receive an access token"
  exit 1
fi
echo "✅ Authentication PASS - access_token: ${ACCESS_TOKEN:0:20}..."

# 2) Schedule deepsearch run
echo -e "\n2) Schedule deepsearch run_async:"
QUESTION="What is RAG-ARC?"
RUN_ASYNC_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$DEEPSEARCH_ENDPOINT/run_async" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d "{\"question\": \"$QUESTION\", \"include_evidence\": true}")

RUN_ASYNC_BODY=$(echo "$RUN_ASYNC_RESPONSE" | sed '$d')
RUN_ASYNC_STATUS=$(echo "$RUN_ASYNC_RESPONSE" | tail -n1)

echo "run_async Status: $RUN_ASYNC_STATUS"
if [ "$RUN_ASYNC_STATUS" != "202" ]; then
  echo "Body: $RUN_ASYNC_BODY"
  echo "❌ Expected 202 from run_async"
  exit 1
fi

RUN_ID=$(echo "$RUN_ASYNC_BODY" | python3 -c 'import sys, json; obj=json.load(sys.stdin); print((obj.get("run_id") if isinstance(obj, dict) else "") or ((obj.get("data") or {}).get("run_id","") if isinstance(obj, dict) else "") )')
if [ -z "$RUN_ID" ]; then
  echo "❌ Did not receive run_id"
  exit 1
fi
echo "✅ Scheduled run_id: $RUN_ID"

# 3) Stream progress (OpenAI-format SSE) and assert tool-call progress appears
echo -e "\n3) Stream progress via SSE (openai format):"
cat > /tmp/deepsearch_sse_reader.py << 'EOF'
#!/usr/bin/env python3
import json
import sys
import httpx

def read(url: str, token: str) -> dict:
    headers = {"Authorization": f"Bearer {token}"}
    progress_hits = 0
    done = False
    with httpx.stream("GET", url, headers=headers, timeout=180.0) as r:
        if r.status_code != 200:
            body = r.read().decode("utf-8", errors="replace")
            return {"http_status": r.status_code, "body": body}
        for line in r.iter_lines():
            if not line or not line.startswith("data:"):
                continue
            data = line.split(":", 1)[1].strip()
            if data == "[DONE]":
                done = True
                break
            chunk = json.loads(data)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = (choices[0] or {}).get("delta") or {}
            for tc in delta.get("tool_calls") or []:
                fn = (tc or {}).get("function") or {}
                if fn.get("name") == "rag_arc_progress":
                    progress_hits += 1
    return {"http_status": 200, "done": done, "progress_hits": progress_hits}

if __name__ == "__main__":
    url = sys.argv[1]
    token = sys.argv[2]
    print(json.dumps(read(url, token), ensure_ascii=False))
EOF
chmod +x /tmp/deepsearch_sse_reader.py

STREAM_URL="$DEEPSEARCH_ENDPOINT/stream/$RUN_ID?format=openai"
STREAM_OUT=$(python3 /tmp/deepsearch_sse_reader.py "$STREAM_URL" "$ACCESS_TOKEN")
echo "$STREAM_OUT"
echo "$STREAM_OUT" | grep -q '"http_status": 200' || { echo "❌ Stream request failed"; exit 1; }
echo "$STREAM_OUT" | grep -q '"done": true' || { echo "❌ Stream did not finish"; exit 1; }
python3 - <<PY
import json
out=json.loads('''$STREAM_OUT''')
assert out.get('progress_hits',0) >= 1
print("✅ DeepSearch SSE progress tool-calls observed:", out.get('progress_hits'))
PY

rm -f /tmp/deepsearch_sse_reader.py

# 4) Poll result endpoint until done
echo -e "\n4) Poll result until done:"
DEADLINE=$(( $(date +%s) + 240 ))
while true; do
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "❌ Timed out waiting for deepsearch result"
    exit 1
  fi
  RESULT_RESPONSE=$(curl -sS -w "\n%{http_code}" -H "Authorization: Bearer $ACCESS_TOKEN" "$DEEPSEARCH_ENDPOINT/result/$RUN_ID")
  RESULT_BODY=$(echo "$RESULT_RESPONSE" | sed '$d')
  RESULT_STATUS=$(echo "$RESULT_RESPONSE" | tail -n1)
  if [ "$RESULT_STATUS" = "200" ]; then
    echo "✅ Result ready"
    echo "$RESULT_BODY" | python3 -c 'import sys, json; data=json.load(sys.stdin); assert data.get("done", True) in (True, "true"); print("✅ Result payload received")'
    break
  fi
  if [ "$RESULT_STATUS" = "409" ]; then
    sleep 2
    continue
  fi
  echo "❌ Unexpected result status: $RESULT_STATUS"
  echo "$RESULT_BODY"
  exit 1
done

echo -e "\n🎉 DeepSearch comprehensive tests passed!"
