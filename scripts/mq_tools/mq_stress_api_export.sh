#!/usr/bin/env bash
set -euo pipefail

API_BASE="${API_BASE:-http://localhost:8000}"
AUTH_ENDPOINT="$API_BASE/auth"
KNOWLEDGE_ENDPOINT="$API_BASE/knowledge"

TOTAL="${1:-30}"
CONCURRENCY="${2:-10}"

USERNAME="${MQ_STRESS_USERNAME:-mq_stress_user}"
PASSWORD="${MQ_STRESS_PASSWORD:-mq_stress_password}"

echo "MQ API stress: export_async total=${TOTAL} concurrency=${CONCURRENCY} api=${API_BASE}"

curl -sS "$API_BASE/" | grep -q "ok" || { echo "health check failed: ${API_BASE}/"; exit 1; }

LOGIN_PRECHECK_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$AUTH_ENDPOINT/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=${USERNAME}&password=${PASSWORD}")

LOGIN_PRECHECK_BODY=$(echo "$LOGIN_PRECHECK_RESPONSE" | sed '$d')
LOGIN_PRECHECK_STATUS=$(echo "$LOGIN_PRECHECK_RESPONSE" | tail -n1)

if [ "$LOGIN_PRECHECK_STATUS" = "200" ]; then
  ACCESS_TOKEN=$(echo "$LOGIN_PRECHECK_BODY" | grep -o '"access_token":"[^"]*"' | sed 's/"access_token":"//' | sed 's/"//')
else
  echo "Registering ${USERNAME}..."
  REGISTER_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$AUTH_ENDPOINT/register" \
    -H "Content-Type: application/json" \
    -d "{\"name\":\"MQ Stress\",\"user_name\":\"${USERNAME}\",\"password\":\"${PASSWORD}\"}")
  REGISTER_STATUS=$(echo "$REGISTER_RESPONSE" | tail -n1)
  if [ "$REGISTER_STATUS" != "201" ]; then
    echo "register failed: status=${REGISTER_STATUS}"
    echo "$REGISTER_RESPONSE"
    exit 1
  fi

  LOGIN_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST "$AUTH_ENDPOINT/token" \
    -H "Content-Type: application/x-www-form-urlencoded" \
    -d "username=${USERNAME}&password=${PASSWORD}")
  LOGIN_BODY=$(echo "$LOGIN_RESPONSE" | sed '$d')
  LOGIN_STATUS=$(echo "$LOGIN_RESPONSE" | tail -n1)
  if [ "$LOGIN_STATUS" != "200" ]; then
    echo "login failed: status=${LOGIN_STATUS}"
    echo "$LOGIN_BODY"
    exit 1
  fi
  ACCESS_TOKEN=$(echo "$LOGIN_BODY" | grep -o '"access_token":"[^"]*"' | sed 's/"access_token":"//' | sed 's/"//')
fi

if [ -z "${ACCESS_TOKEN:-}" ]; then
  echo "no access token"
  exit 1
fi

tmpdir="$(mktemp -d)"
run_ids_file="${tmpdir}/run_ids.txt"
touch "${run_ids_file}"

export ACCESS_TOKEN KNOWLEDGE_ENDPOINT

echo "Enqueueing export tasks..."
seq 1 "${TOTAL}" | xargs -P "${CONCURRENCY}" -I{} bash -c '
  resp=$(curl -sS -w "\n%{http_code}" -X POST "${KNOWLEDGE_ENDPOINT}/graph/export_async" \
    -H "Authorization: Bearer ${ACCESS_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "{\"max_nodes\":50,\"max_edges\":200}")
  body=$(echo "$resp" | sed "$d")
  status=$(echo "$resp" | tail -n1)
  if [ "$status" != "202" ]; then
    echo "enqueue failed status=${status} body=${body}" >&2
    exit 1
  fi
  run_id=$(echo "$body" | grep -o "\"run_id\":\"[^\"]*\"" | head -n1 | sed "s/\"run_id\":\"//" | sed "s/\"//")
  if [ -z "$run_id" ]; then
    echo "missing run_id body=${body}" >&2
    exit 1
  fi
  echo "$run_id"
' >> "${run_ids_file}"

echo "Polling results..."
export API_BASE
ok=0
failed=0
while read -r run_id; do
  if [ -z "$run_id" ]; then
    continue
  fi
  # Poll up to ~60s per task.
  for _ in $(seq 1 120); do
    res=$(curl -sS -w "\n%{http_code}" -H "Authorization: Bearer ${ACCESS_TOKEN}" "${KNOWLEDGE_ENDPOINT}/result/${run_id}" || true)
    code=$(echo "$res" | tail -n1)
    if [ "$code" = "200" ]; then
      ok=$((ok+1))
      break
    fi
    if [ "$code" = "409" ]; then
      sleep 0.5
      continue
    fi
    failed=$((failed+1))
    echo "result failed run_id=${run_id} code=${code} body=$(echo "$res" | sed '$d')" >&2
    break
  done
done < "${run_ids_file}"

echo "done ok=${ok} failed=${failed} total=${TOTAL}"

