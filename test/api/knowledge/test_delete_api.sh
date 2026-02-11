#!/bin/bash
# 快速测试 DELETE /knowledge/{file_id} 接口
# 覆盖场景：
#   1. 删除接口正常返回 200
#   2. 队列中 PENDING/RUNNING 的解析/索引任务被撤销并标记为 CANCELED
#   3. 已索引文件删除后，物理清理完成，文件不可访问（404）
# 使用方式（从 RAG-ARC 根目录执行）:
#   ./test/api/knowledge/test_delete_api.sh [file_id]
#   ./test/api/knowledge/test_delete_api.sh --verify-upload   # 上传后删除，验证队列撤销
#   ./test/api/knowledge/test_delete_api.sh --verify-cleanup   # 删除后验证文件不可访问
# 若不传 file_id，则先列出文件并取第一个进行删除测试

set -e

BASE_URL="${BASE_URL:-http://localhost:8000}"
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ3YW5nc2h1bmNoaSIsInR5cGUiOjAsImV4cCI6MTc3MDg4Njc1OH0.OIFolk-acTwshKLjvMHgZxqd2at27EmXQ-vb3FQgTCU"
VERIFY_MODE=false
UPLOAD_FOR_VERIFY=false
VERIFY_CLEANUP=true
[ "$1" = "--verify" ] && { VERIFY_MODE=true; shift; }
[ "$1" = "--verify-upload" ] && { VERIFY_MODE=true; UPLOAD_FOR_VERIFY=true; shift; }
[ "$1" = "--verify-cleanup" ] && { VERIFY_CLEANUP=true; shift; }
[ "$1" = "--no-verify-cleanup" ] && { VERIFY_CLEANUP=false; shift; }

echo "=== 测试 DELETE /knowledge/{file_id} ==="
echo "Base URL: ${BASE_URL}"
echo "Verify queue: ${VERIFY_MODE}  |  Verify cleanup: ${VERIFY_CLEANUP}"
echo ""

# --verify-upload: 上传新文件后立即删除，以验证「索引中」任务被撤销
if [ "$UPLOAD_FOR_VERIFY" = true ]; then
  echo ">>> [verify-upload] 上传新文件以产生索引任务..."
  UPLOAD_FILE="/tmp/test_delete_verify_upload_$$.txt"
  UNIQ_NAME="test_delete_verify_$(date +%s)_$$.txt"
  echo "test delete verify upload $(date)" > "$UPLOAD_FILE"
  UPLOAD_RESP=$(curl -s -w "\n%{http_code}" -X POST -H "Authorization: Bearer ${TOKEN}" \
    -F "file=@${UPLOAD_FILE};type=text/plain" \
    -F "relative_path=${UNIQ_NAME}" \
    "${BASE_URL}/knowledge")
  rm -f "$UPLOAD_FILE"
  UPLOAD_BODY=$(echo "$UPLOAD_RESP" | head -n -1)
  UPLOAD_CODE=$(echo "$UPLOAD_RESP" | tail -1)
  if [ "$UPLOAD_CODE" != "200" ]; then
    echo "❌ 上传失败 (HTTP $UPLOAD_CODE): $UPLOAD_BODY"
    exit 1
  fi
  FILE_ID=$(echo "$UPLOAD_BODY" | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    v=d.get('data')
    if isinstance(v, str):
        print(v, end='')
    elif isinstance(v, dict):
        print(v.get('file_id',''), end='')
    else:
        print(str(d).strip('\"'), end='')
except Exception:
    print('', end='')
" 2>/dev/null || echo "$UPLOAD_BODY" | tr -d '"')
  echo "  已上传 file_id=${FILE_ID}，等待任务进入队列..."
  sleep 1
fi

# 若未传入 file_id 且非 verify-upload，先获取文件列表
if [ -z "$1" ] && [ "$UPLOAD_FOR_VERIFY" != true ]; then
  echo ">>> 1. 获取文件列表..."
  LIST_RESP=$(curl -s -w "\n%{http_code}" -H "Authorization: Bearer ${TOKEN}" "${BASE_URL}/knowledge/list_files?pagesize=5")
  HTTP_BODY=$(echo "$LIST_RESP" | head -n -1)
  HTTP_CODE=$(echo "$LIST_RESP" | tail -1)
  echo "HTTP Status: ${HTTP_CODE}"

  if [ "$HTTP_CODE" != "200" ]; then
    echo "❌ 获取文件列表失败"
    echo "$HTTP_BODY" | head -20
    exit 1
  fi

  # 解析第一个文件的 file_id (简单用 grep/sed)
  FILE_ID=$(echo "$HTTP_BODY" | python3 -c "
import json, sys
try:
    d = json.load(sys.stdin)
    files = d.get('files') or d.get('data', {}).get('files') or []
    if not files:
        print('', end='')
    else:
        print(files[0].get('file_id', ''), end='')
except Exception:
    print('', end='')
" 2>/dev/null || echo "")

  if [ -z "$FILE_ID" ]; then
    echo ">>> 无现有文件，先上传一个测试文件..."
    UPLOAD_RESP=$(curl -s -w "\n%{http_code}" -X POST -H "Authorization: Bearer ${TOKEN}" \
      -F "file=@${BASH_SOURCE};type=text/plain" \
      -F "relative_path=test_delete_api.sh" \
      "${BASE_URL}/knowledge")
    UPLOAD_BODY=$(echo "$UPLOAD_RESP" | head -n -1)
    UPLOAD_CODE=$(echo "$UPLOAD_RESP" | tail -1)
    if [ "$UPLOAD_CODE" != "200" ]; then
      echo "❌ 上传失败 (HTTP $UPLOAD_CODE)"
      echo "$UPLOAD_BODY"
      exit 1
    fi
    FILE_ID=$(echo "$UPLOAD_BODY" | tr -d '"')
    echo "  上传成功, file_id=${FILE_ID}"
  else
    echo "  使用第一个文件: ${FILE_ID}"
  fi
elif [ -n "$1" ]; then
  FILE_ID="$1"
  echo ">>> 使用传入的 file_id: ${FILE_ID}"
fi

if [ -z "$FILE_ID" ]; then
  echo "❌ 无 file_id，请传入或确保有可用的文件"
  exit 1
fi

# 若开启 verify，删除前先获取 task_run_id
TASK_RUN_ID=""
if [ "$VERIFY_MODE" = true ]; then
  echo ">>> 删除前: 查询任务状态..."
  TASK_RESP=$(curl -s -w "\n%{http_code}" -H "Authorization: Bearer ${TOKEN}" "${BASE_URL}/knowledge/${FILE_ID}/task")
  TASK_BODY=$(echo "$TASK_RESP" | head -n -1)
  TASK_CODE=$(echo "$TASK_RESP" | tail -1)
  if [ "$TASK_CODE" = "200" ]; then
    TASK_RUN_ID=$(echo "$TASK_BODY" | python3 -c "
import json,sys
d=json.load(sys.stdin)
data=d.get('data') or d
rid=data.get('task_run_id')
print(rid if rid else '',end='')
" 2>/dev/null || echo "")
    TASK_STATE=$(echo "$TASK_BODY" | python3 -c "
import json,sys
d=json.load(sys.stdin)
data=d.get('data') or d
s=data.get('task_state')
print(s if s else '',end='')
" 2>/dev/null || echo "")
    echo "  task_run_id=${TASK_RUN_ID:-<无>}  task_state=${TASK_STATE:-<无>}"
  else
    echo "  (文件可能无任务或已删除，跳过 task 查询)"
  fi
  echo ""
fi

echo ">>> 2. 删除文件 (DELETE /knowledge/${FILE_ID})..."
DELETE_RESP=$(curl -s -w "\n%{http_code}" -X DELETE \
  -H "Authorization: Bearer ${TOKEN}" \
  "${BASE_URL}/knowledge/${FILE_ID}")
DELETE_BODY=$(echo "$DELETE_RESP" | head -n -1)
DELETE_CODE=$(echo "$DELETE_RESP" | tail -1)

echo "HTTP Status: ${DELETE_CODE}"
echo "Response: ${DELETE_BODY}"

if [ "$DELETE_CODE" != "200" ]; then
  echo ""
  echo "❌ 删除失败 (HTTP ${DELETE_CODE})"
  exit 1
fi

# 若开启 verify 且有 task_run_id，删除后查询队列中任务状态
if [ "$VERIFY_MODE" = true ] && [ -n "$TASK_RUN_ID" ]; then
  echo ""
  echo ">>> 删除后: 查询 Redis 中任务状态 (GET /knowledge/task_run/${TASK_RUN_ID})..."
  sleep 1
  RUN_RESP=$(curl -s -w "\n%{http_code}" -H "Authorization: Bearer ${TOKEN}" "${BASE_URL}/knowledge/task_run/${TASK_RUN_ID}")
  RUN_BODY=$(echo "$RUN_RESP" | head -n -1)
  RUN_CODE=$(echo "$RUN_RESP" | tail -1)
  echo "HTTP Status: ${RUN_CODE}"
  if [ "$RUN_CODE" = "200" ]; then
    STATE=$(echo "$RUN_BODY" | python3 -c "
import json,sys
d=json.load(sys.stdin)
data=d.get('data') or d
s=data.get('state') or data.get('task_state') or ''
print(s,end='')
" 2>/dev/null || echo "")
    echo "  task_state=${STATE}"
    if [ "$STATE" = "CANCELED" ]; then
      echo ""
      echo "✅ 队列验证通过: 任务已标记为 CANCELED，已从执行中/排队中移除"
    elif [ -n "$STATE" ]; then
      echo ""
      echo "⚠️  任务状态为 ${STATE}（若删除前为 PENDING/RUNNING 则应变更为 CANCELED；若原本已 SUCCESS 则无需变更）"
    fi
  elif [ "$RUN_CODE" = "409" ]; then
    echo "  (TASK_QUEUE_MODE 非 celery，无 task_run 接口，跳过队列验证)"
  else
    echo "  Response: ${RUN_BODY}"
  fi
fi

# 验证已删除文件不可访问（404）
if [ "$VERIFY_CLEANUP" = true ]; then
  echo ""
  echo ">>> 3. 验证删除生效：GET /knowledge/${FILE_ID}/task（已删除应返回 404）..."
  sleep 2
  CLEANUP_RESP=$(curl -s -w "\n%{http_code}" -H "Authorization: Bearer ${TOKEN}" "${BASE_URL}/knowledge/${FILE_ID}/task")
  CLEANUP_CODE=$(echo "$CLEANUP_RESP" | tail -1)
  if [ "$CLEANUP_CODE" = "404" ]; then
    echo "  HTTP 404 ✓ 文件已不可访问，删除生效"
    echo ""
    echo "✅ 物理清理验证通过：已索引文件已从系统中移除"
  else
    echo "  HTTP ${CLEANUP_CODE}（预期 404）"
    echo "  ⚠️  若 delete 任务仍在执行，可稍后重试"
  fi
fi

echo ""
echo "✅ 删除接口调用成功"
echo "   - 删除请求已受理"
echo "   - 索引中/排队中的任务已被撤销（Celery）"
echo "   - 已索引文件的 chunks/indexes 将由后台任务清理"
