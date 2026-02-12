#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${TEST_OWNER_ID:=818eaeef-bed4-4b58-8d4e-e86addc8030e}"
: "${BENCH_OWNER_ID:=212ee819-7e22-4f5d-94bd-30146e514f60}"
: "${SOURCE_OWNER_ID:=$TEST_OWNER_ID}"
: "${TEST_FILE_IDS:=fd16f299-32fe-4390-8b7d-4262e1786bf2,58950123-eae7-4554-91cd-8960bafce174}"
: "${BENCH_PDFS:=docs-proj/test_pdfs/「星鑽」儲蓄壽險計劃 II-產品資料冊.pdf;docs-proj/test_pdfs/星鑽儲蓄壽險計劃II-小册子.pdf}"
: "${REBUILD_REPORT_PATH:=local/bench/rebuild_deepsearch_env_report.json}"
: "${REBUILD_RESET_BENCH_OWNER:=1}"
: "${REBUILD_FAST_MODE:=1}"
: "${PARSER_PARSE_MODE:=native}"

IFS=',' read -r -a TEST_FILE_ID_ARRAY <<<"$TEST_FILE_IDS"
IFS=';' read -r -a BENCH_PDF_ARRAY <<<"$BENCH_PDFS"

cmd=(
  uv run python scripts/indexing/rebuild_deepsearch_env.py
  --test-owner-id "$TEST_OWNER_ID"
  --bench-owner-id "$BENCH_OWNER_ID"
  --source-owner-id "$SOURCE_OWNER_ID"
  --report-path "$REBUILD_REPORT_PATH"
)

for fid in "${TEST_FILE_ID_ARRAY[@]}"; do
  token="${fid// /}"
  [[ -n "$token" ]] && cmd+=(--test-file-id "$token")
done

for pdf in "${BENCH_PDF_ARRAY[@]}"; do
  token="${pdf#${pdf%%[![:space:]]*}}"
  token="${token%${token##*[![:space:]]}}"
  [[ -n "$token" ]] && cmd+=(--bench-pdf "$token")
done

if [[ "$REBUILD_RESET_BENCH_OWNER" == "1" ]]; then
  cmd+=(--reset-bench-owner)
else
  cmd+=(--no-reset-bench-owner)
fi

if [[ "$REBUILD_FAST_MODE" == "1" ]]; then
  cmd+=(--fast-rebuild-mode)
else
  cmd+=(--no-fast-rebuild-mode)
fi

echo "[rebuild] running from $ROOT_DIR"
echo "[rebuild] test_owner=$TEST_OWNER_ID bench_owner=$BENCH_OWNER_ID source_owner=$SOURCE_OWNER_ID"
echo "[rebuild] parser_mode=$PARSER_PARSE_MODE reset_bench=$REBUILD_RESET_BENCH_OWNER fast_mode=$REBUILD_FAST_MODE"
"${cmd[@]}"

echo "[rebuild] completed. report -> $REBUILD_REPORT_PATH"
