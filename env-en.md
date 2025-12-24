# Environment Configuration (English)

All runtime behavior is controlled through `.env`. By default, `.env.example` already contains values that work for local development (Docker services on `localhost`). Only the model/API credentials typically require edits. This document describes every variable, grouped by subsystem.

## 1. Model & LLM Providers

| Variable | Default | Description |
| --- | --- | --- |
| `CHAT_MODEL_PROVIDER` | `openai` | Chat provider (`openai` = OpenAI-compatible API, `huggingface` = local Transformers). |
| `CHAT_MODEL_NAME` | _(empty)_ | Optional preferred chat model name (overrides `OPENAI_CHAT_MODEL` when set). |
| `CHAT_MODEL_DEVICE` | `cpu` | HuggingFace chat runtime device (used when `CHAT_MODEL_PROVIDER=huggingface`). |
| `CHAT_MODEL_CACHE_FOLDER` | _(empty)_ | Optional HuggingFace cache folder for chat weights/tokenizers. |
| `CHAT_API_KEY` | _(empty)_ | API key for chat provider (required for hosted APIs). |
| `CHAT_API_BASE_URL` | _(empty)_ | Base URL for OpenAI-compatible chat endpoints (e.g. `https://api.openai.com/v1`). |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Legacy/default chat model name used when `CHAT_MODEL_NAME` is empty. |
| `OPENAI_API_BASE` | _(empty)_ | Optional legacy alias for OpenAI-compatible base URL. |
| `EMBEDDING_MODEL_PROVIDER` | `openai` | Embedding provider (`openai` = OpenAI-compatible API, `huggingface` = local SentenceTransformers). |
| `EMBEDDING_API_KEY` | _(empty)_ | API key for embedding provider (required for hosted APIs). |
| `EMBEDDING_API_BASE_URL` | _(empty)_ | Base URL for OpenAI-compatible embedding endpoints. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Default embedding model name. |
| `EMBEDDING_DEVICE` | `cpu` | HuggingFace embedding runtime device (used when `EMBEDDING_MODEL_PROVIDER=huggingface`). |
| `EMBEDDING_CACHE_FOLDER` | _(empty)_ | Optional HuggingFace cache folder for embedding weights. |
| `EMBEDDING_DIMENSIONS` | _(empty)_ | Required for local HuggingFace embeddings: embedding vector dimension. For OpenAI-compatible APIs you can leave it empty (auto-detected) or set it to override. |
| `OCR_MODEL_PROVIDER` | `openai` | OCR/VLM provider (`openai`, `vllm`, `dots_ocr`). |
| `OCR_API_KEY` | _(empty)_ | API key for OCR provider (required for hosted APIs). |
| `OCR_API_BASE_URL` | _(empty)_ | Base URL for OCR provider. |
| `OPENAI_OCR_MODEL` | `gpt-4o-mini` | OCR/VLM model name. |
| `DOTS_OCR_CACHE_FOLDER` | `./models/dots_ocr` | Local cache for dots_ocr weights. |
| `DOTS_OCR_LOADING_METHOD` | `huggingface` | DotsOCR loading method (`huggingface` for local Transformers, `vllm` for OpenAI-compatible server). |
| `DOTS_OCR_USE_CHINA_MIRROR` | `false` | Enable a HuggingFace mirror when downloading dots_ocr weights. |
| `DOTS_OCR_USE_SNAPSHOT_DOWNLOAD` | `false` | Use HuggingFace `snapshot_download` layout (helps avoid dynamic module issues). |
| `DOTS_OCR_DEVICE` | `cpu` | DotsOCR runtime device (falls back to `DEVICE`). |
| `DOTS_OCR_MODEL_PATH` | `rednote-hilab/dots.ocr` | HuggingFace repo id for the dots_ocr model (when `DOTS_OCR_LOADING_METHOD=huggingface`). |
| `DOTS_OCR_BASE_URL` | `http://localhost:8000/v1` | Base URL for vLLM/OpenAI-compatible dots_ocr server (when `DOTS_OCR_LOADING_METHOD=vllm`). |
| `DOTS_OCR_API_KEY` | _(empty)_ | API key for the vLLM/OpenAI-compatible dots_ocr server (when `DOTS_OCR_LOADING_METHOD=vllm`). |
| `DOTS_OCR_VLLM_MODEL_NAME` | `model` | Model name exposed by the vLLM/OpenAI-compatible server. |
| `DOTS_OCR_MAX_COMPLETION_TOKENS` | `16384` | Max completion tokens for OCR generations. |
| `DOTS_OCR_TEMPERATURE` | `0.1` | OCR generation temperature. |
| `DOTS_OCR_TOP_P` | `1.0` | OCR generation top-p. |
| `USE_CHINA_MIRROR` | `false` | Enable a HuggingFace mirror for local models (embedding/reranker, etc.). |
| `RERANKER_MODEL_NAME` | `Qwen/Qwen3-Reranker-0.6B` | Default local reranker model name (used when `MODEL_PROFILE=local`). |
| `RERANKER_CACHE_FOLDER` | `./models/Qwen` | Cache path for reranker checkpoints. |
| `RERANKER_DEVICE` | `cpu` | Reranker runtime device. |
| `OPENAI_API_KEY` | _(empty)_ | Optional shared key reused across OpenAI-compatible modules when component-specific keys are empty. |
| `OPENAI_BASE_URL` | _(empty)_ | Optional shared base URL reused across OpenAI-compatible modules. |
| `DEVICE` | `cpu` | Optional shared default device used when component-specific device vars are empty. |
| `EMBEDDING_MODEL_NAME` | `Qwen/Qwen3-Embedding-0.6B` | Local HuggingFace embedding model name when `EMBEDDING_MODEL_PROVIDER=huggingface`. |
| `MODEL_PROFILE` | `api` | Chooses config profile (`api` or `local`). Impacts default JSON configs. |
| `MINILM_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Default MiniLM model repo id used by `download_models.py`. |
| `MINILM_CACHE_FOLDER` | `./models/all-MiniLM-L6-v2` | Cache folder used by `download_models.py` when downloading MiniLM. |

## 1.1 Index & Storage Paths

| Variable | Default | Description |
| --- | --- | --- |
| `FILE_STORE_BASE_PATH` | `./data/file_store` | Local blob store base path for original files. |
| `PARSED_CONTENT_STORE_BASE_PATH` | `./data/parsed_content_store` | Parsed content store path. |
| `CHUNK_STORE_BASE_PATH` | `./data/chunk_store` | Chunk store path. |
| `LOCAL_BLOB_STORE_BASE_PATH` | `./data/files` | Legacy alias for `LOCAL_FILE_STORAGE_PATH` (only used when a JSON `base_path` is not provided). |
| `FAISS_INDEX_PATH` | `./data/unified_faiss_index` | Unified FAISS index directory. |
| `BM25_INDEX_PATH` | `./data/unified_bm25_index` | Unified BM25 index directory. |
| `GRAPH_STORAGE_PATH` | `./data/graph_index_neo4j` | Graph index / embedding cache directory (Neo4j HippoRAG). |
| `GRAPH_INDEX_NAME` | `index` | Graph index file name prefix. |

## 2. Evidence Output Controls

| Variable | Default | Description |
| --- | --- | --- |
| `ENABLE_ALL_EVIDENCE` | `false` | When `true`, evidence payloads skip all trimming limits. |
| `CHAT_TOP_CHUNKS` | `5` | Maximum chunks returned in chat evidence. |
| `CHAT_TOP_TRIPLES` | `5` | Maximum graph triples returned in chat evidence. |
| `CHAT_TOP_SEED_ENTITIES` | `5` | Maximum seed entities surfaced in chat evidence. |
| `DEEPSEARCH_TOP_CHUNKS` | `10` | Maximum chunks returned in DeepSearch evidence and displayed in report appendix (first 100 chars preview). |
| `DEEPSEARCH_TOP_TRIPLES` | `30` | Maximum graph triples returned in DeepSearch evidence. |
| `DEEPSEARCH_TOP_SEED_ENTITIES` | `15` | Maximum seed entities surfaced in DeepSearch evidence. |
| `DEEPSEARCH_GRAPH_NODE_LIMIT` | `75` | Cap for DeepSearch graph snapshots (entity + chunk nodes). |
| `DEEPSEARCH_GRAPH_EDGE_LIMIT` | `200` | Cap for DeepSearch edge exports between the retained nodes. |
| `DEEPSEARCH_MAX_REASONING_STEPS` | `32` | Maximum reasoning steps returned in DeepSearch payloads. |
| `DEEPSEARCH_MAX_STAGE_HISTORY` | `10` | Maximum stage history entries returned in DeepSearch payloads. |
| `DEEPSEARCH_MAX_EXTERNAL_CALLS` | `5` | Maximum external call entries returned in DeepSearch payloads. |
| `DEEPSEARCH_MAX_TOOL_METADATA` | `5` | Maximum tool metadata entries returned in DeepSearch payloads. |
| `SEMANTIC_UNIT_MAX_MATCHED_SLICES` | `3` | Maximum slice snippets attached to a semantic-unit anchor (post-retrieval merge). |
| `TABLE_MAX_MERGED_ROWS` | `30` | Maximum table data rows merged into a table anchor after retrieval. |
| `SEMANTIC_UNIT_MAX_MERGED_SLICE_CHARS` | `1200` | Maximum characters appended per matched slice when merging into `anchor.content`. |
| `SEMANTIC_UNIT_MAX_MERGED_TOTAL_CHARS` | `3000` | Total character budget for merged slice snippets appended to an anchor. |

## 2.1 Semantic-Unit Chunking Controls

These knobs apply when the knowledge config selects `semantic_unit_chunker` (for example `config/json_configs/knowledge_semantic_unit.json`).

| Variable | Default | Description |
| --- | --- | --- |
| `SEMANTIC_CHUNKING_LEVEL` | `basic` | Semantic unit chunking level: `disabled`/`basic`/`standard`/`advanced`. |
| `TABLE_SMALL_MAX_TOKENS` | _(empty)_ | Override table small/large threshold (leave empty to use code defaults). |
| `TABLE_SLICE_MAX_TOKENS` | _(empty)_ | Override target token budget for table slices. |
| `TABLE_SLICE_OVERLAP_ROWS` | _(empty)_ | Override overlap rows for table slices. |
| `CODE_SMALL_MAX_TOKENS` | _(empty)_ | Override code small/large threshold. |
| `CODE_SLICE_MAX_TOKENS` | _(empty)_ | 🔶 Reserved: does not emit code slices (fenced code blocks are kept intact); used only for future function/class-level splitting token budgets. |
| `CODE_SLICE_OVERLAP_LINES` | _(empty)_ | 🔶 Reserved: does not emit code slices (fenced code blocks are kept intact); used only for future splitting overlap lines. |
| `LIST_SMALL_MAX_TOKENS` | _(empty)_ | Override list small/large threshold. |
| `LIST_SLICE_MAX_TOKENS` | _(empty)_ | Override target token budget for list slices. |
| `LIST_SLICE_OVERLAP_ITEMS` | _(empty)_ | Override overlap items for list slices. |

## 3. Development / Owner Scope

| Variable | Default | Description |
| --- | --- | --- |
| `DEVELOP_MODE` | `false` | When `true`, Docker services expose ports to `localhost` and the CLI auto-creates a test user. |
| `DEVELOP_OWNER_ID` | `00000000-0000-0000-0000-000000000001` | Default owner UUID used by CLI/tests in develop mode. |
| `DEVELOP_OWNER_USERNAME` | `dev_cli_user` | Username for the auto-created develop-mode user. |
| `DEVELOP_OWNER_PASSWORD` | `dev-cli-password` | Password for the develop-mode user. |
| `ADMIN_OWNER_ID` | `00000000-0000-0000-0000-00000000ABCD` | Optional admin UUID with global graph access. Leave empty to disable. |

## 4. PostgreSQL

| Variable | Default | Description |
| --- | --- | --- |
| `POSTGRES_HOST` | `localhost` | Hostname for Postgres. |
| `POSTGRES_PORT` | `5555` | Postgres port inside Docker. |
| `POSTGRES_USER` | `postgres` | Database username. |
| `POSTGRES_PASSWORD` | `123` | Database password. |
| `POSTGRES_DB` | `rag_arc` | Default database name. |
| `POSTGRES_HOST_PORT` | `5555` | Host port mapped when `EXPOSE_POSTGRES=true`. |
| `EXPOSE_POSTGRES` | `false` | Whether Docker exposes the Postgres port to the host. |

## 5. Redis

| Variable | Default | Description |
| --- | --- | --- |
| `REDIS_HOST` | `localhost` | Redis hostname. |
| `REDIS_PORT` | `6379` | Redis port inside Docker. |
| `REDIS_DB` | `0` | Redis logical DB. |
| `REDIS_PASSWORD` | _(empty)_ | Redis password if enabled. |
| `REDIS_HOST_PORT` | `6379` | Host port exposed when `EXPOSE_REDIS=true`. |
| `EXPOSE_REDIS` | `false` | Whether to expose Redis outside Docker. |

## 5.1 Celery / Long-Task Queue (Celery + Redis)

When `TASK_QUEUE_MODE=celery`, these long-running operations are executed by Celery workers and can scale across processes:
- knowledge file indexing / deletion
- DeepSearch `run_async` (SSE progress supports `last_event_id` replay)
- knowledge export tasks: `/knowledge/graph/export_async`, `/knowledge/mindmap/export_async`

| Variable | Default | Description |
| --- | --- | --- |
| `TASK_QUEUE_MODE` | `inprocess` | Background task mode: `inprocess` (in-API) or `celery` (distributed workers). |
| `CELERY_BROKER_URL` | _(empty)_ | Broker URL (defaults to `redis://REDIS_HOST:REDIS_PORT/REDIS_DB` when empty). |
| `CELERY_RESULT_BACKEND` | _(empty)_ | Result backend (defaults to broker; for long tasks prefer RedisTaskQueue result keys). |
| `CELERY_QUEUE_INDEXING` | `indexing` | Queue name for indexing/deletion tasks. |
| `CELERY_QUEUE_DEEPSEARCH` | `deepsearch` | Queue name for DeepSearch tasks. |
| `CELERY_QUEUE_EXPORT` | _(empty)_ | Queue name for export tasks (graph/mindmap). When empty, falls back to `CELERY_QUEUE_INDEXING`. |
| `CELERY_TASK_IGNORE_RESULT` | `true` | Disable Celery result-backend writes by default (recommended for long tasks). |
| `CELERY_RESULT_EXPIRES_SECONDS` | `3600` | Expiration (seconds) for Celery result backend records. |
| `CELERY_TASK_ACKS_LATE` | `true` | Acknowledge tasks only after completion (requires idempotency/locking). |
| `CELERY_ACKS_ON_FAILURE_OR_TIMEOUT` | `true` | Acknowledge on failure/timeout (used with acks_late). |
| `CELERY_REJECT_ON_WORKER_LOST` | `true` | Re-queue tasks when a worker is lost. |
| `CELERY_WORKER_PREFETCH_MULTIPLIER` | `1` | Prefetch multiplier (long tasks usually want `1`). |
| `CELERY_TASK_SOFT_TIME_LIMIT_SECONDS` | `0` | Soft time limit in seconds (`0` disables). |
| `CELERY_TASK_TIME_LIMIT_SECONDS` | `0` | Hard time limit in seconds (`0` disables). |
| `CELERY_VISIBILITY_TIMEOUT_SECONDS` | `86400` | Redis broker visibility timeout (seconds; must exceed max task runtime). |
| `MQ_NAMESPACE` | `rag-arc:mq` | RedisTaskQueue namespace prefix. |
| `MQ_TASK_RUN_TTL_SECONDS` | `86400` | TTL (seconds) for TaskRun KV records. |
| `MQ_PROGRESS_TTL_SECONDS` | `86400` | TTL (seconds) for per-run progress streams / seq maps. |
| `MQ_RESULT_TTL_SECONDS` | `86400` | TTL (seconds) for result keys. |
| `MQ_STREAM_MAXLEN` | `20000` | Max length for Redis Streams (approximate trimming). |
| `MQ_FAILFAST_ON_REDIS_DOWN` | _(empty)_ | Whether to fail-fast when Redis is unavailable: default is fail-fast in `celery` mode and best-effort in `inprocess` mode. |
| `FILE_OP_LOCK_TTL_SECONDS` | `21600` | Distributed file-operation lock TTL (seconds; shared by index/delete). |
| `CELERY_TASK_MAX_RETRIES` | `3` | Maximum retry attempts for task exceptions. |
| `CELERY_TASK_RETRY_COUNTDOWN_SECONDS` | `5` | Countdown (seconds) before retrying on exceptions. |
| `CELERY_TASK_LOCK_MAX_RETRIES` | `30` | Maximum retry attempts when file lock is busy. |
| `CELERY_TASK_LOCK_RETRY_COUNTDOWN_SECONDS` | `2` | Countdown (seconds) before retrying when file lock is busy. |

### 5.1.1 Running locally / in tests

- Start Celery workers: `bash local/tmp/start_mq_workers.sh` (loads `.env`).
- Stop Celery workers: `bash local/tmp/stop_mq_workers.sh`.
- Optional: archive Redis Streams into Postgres: `uv run python scripts/message_queue_sync.py --daemon` (or `--once`).

## 6. DeepSearch Defaults

Planner/graph defaults. Leave as-is unless customizing behavior.

| Variable | Default | Description |
| --- | --- | --- |
| `DEEPSEARCH_DEFAULT_ADAPTER` | `hipporag` | Graph adapter registered in the registry. |
| `DEEPSEARCH_PLANNER_MODE` | `react` | Planner runtime (`react`, `iter_research`, `parallel_thinking`). |
| `DEEPSEARCH_GRAPH_STRATEGY` | `ppr_chain` | Graph reasoning strategy label. |
| `DEEPSEARCH_PLANNER_MAX_STEPS` | `6` | Max reasoning steps per plan. |
| `DEEPSEARCH_PLANNER_ENABLE_SUBQUESTION` | `true` | Allow planner to spawn sub-questions. |
| `DEEPSEARCH_PLANNER_DISABLE_LLM` | `false` | Force planner to run without LLM (for tests). |
| `DEEPSEARCH_PLANNER_LLM_PROVIDER` | _(empty)_ | Optional: planner-specific LLM provider (leave empty to reuse global chat config). |
| `DEEPSEARCH_PLANNER_MODEL_NAME` | _(empty)_ | Optional: planner-specific model name. |
| `DEEPSEARCH_PLANNER_MAX_TOKENS` | _(empty)_ | Optional: planner-specific max tokens override. |
| `DEEPSEARCH_PLANNER_TEMPERATURE` | _(empty)_ | Optional: planner-specific temperature override. |
| `DEEPSEARCH_PLANNER_API_KEY` | _(empty)_ | Optional: planner-specific API key override. |
| `DEEPSEARCH_PLANNER_BASE_URL` | _(empty)_ | Optional: planner-specific base URL override. |
| `DEEPSEARCH_PLANNER_ORGANIZATION` | _(empty)_ | Optional: planner-specific organization override. |
| `DEEPSEARCH_PLANNER_TIMEOUT` | _(empty)_ | Optional: planner-specific request timeout. |
| `DEEPSEARCH_PLANNER_MAX_RETRIES` | _(empty)_ | Optional: planner-specific retry count. |
| `DEEPSEARCH_PERSIST_PLAN` | `true` | Persist plan JSON to disk. |
| `DEEPSEARCH_PLAN_OUTPUT_DIR` | `./local/deepsearch_runs` | Folder for persisted plans. |
| `DEEPSEARCH_TOOL_ARTIFACT_DIR` | `./local/deepsearch_artifacts` | Output directory for tool telemetry. |
| `DEEPSEARCH_ALLOW_EXTERNAL_CHANNEL` | `false` | Planner-only flag for emitting `web` steps (used when `DEEPSEARCH_EXTERNAL_SEARCH_ENABLED` is not set). |
| `DEEPSEARCH_EXTERNAL_SEARCH_ENABLED` | `false` | Runtime override for external search enablement (config SoT: `external_channel.enabled` + `gap_detection.enable_external_on_gap`). |
| `DEEPSEARCH_TELEMETRY_ENABLED` | `true` | Enable telemetry capture for tool runs (local artifacts). |
| `TAVILY_API_KEY` | _(empty)_ | API key for Tavily (when external search enabled). |
| `DEEPSEARCH_WEB_PROVIDER` | _(empty)_ | External search routing hint (`tavily` / `tool` / `mcp`; unknown values fall back to `tavily`). |
| `DEEPSEARCH_TOOL_HINTS` | _(empty)_ | JSON list to override planner tool hints. |
| `DEEPSEARCH_TOOL_MCP_CONFIG_PATH` | _(empty)_ | Custom JSON config for tool MCP server. |
| `DEEPSEARCH_TOOL_MCP_ADAPTER_CONFIG` | _(empty)_ | JSON file describing adapter overrides. |
| `DEEPSEARCH_TOOL_MCP_ADAPTER_NAME` | _(empty)_ | Adapter name when not using config path. |
| `DEEPSEARCH_TOOL_MCP_ADAPTER_PARAMS` | `{}` | JSON dictionary of adapter kwargs. |
| `DEEPSEARCH_TOOL_MCP_SCOPE_ID` | _(empty)_ | Scope ID used when MCP server runs standalone. |
| `DEEPSEARCH_TOOL_MCP_SCOPE_TYPE` | `owner` | Scope type label. |
| `DEEPSEARCH_TOOL_MCP_SCOPE_LABELS` | `[]` | JSON list of labels for MCP scope. |
| `DEEPSEARCH_TOOL_MCP_SCOPE_ATTRIBUTES` | `{}` | JSON dict of extra scope attributes. |
| `DEEPSEARCH_TOOL_MCP_TOOLS` | _(empty)_ | Optional comma separated tool allowlist. |
| `DEEPSEARCH_ALLOW_SEMANTIC_CHANNEL` | `true` | Allow semantic traversal branch. |
| `DEEPSEARCH_CHAIN_DEPTH` | `4` | Graph traversal depth. |
| `DEEPSEARCH_ENABLE_FINANCE_HOOKS` | `false` | Enable finance-specific heuristics. |
| `DEEPSEARCH_MCP_SERVER_URI` | _(empty)_ | Remote MCP URI for DeepSearch (disable by default). |
| `DEEPSEARCH_MCP_API_KEY` | _(empty)_ | API key for remote MCP. |
| `DEEPSEARCH_MCP_TRANSPORT` | `auto` | Transport for MCP clients (`auto`/`sse`/`stdio`). |
| `DEEPSEARCH_MCP_STDIO_COMMAND` | _(empty)_ | Command to spawn local MCP server when using stdio. |
| `DEEPSEARCH_MCP_STDIO_ENV` | `{}` | JSON environment overrides for stdio transport. |
| `DEEPSEARCH_MCP_HEADERS` | `{}` | Extra HTTP headers when using SSE/HTTP transport. |
| `DEEPSEARCH_MCP_TIMEOUT` | `30` | HTTP connect timeout. |
| `DEEPSEARCH_MCP_READ_TIMEOUT` | `300` | HTTP read timeout. |
| `DEEPSEARCH_MCP_PERSISTENT_SESSION` | `true` | Reuse MCP HTTP sessions. |
| `DEEPSEARCH_MCP_ENABLE_GRAPH_CONTEXT` | `true` | Attach graph context to MCP requests. |
| `DEEPSEARCH_MCP_GRAPH_CONTEXT_FIELD` | `__graph_context__` | Field name for graph context injection. |
| `DEEPSEARCH_GAP_COVERAGE_THRESHOLD` | `0.7` | Coverage threshold for gap detection. |
| `DEEPSEARCH_GAP_CONFIDENCE_THRESHOLD` | `0.6` | Confidence threshold for gap detection. |
| `DEEPSEARCH_GAP_EXPECTED_MIN_CHUNKS` | `3` | Minimum expected chunk count before triggering external search. |
| `DEEPSEARCH_CONSISTENCY_CHECK` | `true` | Enable LLM-based consistency check to validate report claims against evidence. |
| `DEEPSEARCH_PARALLEL_SECTIONS` | `false` | Generate report sections in parallel (faster but uses more API calls). |
| `DEEPSEARCH_QUALITY_LOOP_ENABLED` | `false` | Enable iterative quality gating (research → verify → iterate). |
| `DEEPSEARCH_QUALITY_LOOP_MAX_ROUNDS` | `2` | Maximum rounds (initial + follow-ups) for the quality loop. |
| `DEEPSEARCH_QUALITY_LOOP_MIN_CITATION_SENTENCE_COVERAGE` | `0.6` | Minimum fraction of report sentences that must include at least one valid citation. |
| `DEEPSEARCH_QUALITY_LOOP_REQUIRE_CONSISTENCY` | `true` | Fail the quality gate when consistency checking reports issues. |
| `DEEPSEARCH_QUALITY_LOOP_MAX_UNCITED_SENTENCES` | `6` | Maximum uncited sentences surfaced as repair targets (used to drive follow-up retrieval/rewrite). |
| `DEEPSEARCH_QUALITY_LOOP_MAX_ACTIONS` | `6` | Maximum follow-up actions produced by the quality gate. |
| `DEEPSEARCH_QUALITY_LOOP_ENABLE_LLM_JUDGE` | `true` | Enable the rubric-based LLM judge (called only when deterministic checks fail or gaps exist). |
| `DEEPSEARCH_QUALITY_LOOP_JUDGE_TEMPERATURE` | `0.0` | Temperature for the quality judge. |
| `DEEPSEARCH_QUALITY_LOOP_JUDGE_MAX_RETRIES` | `1` | Retry attempts for the quality judge call. |
| `DEEPSEARCH_QUALITY_LOOP_TRIGGER_EXTERNAL_ON_FAILURE` | `true` | Allow the quality gate to request external search actions (still requires external search to be enabled). |

### Example: enabling MCP routing for remote tools

```bash
# Route DeepSearch tools through a remote MCP server (when certain tools are marked mcp_only/mcp_fallback)
DEEPSEARCH_MCP_SERVER_URI="http://127.0.0.1:8765/mcp/tools"
DEEPSEARCH_MCP_TRANSPORT="sse"
DEEPSEARCH_MCP_HEADERS='{"Authorization": "Bearer your-mcp-token"}'
# Constrain which tools the MCP server exposes (optional)
DEEPSEARCH_TOOL_MCP_TOOLS="graph.context_rollup,graph.think"
# Provide a default graph scope for the standalone MCP server
DEEPSEARCH_TOOL_MCP_SCOPE_ID="00000000-0000-0000-0000-000000000001"
DEEPSEARCH_TOOL_MCP_SCOPE_TYPE="owner"
DEEPSEARCH_TOOL_MCP_SCOPE_LABELS='["demo", "shared"]'
```

## 7. Application Settings

| Variable | Default | Description |
| --- | --- | --- |
| `JWT_SECRET_KEY` | `your-secret-key-change-this-in-production` | Secret used to sign JWT tokens. Replace in production. |
| `HF_TOKEN` | _(empty)_ | HuggingFace token for downloading gated models (optional). |
| `HF_ENDPOINT` | _(empty)_ | Optional HuggingFace endpoint override (e.g. `https://hf-mirror.com`). |
| `LOG_LEVEL` | `INFO` | Python logging level (`DEBUG`, `INFO`, etc.). |

## 8. File Storage & Parser Paths

| Variable | Default | Description |
| --- | --- | --- |
| `PARSER_OUTPUT_DIR` | `./data/parsed_files` | Unified base directory for parser outputs (native/dots_ocr/vlm_ocr subfolders). |
| `NATIVE_PARSER_OUTPUT_DIR` | _(empty)_ | Optional override for native parser output directory. |
| `DOTSOCR_OUTPUT_DIR` | _(empty)_ | Optional override for dots_ocr output directory. |
| `VLMOCR_OUTPUT_DIR` | _(empty)_ | Optional override for VLM OCR output directory. |
| `OCR_MODEL_NAME` | _(empty)_ | Optional backward-compatible OCR model name alias. |
| `RAGARC_RUNTIME_DIR` | `./local/runtime` | Fallback runtime root when preferred local directories are not writable. |
| `LOCAL_FILE_STORAGE_PATH` | `./data/files` | Default root for `local_blob_store` when JSON `base_path` is not provided. |

## 9. Neo4j Graph Database

| Variable | Default | Description |
| --- | --- | --- |
| `NEO4J_URL` | `bolt://localhost:7687` | Connection string for Neo4j. |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username. |
| `NEO4J_PASSWORD` | `12345678` | Neo4j password. |
| `NEO4J_DATABASE` | `neo4j` | Database name/alias. |
| `EXPOSE_NEO4J` | `false` | Whether to expose Neo4j browser/bolt port. |
| `NEO4J_HTTP_PORT` | `7474` | Host HTTP port when `EXPOSE_NEO4J=true`. |
| `NEO4J_BOLT_PORT` | `7687` | Host bolt port when `EXPOSE_NEO4J=true`. |

## 10. Optional MinIO Object Storage

| Variable | Default | Description |
| --- | --- | --- |
| `MINIO_USERNAME` | `ROOTNAME` | MinIO access key / username (used only when MinIO integration is enabled). |
| `MINIO_PASSWORD` | `CHANGEME123` | MinIO secret key / password. |

The `.env.example` also includes commented placeholders for:
- `MINIO_ENDPOINT`
- `MINIO_BUCKET`
- `MINIO_SECURE`

Uncomment and configure them only when integrating object storage for parsed files.

## 11. Build / Advanced Runtime

| Variable | Default | Description |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | _(empty)_ | Optional: constrain GPU visibility when running local models. |
| `PYTHONPATH` | _(empty)_ | Optional: extra Python import roots for subprocesses (e.g. vLLM launcher). |
| `UV_INSTALL_URL` | `https://astral.sh/uv/install.sh` | Optional: override `uv` installer URL used by `build.sh`. |
| `UV_INDEX_URL` | `https://pypi.org/simple` | Optional: override Python package index used by `build.sh`. |
| `PYTORCH_INDEX_URL` | _(empty)_ | Optional: override PyTorch wheel index (primarily for GPU builds). |

## 12. CLI Defaults

| Variable | Default | Description |
| --- | --- | --- |
| `CLI_OWNER_ID` | _(empty)_ | Optional: pinned owner id used by CLI commands when present. |
| `CLI_OWNER_ID_FILE` | _(empty)_ | Optional: where to persist generated CLI owner id (default: `~/.rag_arc_owner_id`). |
| `DEFAULT_OWNER_ID` | _(empty)_ | Optional legacy alias checked by CLI owner resolution. |
| `RAG_ARC_OWNER_ID` | _(empty)_ | Optional legacy alias checked by CLI owner resolution. |

## 13. Quick Start / Test Hooks

| Variable | Default | Description |
| --- | --- | --- |
| `QUICK_START_OWNER_ID` | _(empty)_ | Optional: owner id used by quick-start scripts/examples. |
| `RAG_OUTPUT_DIR` | _(empty)_ | Optional: output directory for RAG pipeline artifacts. |
| `DEEPSEARCH_EXPERIMENT_OUTPUT_DIR` | _(empty)_ | Optional: output directory for DeepSearch experiment artifacts. |
| `DEEPSEARCH_CITATION_ALIASES` | _(empty)_ | Optional: JSON mapping for citation aliases. |
| `DEEPSEARCH_TOOL_AUDIT_LABEL` | _(empty)_ | Optional: label attached to tool audit records. |
| `DEEPSEARCH_TOOL_MCP_AUDIT_LABEL` | _(empty)_ | Optional: label attached to MCP tool audit records. |
| `DEEPSEARCH_TOOL_MCP_INSTRUCTIONS` | _(empty)_ | Optional: extra planner instructions for MCP tool usage. |
| `DEEPSEARCH_TOOL_MCP_SCOPE_OVERRIDE_POLICY` | _(empty)_ | Optional: policy controlling when MCP scope is overridden. |
| `DEEPSEARCH_TOOL_MCP_SCOPE_OVERRIDE_TOKEN` | _(empty)_ | Optional: token used to authorize MCP scope overrides. |
| `DEEPSEARCH_RUN_LLM_INTEGRATION_TESTS` | `0` | Optional: run DeepSearch LLM integration tests when set to `1`. |

### Optional local-model smoke tests (pytest)

| Variable | Default | Description |
| --- | --- | --- |
| `RUN_RAGARC_GPT2_CHAT_TESTS` | `0` | Opt-in: run local chat smoke tests using `models/gpt2` when set to `1`. |
| `RAGARC_GPT2_MODEL_DIR` | `./models/gpt2` | Path to local tiny-gpt2 directory for chat tests. |
| `RUN_RAGARC_LOCAL_EMBEDDING_TESTS` | `0` | Opt-in: run local embedding smoke tests when set to `1`. |
| `RAGARC_ST_MODEL_SNAPSHOTS` | `./models/all-MiniLM-L6-v2/models--sentence-transformers--all-MiniLM-L6-v2/snapshots` | Path to local SentenceTransformer snapshots. |
| `RUN_RAGARC_LOCAL_RERANK_TESTS` | `0` | Opt-in: run local reranker smoke tests when set to `1`. |
| `RAGARC_RERANKER_SNAPSHOTS` | `./models/Qwen/models--Qwen--Qwen3-Reranker-0.6B/snapshots` | Path to local reranker snapshots. |
| `RAGARC_ALLOW_LARGE_MODELS` | `0` | Safety gate: must be `1` for large local model tests. |

## 14. Azure OpenAI (Optional)

| Variable | Default | Description |
| --- | --- | --- |
| `API_VERSION` | _(empty)_ | Azure OpenAI API version (when using Azure provider). |
| `AZURE_OPENAI_API_KEY` | _(empty)_ | Azure OpenAI API key. |

## 15. Test-only placeholders (env substitution)

These are used by internal env-substitution tests and can be left empty.

| Variable | Default | Description |
| --- | --- | --- |
| `APP_NAME` | _(empty)_ | Test placeholder. |
| `APP_VALUE` | _(empty)_ | Test placeholder. |
| `BASE_URL` | _(empty)_ | Test placeholder. |
| `EXISTING_VAR` | _(empty)_ | Test placeholder. |
| `LIST_VAR` | _(empty)_ | Test placeholder. |
| `MIXED_VAR` | _(empty)_ | Test placeholder. |
| `NESTED_VAR` | _(empty)_ | Test placeholder. |
| `STRING_VAR` | _(empty)_ | Test placeholder. |
| `TEST_VAR` | _(empty)_ | Test placeholder. |
| `VAR1` | _(empty)_ | Test placeholder. |
| `VAR2` | _(empty)_ | Test placeholder. |

## 16. Integration / Test Flags

Set to `1` (or any non-empty value) to opt-in when the required services/models are available; leave empty to skip.

| Variable | Default | Description |
| --- | --- | --- |
| `RUN_RAGARC_INTEGRATION_TESTS` | _(empty)_ | Run integration test suites. |
| `RUN_RAGARC_POSTGRES_TESTS` | _(empty)_ | Run Postgres-dependent test suites. |
| `RUN_RAGARC_CHAT_STORAGE_TESTS` | _(empty)_ | Run chat-storage test suites. |
| `RUN_RAGARC_VECTOR_TESTS` | _(empty)_ | Run vector-store test suites. |
| `RUN_RAGARC_MQ_STRESS_TESTS` | _(empty)_ | Optional: run real-Redis message-queue stress smoke tests (`test/stress/test_mq_stress_real_redis.py`) when set to `1`. |
| `RAGARC_E2E_TOKEN` | _(empty)_ | Token used by `test/test_complete_e2e_api.py` to authenticate API requests. |

---

**Tip:** Copy `.env.example` to `.env`, fill in the API keys for the LLM provider you use, and everything else should function with Docker defaults. Adjust the remaining variables only when you need custom deployments or scopes.
