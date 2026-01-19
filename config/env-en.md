# Environment variables (minimal `.env`) + full reference

RAG-ARC uses two configuration layers:

- **`.env`**: secrets + feature switches (recommended for most users).
- **`config/`**: all tunable parameters (retrieval/chunking/top_k/thresholds/budgets/paths/etc.).

The runtime still supports many environment variables as **override knobs** for compatibility.
This document explains **all** supported env variables and what each one means.

Recommended workflow:
1) Keep `.env` small (only secrets / enablement).
2) Tune behavior via `config/` (JSON + Python).

---

## 0. Minimal `.env` (recommended)

Start from:

```bash
cp .env.example .env
```

Required secrets (no defaults):
- `OPENAI_API_KEY` / `OPENAI_BASE_URL` (or component-specific `CHAT_*` / `EMBEDDING_*` / `OCR_*`)
- `JWT_SECRET_KEY` (generate with `openssl rand -hex 32`)

Optional feature switches (have defaults):
- `bench_mode`, `TASK_QUEUE_MODE`, `MODEL_PROFILE`, `DEVELOP_MODE`, `ADMIN_OWNER_ID`

Benchmark/experiment mode:
- `bench_mode` (default `0`): when set to `1`, benchmark runners (see `application/rag_inference/module_bench.py`, `application/rag_inference/deepsearch/service_bench.py`) execute algorithm-only flows and return plain-text answers (no citations/reports/external web steps).

Optional external web search (only if enabled in config):
- `TAVILY_API_KEY`

Optional infrastructure overrides (defaults work for local/Docker; set only when needed):
- PostgreSQL: `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB`
- Redis: `REDIS_HOST`, `REDIS_PORT`, `REDIS_DB`, `REDIS_PASSWORD`
- Neo4j: `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`

---

## 0.1 Advanced configuration (edit `config/`, not `.env`)

Recommended places to tune parameters:

- `config/json_configs/rag_inference*.json`: chat pipeline, retrievers, rerankers, model selection.
- `config/json_configs/knowledge*.json`: parsing, chunking, indexing/graph build.
- `config/json_configs/deepsearch_service.json`: DeepSearch planner/tools/report/quality gates.
- `config/output_limits.py`: response trimming / evidence caps.
- `config/core/deepsearch/*_defaults.py`: DeepSearch loop/tool/report defaults used at runtime.

How secrets flow into configs:
- JSON supports `${ENV_VAR}` placeholders (e.g. `${OPENAI_API_KEY}`), so keep secrets in `.env` and reference them from `config/`.

## 1. Model & LLM Providers

| Variable | Default | Description |
| --- | --- | --- |
| `CHAT_MODEL_PROVIDER` | `openai` | Chat provider (`openai` = OpenAI-compatible API, `huggingface` = local Transformers). |
| `CHAT_MODEL_NAME` | _(empty)_ | Optional preferred chat model name. When `CHAT_MODEL_PROVIDER=huggingface`, this can be a HuggingFace repo id or a local filesystem path. |
| `CHAT_MODEL_DEVICE` | `cpu` | HuggingFace chat runtime device (used when `CHAT_MODEL_PROVIDER=huggingface`). |
| `CHAT_MODEL_CACHE_FOLDER` | _(empty)_ | Optional HuggingFace cache folder for chat weights/tokenizers. |
| `CHAT_API_KEY` | _(empty)_ | **Required** (when `CHAT_MODEL_PROVIDER=openai`): API key for chat provider. |
| `CHAT_API_BASE_URL` | _(empty)_ | **Required** (when `CHAT_MODEL_PROVIDER=openai`): Base URL for OpenAI-compatible chat endpoints (e.g. `https://api.openai.com/v1`). |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Legacy/default chat model name used when `CHAT_MODEL_NAME` is empty. |
| `LOW_COST_MODEL` | _(empty)_ | Optional: cheaper model used for exploration-heavy calls (planning/reflection/quality checks). When empty, the system reuses the main chat model. |
| `OPENAI_API_BASE` | _(empty)_ | Optional legacy alias for OpenAI-compatible base URL. |
| `EMBEDDING_MODEL_PROVIDER` | `openai` | Embedding provider (`openai` = OpenAI-compatible API, `huggingface` = local SentenceTransformers). |
| `EMBEDDING_API_KEY` | _(empty)_ | **Required** (when `EMBEDDING_MODEL_PROVIDER=openai`): API key for embedding provider. |
| `EMBEDDING_API_BASE_URL` | _(empty)_ | **Required** (when `EMBEDDING_MODEL_PROVIDER=openai`): Base URL for OpenAI-compatible embedding endpoints. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model name. When set, it takes precedence over `EMBEDDING_MODEL_NAME` for `EMBEDDING_MODEL_PROVIDER=openai`. |
| `EMBEDDING_DEVICE` | `cpu` | HuggingFace embedding runtime device (used when `EMBEDDING_MODEL_PROVIDER=huggingface`). |
| `EMBEDDING_CACHE_FOLDER` | _(empty)_ | Optional HuggingFace cache folder for embedding weights. |
| `EMBEDDING_DIMENSIONS` | _(empty)_ | Optional override for embedding vector dimension. When empty, the system can auto-detect the dimension (and will cache it). |
| `EMBEDDING_TIMEOUT_SECONDS` | `20` | Embedding request timeout (seconds). Lower this to fail-fast when the embedding endpoint is flaky. |
| `EMBEDDING_MAX_RETRIES` | `0` | Embedding request retries (OpenAI SDK retries). Prefer `0` when the upstream gateway enforces long rate-limits; use `EMBEDDING_RATE_LIMIT_*` backoff instead. |
| `EMBEDDING_RATE_LIMIT_MAX_RETRIES` | `6` | Extra retries when embeddings hit HTTP 429 rate limits. |
| `EMBEDDING_RATE_LIMIT_DEFAULT_SLEEP_SECONDS` | `60` | Default sleep (seconds) between 429 retries when Retry-After is not available. |
| `EMBEDDING_RATE_LIMIT_MAX_SLEEP_SECONDS` | `60` | Max sleep (seconds) between 429 retries (cap). |
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
| `OPENAI_API_KEY` | _(empty)_ | Shared fallback key (used when component-specific keys are empty). **Required** when any OpenAI-compatible module runs with its `*_API_KEY` unset. |
| `OPENAI_BASE_URL` | _(empty)_ | Shared fallback base URL (used when component-specific base URLs are empty). **Required** when any OpenAI-compatible module runs with its `*_API_BASE_URL` unset. |
| `DEVICE` | `cpu` | Optional shared default device used when component-specific device vars are empty. |
| `EMBEDDING_MODEL_NAME` | `Qwen/Qwen3-Embedding-0.6B` | Embedding model name (primarily for `EMBEDDING_MODEL_PROVIDER=huggingface`). For `openai`, this is a fallback only when `OPENAI_EMBEDDING_MODEL` is empty. |
| `MODEL_PROFILE` | `api` | Chooses config profile (`api` or `local`). Impacts default JSON configs. |
| `MINILM_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Default MiniLM model repo id used by `download_models.py`. |
| `MINILM_CACHE_FOLDER` | `./models/all-MiniLM-L6-v2` | Cache folder used by `download_models.py` when downloading MiniLM. |

## 1.2 DeepSearch: file-scope cross-language rewrite (advanced)

These vars control when DeepSearch attempts a single query rewrite (via the retriever LLM) to bridge file-name language mismatch under file-scope filtering.

| Variable | Default | Description |
| --- | --- | --- |
| `DEEPSEARCH_FILE_SCOPE_XLANG_RETRY` | `1` | Enable/disable the xlang rewrite attempt (`0/false/no/off` disables). |
| `DEEPSEARCH_FILE_SCOPE_XLANG_ALPHA_RATIO_TO_ZH_MIN` | `0.25` | If ASCII-letter ratio ≥ this AND CJK ratio < `..._CJK_RATIO_TO_ZH_MAX`, treat query as English-like and request Chinese rewrite additions. |
| `DEEPSEARCH_FILE_SCOPE_XLANG_CJK_RATIO_TO_ZH_MAX` | `0.05` | See above; max allowed CJK ratio for the en→zh rewrite trigger. |
| `DEEPSEARCH_FILE_SCOPE_XLANG_CJK_RATIO_TO_EN_MIN` | `0.15` | If CJK ratio ≥ this AND ASCII-letter ratio < `..._ALPHA_RATIO_TO_EN_MAX`, treat query as Chinese-like and request English rewrite additions. |
| `DEEPSEARCH_FILE_SCOPE_XLANG_ALPHA_RATIO_TO_EN_MAX` | `0.08` | See above; max allowed ASCII-letter ratio for the zh→en rewrite trigger. |

## 1.1 Index & Storage Paths

| Variable | Default | Description |
| --- | --- | --- |
| `FILE_STORE_BASE_PATH` | `./data/file_store` | Local blob store base path for original files (relative paths are resolved against the repo root). |
| `PARSED_CONTENT_STORE_BASE_PATH` | `./data/parsed_content_store` | Parsed content store path (relative paths are resolved against the repo root). |
| `CHUNK_STORE_BASE_PATH` | `./data/chunk_store` | Chunk store path (relative paths are resolved against the repo root). |
| `LOCAL_BLOB_STORE_BASE_PATH` | `./data/files` | Legacy alias for `LOCAL_FILE_STORAGE_PATH` (only used when a JSON `base_path` is not provided). |
| `FAISS_INDEX_PATH` | `./data/unified_faiss_index` | Unified FAISS index directory. |
| `BM25_INDEX_PATH` | `./data/unified_bm25_index` | Unified BM25 index directory. |
| `GRAPH_STORAGE_PATH` | `./data/graph_index_neo4j` | Graph index / embedding cache directory (Neo4j HippoRAG). |
| `GRAPH_INDEX_NAME` | `index` | Graph index file name prefix. |
| `KG_SCHEMA_PATH` | `./kg_schema.yml` | KG schema YAML path for Neo4j HippoRAG (predicate governance + direction-sensitive set). Optional: `./fin_kg_schema.yml` for finance/insurance deployments. |
| `GRAPH_INDEX_EMBED_FAILURE_POLICY` | `zero` | Graph-index embedding failure policy: `zero` (fill a small number of failed items with zero vectors and log) / `raise` (fail the indexing task on any failure). |
| `GRAPH_INDEX_EMBED_MAX_FAILURE_RATIO` | `0.05` | Even with `..._POLICY=zero`, fail-fast when failure ratio ≥ this threshold (avoids silently building an all-zero bad index). |
| `GRAPH_INDEX_EMBED_MAX_FAILURE_COUNT` | `50` | Even with `..._POLICY=zero`, fail-fast when failure count ≥ this threshold. |
| `INDEX_TEXT_TITLE_PREFIX_ENABLED` | `true` | Whether to inject a filename-derived `title=` prefix into each chunk’s `metadata.index_text` (improves recall when doc/product name mostly lives in the filename). |
| `INDEX_TEXT_TITLE_MAX_CHARS` | `160` | Max character length of the injected `title=` prefix (keeps index/prompt noise bounded). |

Notes:
- **FAISS fingerprint guard**: the FAISS `.pkl` metadata stores an `embedding_fingerprint` (provider/model/dim). If you switch embedding models/dimensions, set a new `FAISS_INDEX_PATH` (recommended) or rebuild the index; otherwise the system will fail-fast to avoid silent corruption.
- **Path consistency (important)**: indexing and online retrieval must use the same `GRAPH_STORAGE_PATH` / `FAISS_INDEX_PATH` / `BM25_INDEX_PATH`. If you index into directory A but serve from directory B, you may see “the target file/chunks exist in Neo4j + chunk_store, but retrieval hits unrelated files”.
- **E2E isolation**: for real-service tests, point the path knobs above to an isolated directory (for example under `./local/e2e_*`) to avoid polluting `./data/*`.
- **KG domain fallback**: when chunks do not provide `chunk.domain` (or `chunk.metadata["domain"]`), Neo4j indexing falls back to the loaded schema's `default_domain` (for example `finance_insurance` when using `./fin_kg_schema.yml`).
- **HippoRAG PPR directionality (important)**: for general-purpose retrieval stability, `pruned_hipporag_neo4j_retrieval.ppr_directed_mode` defaults to `off` (undirected PPR). `direction_sensitive_relations` in `KG_SCHEMA_PATH` is still used by DeepSearch / fast graph tools for directional constraints and validation; to enable directed PPR at retrieval time, explicitly set `ppr_directed_mode=auto/on` in `config/json_configs/rag_inference*.json` or `config/json_configs/deepsearch_service.json` under `retriever_config`.

## 2. Evidence Output Controls

| Variable | Default | Description |
| --- | --- | --- |
| `ENABLE_ALL_EVIDENCE` | `false` | When `true`, evidence payloads skip all trimming limits. |
| `CHAT_TOP_CHUNKS` | `5` | Maximum chunks returned in chat evidence. |
| `CHATBOT_LLM_TOP_SOURCES` | `10` | Max Sources passed to the chatbot LLM (can be higher than UI `CHATBOT_TOP_SOURCES` to improve coverage for broad queries like "features/benefits"). |
| `USER_TYPE` | `0` | Prompt style/domain selector used by `rag_inference` prompt layering (`config/prompts/rag_inference_prompts.yaml`). |
| `RAG_INFERENCE_PROMPTS_YAML_PATH` | (empty) | Optional override path for `config/prompts/rag_inference_prompts.yaml`. When empty, uses the repo default. |
| `CHAT_TOP_TRIPLES` | `5` | Maximum graph triples returned in chat evidence. |
| `CHAT_TOP_SEED_ENTITIES` | `5` | Maximum seed entities surfaced in chat evidence. |
| `CHAT_MAX_IMAGE_INPUTS` | `4` | Maximum local images attached to a single chat request when the model supports multimodal inputs (MinerU image assets). Set to `0` (or any negative value) to remove the limit. |
| `RAG_RETRIEVAL_OBSERVABILITY` | `false` | When `true`, emit retrieval observability logs/progress (per-retriever file distribution, fused distribution, rerank distribution) to debug "wrong file recalled" and follow-up drift. |
| `RAG_RETRIEVAL_LOG_TOP_FILES` | `10` | Max file ids shown in retrieval distribution logs (counted by file_id). |
| `RAG_RETRIEVAL_LOG_TOP_CHUNKS` | `5` | Max chunk previews shown in retrieval observability logs. |
| `QUERY_VARIANTS_ENABLED` | `true` | Enable retrieval-time query variants (MultiPath generates variants per retriever and unions results). Improves recall on mixed-script/variant corpora. |
| `QUERY_VARIANTS_LANGS` | `zh-Hans,en,zh-Hant` | Comma-separated variant targets, in order. `zh-Hans/zh-Hant` use OpenCC conversion (when installed). `en` is best-effort ASCII token extraction (no translation). |
| `QUERY_VARIANTS_ZH_HANS_HANT_ENABLED` | `true` | Enable Simplified/Traditional Chinese variants (requires OpenCC). |
| `QUERY_VARIANTS_MAX` | `3` | Max number of query variants (including the original). |
| `RAG_RETRIEVAL_WEIGHT_DENSE` | `1.0` | MultiPath RRF fusion weight for the dense retriever. |
| `RAG_RETRIEVAL_WEIGHT_BM25` | `1.0` | MultiPath RRF fusion weight for the BM25 retriever. |
| `RAG_RETRIEVAL_WEIGHT_GRAPH` | `1.5` | MultiPath RRF fusion weight for the graph retriever (lower than before to avoid drowning out dense/bm25 on sparse detail queries). |
| `RAG_RETRIEVAL_DYNAMIC_QUOTA_ENABLED` | `true` | Enable LLM-driven per-query routing ratios that allocate MultiPath candidate quotas across retrievers (coverage floor). Falls back to static ratios when disabled. |
| `RAG_INTENT_ROUTING_ENABLED` | `false` | Enable intent classification + intent-aware query rewrite (single LLM call returns JSON: intent/anchors/rewritten_query). Helps multi-turn chats avoid unnecessary retrieval and drift. |
| `RAG_REWRITE_HISTORY_USER_ONLY` | `true` | Feed only USER turns into the rewrite context (exclude assistant) to reduce assistant-poisoning effects. |
| `RAG_REWRITE_HISTORY_MOST_RECENT_FIRST` | `true` |Order rewrite history context as most-recent-first (aligned with rewrite prompt wording). |
| `RAG_EVIDENCE_CONSISTENCY_ENABLED` | `false` | Enable evidence consistency filtering: use rewrite-produced anchors to keep retrieval evidence within the same company/product file set (reduces cross-product mixing). |
| `RAG_EVIDENCE_MIN_KEEP` | `5` | Minimum number of chunks to keep after evidence consistency filtering; if not met, the filter is skipped and diagnostics are emitted. |
| `DEEPSEARCH_TOP_CHUNKS` | `10` | Maximum chunks returned in DeepSearch evidence and displayed in report appendix (first 100 chars preview). |
| `DEEPSEARCH_TOP_TRIPLES` | `30` | Maximum graph triples returned in DeepSearch evidence. |
| `DEEPSEARCH_TOP_SEED_ENTITIES` | `15` | Maximum seed entities surfaced in DeepSearch evidence. |
| `DEEPSEARCH_MAX_IMAGE_INPUTS` | `6` | Maximum local images attached to DeepSearch report generation when the model supports multimodal inputs (MinerU image assets). Set to `0` (or any negative value) to remove the limit. |
| `DEEPSEARCH_GRAPH_NODE_LIMIT` | `75` | Cap for DeepSearch graph snapshots (entity + chunk nodes). |
| `DEEPSEARCH_GRAPH_EDGE_LIMIT` | `200` | Cap for DeepSearch edge exports between the retained nodes. |
| `DEEPSEARCH_MAX_REASONING_STEPS` | `32` | Maximum reasoning steps returned in DeepSearch payloads. |
| `DEEPSEARCH_MAX_STAGE_HISTORY` | `10` | Maximum stage history entries returned in DeepSearch payloads. |
| `DEEPSEARCH_MAX_EXTERNAL_CALLS` | `5` | Maximum external call entries returned in DeepSearch payloads. |
| `DEEPSEARCH_MAX_TOOL_METADATA` | `5` | Maximum tool metadata entries returned in DeepSearch payloads. |
| `DEEPSEARCH_WEAVER_EVIDENCE_PREVIEW_CHARS` | `180` | Evidence preview character limit in the DeepSearch Weaver trace rendering. |
| `DEEPSEARCH_WEAVER_EVIDENCE_SAMPLE_COUNT` | `3` | Number of evidence samples included in the DeepSearch Weaver trace rendering. |
| `DEEPSEARCH_SOURCE_MAX_CHARS` | `1600` | Max characters included in each DeepSearch `report.sources[*].description` (HippoRAG-compatible sources payload). |
| `DEEPSEARCH_SOURCE_TITLE_MAX_CHARS` | `80` | Max characters included in each DeepSearch `report.sources[*].title`. |
| `DEEPSEARCH_GRAPH_EXPORT_MAX_EDGES` | `2000` | Hard cap on exported edges for DeepSearch subgraph visualization (Neo4j exporter). |
| `KNOWLEDGE_GRAPH_EXPORT_MAX_NODES` | `1000` | Upper bound for `/knowledge/graph/export*` max_nodes to prevent expensive graph exports. |
| `KNOWLEDGE_GRAPH_EXPORT_MAX_EDGES` | `5000` | Upper bound for `/knowledge/graph/export*` max_edges to prevent expensive graph exports. |
| `KNOWLEDGE_MINDMAP_EXPORT_MAX_CHUNKS` | `60` | Maximum chunks sampled when exporting a file-level mindmap (prevents oversize LLM prompts). |
| `KNOWLEDGE_MINDMAP_EXPORT_SEGMENT_SNIPPET_CHARS` | `600` | Per-chunk content snippet size included in the mindmap export merge prompt. |
| `GRAPH_EXPORT_CHUNK_CONTENT_PREVIEW_CHARS` | `240` | Maximum chunk content characters included in graph export payloads (visualization preview). |
| `GRAPH_EXPORT_EDGE_FETCH_FACTOR` | `10` | Multiplier used by exporters to cap how many edges are fetched before sampling (fetch_limit = max_edges * factor). |
| `GRAPH_EXPORT_EDGE_FETCH_MAX` | `50000` | Absolute cap for exporter edge fetch limits (prevents oversized Neo4j edge queries). |
| `GRAPH_EXPORT_FILTER_NUMERIC_TIME_ENTITIES` | `true` | Whether graph exporters filter numeric/date/time-like entity nodes (visualization noise reduction). Set `false` for finance/insurance workloads where those nodes are important. |
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
| `ORG_ADMIN_OWNER_IDS` | _(empty)_ | Org admin owner id allowlist (comma-separated UUIDs). Used to authorize CRUD operations on the enterprise shared library (`SHARE_OWNER_ID`). |
| `SHARE_OWNER_ID` | _(empty)_ | Owner id (UUID) for the enterprise shared knowledge domain (share). Used by algorithm-level visibility scopes (`me` / `me+share`). Must differ from `ADMIN_OWNER_ID`; leave empty to disable share-merged retrieval. |
| `CHATBOT_SHARED_DOCUMENT_OWNER_ID` | `00000000-0000-0000-0000-000000000001` | Shared owner UUID for chatKB (`type=1`) unified retrieval; chatbot and rag_inference default to this owner when `type=1`. |

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
| `TASK_QUEUE_MODE` | `celery` | Background task mode: `inprocess` (in-API) or `celery` (distributed workers). |
| `CELERY_BROKER_URL` | _(empty)_ | Broker URL (defaults to `redis://REDIS_HOST:REDIS_PORT/REDIS_DB` when empty). |
| `CELERY_RESULT_BACKEND` | _(empty)_ | Result backend (defaults to broker; for long tasks prefer RedisTaskQueue result keys). |
| `CELERY_QUEUE_INDEXING` | `indexing` | Queue name for indexing/deletion tasks. |
| `CELERY_QUEUE_DEEPSEARCH` | `deepsearch` | Queue name for DeepSearch tasks. |
| `CELERY_QUEUE_EXPORT` | `export` | Queue name for export tasks (graph/mindmap). When empty, falls back to `CELERY_QUEUE_INDEXING`. |
| `CELERY_LOGLEVEL` | `info` | Worker log level (used by `./start.sh` when auto-starting workers). |
| `CELERY_WORKER_CONCURRENCY` | `2` | Worker concurrency (processes/threads; used by `./start.sh` when auto-starting workers). |
| `CELERY_WORKER_POOL` | `prefork` | Worker pool implementation (used by `./start.sh` when auto-starting workers). |
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
| `MQ_RESULT_MAX_INLINE_BYTES` | `262144` | Max JSON size (bytes) stored inline in Redis; when exceeded, result is stored externally (local/MinIO) and Redis stores a small ref envelope (`0` disables externalization). |
| `MQ_RESULT_STORE` | `local` | External result store backend: `local` or `minio`. |
| `MQ_RESULT_LOCAL_DIR` | `local/mq_results` | Base directory for `local` external results. |
| `MQ_RESULT_MINIO_ENDPOINT` | _(empty)_ | MinIO endpoint for `minio` result store (TODO implementation). |
| `MQ_RESULT_MINIO_BUCKET` | _(empty)_ | MinIO bucket for `minio` result store (TODO implementation). |
| `MQ_STREAM_MAXLEN` | `20000` | Max length for Redis Streams (approximate trimming). |
| `MQ_PROGRESS_MAX_STRING_CHARS` | `4000` | Per-string truncation limit for progress event payloads stored in Redis Streams (protects Redis from huge trace/tool outputs). |
| `MQ_PROGRESS_MAX_LIST_ITEMS` | `200` | Per-list truncation limit for progress event payloads stored in Redis Streams. |
| `MQ_PROGRESS_MAX_DEPTH` | `6` | Max recursion depth when trimming progress event payloads before persisting to Redis Streams. |
| `MQ_FAILFAST_ON_REDIS_DOWN` | _(empty)_ | Whether to fail-fast when Redis is unavailable: default is fail-fast in `celery` mode and best-effort in `inprocess` mode. |
| `FILE_OP_LOCK_TTL_SECONDS` | `21600` | Distributed file-operation lock TTL (seconds; shared by index/delete). |
| `CELERY_TASK_MAX_RETRIES` | `3` | Maximum retry attempts for task exceptions. |
| `CELERY_TASK_RETRY_COUNTDOWN_SECONDS` | `5` | Countdown (seconds) before retrying on exceptions. |
| `CELERY_TASK_LOCK_MAX_RETRIES` | `30` | Maximum retry attempts when file lock is busy. |
| `CELERY_TASK_LOCK_RETRY_COUNTDOWN_SECONDS` | `2` | Countdown (seconds) before retrying when file lock is busy. |
| `MQ_AUTO_START_WORKERS` | `true` | Auto-start `rag-arc-worker-*` containers during `./start.sh` when `TASK_QUEUE_MODE=celery`. |
| `MQ_SYNC_TO_POSTGRES_ENABLED` | `true` | Start `rag-arc-mq-sync` daemon during `./start.sh` to archive Redis Streams into Postgres. |
| `MQ_SYNC_POLL_INTERVAL_SECONDS` | `2` | Sync poll interval seconds (daemon mode). |
| `MQ_SYNC_BATCH_SIZE` | `2000` | Max stream entries read per sync pass. |
| `MQ_SYNC_BLOCK_MS` | `1000` | Redis XREAD block time (ms) per sync pass (`0` disables blocking). |
| `MQ_STARTUP_HEALTHCHECK` | `true` | Run a small MQ startup healthcheck during `./start.sh` (writes a tiny event, runs one sync pass, verifies tables). |

### 5.1.1 Running locally / in tests

- When `TASK_QUEUE_MODE=celery` and `MQ_AUTO_START_WORKERS=true`, `./start.sh` automatically starts the Celery worker containers.
- Stop Celery workers: `./stop.sh` (or `bash scripts/mq_tools/stop_mq_workers_local.sh`).
- Manual start (outside Docker): `bash scripts/mq_tools/start_mq_workers_local.sh` (loads `.env`, logs in `log/mq_workers/`).
- Optional: archive Redis Streams into Postgres: `uv run python scripts/mq_tools/message_queue_sync.py --daemon` (or `--once`).

## 6. DeepSearch Defaults

Planner/graph defaults. Leave as-is unless customizing behavior.

Note: DeepSearch tool parameters (including the deterministic `code.python` math/finance verification tool) are configured via `config/json_configs/deepsearch_service.json` → `tool_manager.enabled_tools[...]` rather than individual environment variables.

### code.python tool parameters (JSON config)

Location: `config/json_configs/deepsearch_service.json` → `tool_manager.enabled_tools["code.python"].params`.

| Key | Default | Description |
| --- | --- | --- |
| `allowed_imports` | `["json","math","decimal","fractions","statistics","datetime","numpy","pandas","scipy"]` | Whitelist of importable top-level packages inside `code.python` (anything else raises `ImportError`). |
| `timeout_seconds` | `6.0` | Wall-clock timeout for the Python subprocess execution. |
| `max_code_chars` | `12000` | Maximum accepted size of `extra.code`; larger payloads fail fast with `code_too_large`. |
| `max_stdout_chars` | `8000` | Maximum captured stdout characters (bounded writer; additional output is truncated). |
| `max_stderr_chars` | `8000` | Maximum captured stderr characters (bounded writer; additional output is truncated). |
| `max_result_chars` | `12000` | Maximum serialized length of `result` (or its JSON string) returned as `result_text`. |
| `max_memory_mb` | `1024` | Best-effort memory cap for the subprocess (applies `RLIMIT_AS/RLIMIT_DATA` when available). |
| `emit_result_evidence` | `true` | When true, the tool emits an `EvidenceChunk` (source=`code.python`) so the next `<think>` can read the result via `context_evidences`. |
| `disable_file_io` | `true` | When true, `open()` is blocked inside the subprocess (prevents file I/O by default). |

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
| `DEEPSEARCH_ARTIFACT_DIR` | _(empty)_ | Optional: per-run DeepSearch artifact root (writes `run_id/plan_result.json`, `reasoning.json`, `report.json`, `report.md`, and snapshot/manifest JSON; when `artifacts.version=2`, also writes `manifest.json`, `dev.json`, `public.json`, and `state_snapshot.json` becomes a lightweight manifest; when `artifacts.dedupe.enabled=true`, also writes `evidence_pool.json` and replaces duplicated large blocks in `reasoning.json`/`report.json` with refs). |
| `DEEPSEARCH_TOOL_ARTIFACT_DIR` | `./local/deepsearch_artifacts` | Output directory for tool telemetry/artifacts (also used as the default run artifact root in `config/json_configs/deepsearch_service.json`). |
| `DEEPSEARCH_ALLOW_EXTERNAL_CHANNEL` | `false` | Planner-only flag for emitting `web` steps (used when `DEEPSEARCH_EXTERNAL_SEARCH_ENABLED` is not set). |
| `DEEPSEARCH_EXTERNAL_SEARCH_ENABLED` | `false` | Runtime override for external search enablement (config SoT: `external_channel.enabled` + `gap_detection.enable_external_on_gap`). |
| `DEEPSEARCH_SECTIONWISE_WRITER` | `false` | Enable section-wise report writing with Memory Bank retrieval + recency retention. |
| `DEEPSEARCH_BUDGET_TIER` | _(empty)_ | Optional runtime override for complexity→budget scaling (`low` / `default`); when empty, DeepSearch uses a heuristic based on the question. |
| `DEEPSEARCH_TELEMETRY_ENABLED` | `true` | Enable telemetry capture for tool runs (local artifacts). |
| `TAVILY_API_KEY` | _(empty)_ | API key for Tavily web search (used by both HippoRAG Q&A and DeepSearch when web search is enabled). |
| `DEEPSEARCH_WEB_PROVIDER` | _(empty)_ | External search routing hint (`tavily` / `tool` / `mcp`; unknown values fall back to `tavily`). |
| `DEEPSEARCH_EXTERNAL_CACHE_MODE` | `auto` | External search record/replay mode: `off` / `record` / `replay` / `auto`. |
| `DEEPSEARCH_EXTERNAL_CACHE_DIR` | `./local/deepsearch_artifacts/external_cache` | External search cache directory. |
| `DEEPSEARCH_TOOL_HINTS` | _(empty)_ | JSON list to override planner tool hints. |
| `DEEPSEARCH_TOOL_MCP_CONFIG_PATH` | _(empty)_ | Custom JSON config for tool MCP server. |
| `DEEPSEARCH_TOOL_MCP_ADAPTER_CONFIG` | _(empty)_ | JSON file describing adapter overrides. |
| `DEEPSEARCH_TOOL_MCP_ADAPTER_NAME` | _(empty)_ | Adapter name when not using config path. |
| `DEEPSEARCH_TOOL_MCP_ADAPTER_PARAMS` | `{}` | JSON dictionary of adapter kwargs. |
| `DEEPSEARCH_TOOL_MCP_SCOPE_ID` | _(empty)_ | Scope ID used when MCP server runs standalone. |
| `DEEPSEARCH_TOOL_MCP_SCOPE_TYPE` | `owner` | Scope type label. |
| `DEEPSEARCH_TOOL_MCP_SCOPE_LABELS` | `[]` | JSON list of labels for MCP scope. |
| `DEEPSEARCH_TOOL_MCP_SCOPE_ATTRIBUTES` | `{}` | JSON dict of extra scope attributes. |
| `DEEPSEARCH_TOOL_MCP_TOOLS` | _(empty)_ | Optional comma separated tool allowlist; use `__all__` to expose every built-in tool. |
| `DEEPSEARCH_ALLOW_SEMANTIC_CHANNEL` | `true` | Allow semantic traversal branch. |
| `DEEPSEARCH_CHAIN_DEPTH` | `4` | Graph traversal depth. |
| `DEEPSEARCH_TOOL_CONTEXT_MAX_EVIDENCES` | `5` | Max number of `context_evidences` sent to tool calls (recency retention). |
| `DEEPSEARCH_TOOL_CONTEXT_MAX_CHARS` | `800` | Max characters per evidence content in tool prompts (truncates beyond this). |
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
| `JWT_SECRET_KEY` | _(empty)_ | JWT signing secret. If empty, the API auto-generates one and persists it under `RAGARC_RUNTIME_DIR` (default: `./local/runtime/jwt_secret_key`). Set explicitly in production. |
| `HF_TOKEN` | _(empty)_ | HuggingFace token for downloading gated models (optional). |
| `HF_ENDPOINT` | _(empty)_ | Optional HuggingFace endpoint override (e.g. `https://hf-mirror.com`). |
| `LOG_LEVEL` | `INFO` | Python logging level (`DEBUG`, `INFO`, etc.). |
| `RAGARC_DEPENDENCY_CHECK_MODE` | `warn` | Dependency check mode for app startup (Postgres/Redis/Neo4j): `off`/`warn`/`strict`. Note: the API startup currently defaults to `strict` when this env var is unset. |
| `RAGARC_INDEXING_DEPENDENCY_CHECK_MODE` | `strict` | Dependency check mode for knowledge indexing tasks (used by `/knowledge/*` indexing and Celery tasks): `off`/`warn`/`strict`. Unit tests set this to `off` for hermetic runs. |
| `KNOWLEDGE_ACTIVE_CHECK_BLOB_EXISTS` | `true` | Whether `Knowledge.is_file_active` also checks that the underlying blob exists (prevents returning citations that later 404 in file download flows). |

## 8. File Storage & Parser Paths

| Variable | Default | Description |
| --- | --- | --- |
| `PARSER_PARSE_MODE` | `native` | PDF/image parse mode: `native` (no OCR; PDF text extraction only), `dotsocr` (local OCR), `mineru` (remote MinerU service). |
| `PARSER_OUTPUT_DIR` | `./data/parsed_files` | Unified base directory for parser outputs (native/dots_ocr/vlm_ocr subfolders). |
| `NATIVE_PARSER_OUTPUT_DIR` | _(empty)_ | Optional override for native parser output directory. |
| `DOTSOCR_OUTPUT_DIR` | _(empty)_ | Optional override for dots_ocr output directory. |
| `VLMOCR_OUTPUT_DIR` | _(empty)_ | Optional override for VLM OCR output directory. |
| `MINERU_SERVER_URL` | _(empty)_ | Required when `PARSER_PARSE_MODE=mineru`: MinerU server base URL (e.g. `http://127.0.0.1:8899`). |
| `MINERU_HEALTHCHECK_TIMEOUT_S` | `2` | Startup/indexing healthcheck timeout seconds for `GET $MINERU_SERVER_URL/health` when `PARSER_PARSE_MODE=mineru`. |
| `MINERU_FALLBACK_TO_NATIVE_ON_FAILURE` | `true` | When `PARSER_PARSE_MODE=mineru`, fallback to native PDF text extraction if MinerU parsing fails (e.g. service not running). Fallback is recorded in parse result metadata (`metadata.parser_fallback`). |
| `MINERU_TIMEOUT_S` | `900` | Optional: HTTP timeout seconds for remote MinerU parsing/downloads. |
| `MINERU_POLL_INTERVAL_S` | `5` | Optional: polling interval seconds for MinerU async parse status. |
| `MINERU_POLL_TIMEOUT_S` | `0` | Optional: max seconds to wait for MinerU parse completion; `0` or negative means no limit. |
| `MINERU_START_PAGE` | `0` | Optional: start page (0-based) for MinerU parsing. |
| `MINERU_END_PAGE` | _(empty)_ | Optional: end page (0-based, inclusive). If empty, parse to the end. |
| `TOKEN_CHUNK_SIZE` | `1000` | Token chunk size for `token_chunker` (also used as `semantic_unit_chunker.fallback_chunker_config`). |
| `TOKEN_CHUNK_OVERLAP` | `100` | Token overlap for `token_chunker` (also used as `semantic_unit_chunker.fallback_chunker_config`). |
| `TOKEN_URL_ATOMIC_CONTEXT_TOKENS` | `10` | URL atomic protection: keep this many tokens before/after each URL together (applies to `token_chunker` and semantic-unit fallback). |
| `OCR_MODEL_NAME` | _(empty)_ | Optional backward-compatible OCR model name alias. |
| `RAGARC_RUNTIME_DIR` | `./local/runtime` | Fallback runtime root when preferred local directories are not writable. |
| `LOCAL_FILE_STORAGE_PATH` | `./data/files` | Default root for `local_blob_store` when JSON `base_path` is not provided (relative paths are resolved against the repo root). |

## 9. Neo4j Graph Database

| Variable | Default | Description |
| --- | --- | --- |
| `NEO4J_URL` | `bolt://localhost:7687` | Connection string for Neo4j. |
| `NEO4J_USERNAME` | `neo4j` | Neo4j username. |
| `NEO4J_PASSWORD` | _(empty)_ | Neo4j password. |
| `NEO4J_DATABASE` | `neo4j` | Database name/alias. |
| `EXPOSE_NEO4J` | `false` | Whether to expose Neo4j browser/bolt port. |
| `NEO4J_HTTP_PORT` | `7474` | Host HTTP port when `EXPOSE_NEO4J=true`. |
| `NEO4J_BOLT_PORT` | `7687` | Host bolt port when `EXPOSE_NEO4J=true`. |

## 10. Optional MinIO Object Storage

| Variable | Default | Description |
| --- | --- | --- |
| `MINIO_USERNAME` | `ROOTNAME` | MinIO access key / username (used only when MinIO integration is enabled). |
| `MINIO_PASSWORD` | `CHANGEME123` | MinIO secret key / password. |

Common MinIO variables (set them only when enabling object storage integration):
- `MINIO_ENDPOINT`
- `MINIO_BUCKET`
- `MINIO_SECURE`

These are not required for the default local/Docker setup.

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

## 17. JSON Config Notes (non-env)

RAG-ARC follows a single-source-of-truth config flow:

- Runtime secrets / deployment-specific values live in environment variables (`.env`, see `.env.example`).
- Tunable knobs (thresholds, budgets, tool selection, paths, feature gates) live in JSON under `config/json_configs/`.

Entry points:

- DeepSearch service: `config/json_configs/deepsearch_service.json`
- RAG inference (HippoRAG Q&A): `config/json_configs/rag_inference.json`
- Knowledge pipelines: `config/json_configs/knowledge.json`

DeepSearch web search policy (in `config/json_configs/deepsearch_service.json`):

- `planner.web_step_policy="realtime_required"` injects/forces at least one `channel="web"` step when the question asks for realtime/latest/current info (e.g. FX rates/news).
- `external_channel.execute_forced_tasks_without_gap=true` executes those forced tasks even when gap detection thinks coverage is sufficient.

DeepSearch tool budget (in `config/json_configs/deepsearch_service.json`):

- `tool_budget.max_calls_total` caps total tool invocations per DeepSearch run (tool_manager + optional external calls; does not count graph adapter traversals).
- Remaining budget is attached to `graph_context.metadata.tool_budget` for LLM visibility and also surfaced in tool diagnostics.

---

**Tip:** Copy `.env.example` to `.env`, fill in the required secrets (`OPENAI_API_KEY`/`OPENAI_BASE_URL` (or per-module `*_API_KEY`/`*_API_BASE_URL`) and `JWT_SECRET_KEY`), and the rest should work with built-in defaults. For advanced tuning, prefer changing `config/` (JSON/Python) over adding many env overrides.
