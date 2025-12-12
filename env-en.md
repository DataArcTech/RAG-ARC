# Environment Configuration (English)

All runtime behavior is controlled through `.env`. By default, `.env.example` already contains values that work for local development (Docker services on `localhost`). Only the model/API credentials typically require edits. This document describes every variable, grouped by subsystem.

## 1. Model & LLM Providers

| Variable | Default | Description |
| --- | --- | --- |
| `CHAT_MODEL_PROVIDER` | `openai` | Provider identifier for chat/LangChain-compatible models (`openai`, `huggingface`, etc.). |
| `CHAT_API_KEY` | _(empty)_ | API key for the chat provider (required when using hosted models). |
| `CHAT_API_BASE_URL` | _(empty)_ | Base URL for OpenAI-compatible chat endpoints. |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | Default chat model name. |
| `EMBEDDING_MODEL_PROVIDER` | `openai` | Provider used for dense embeddings. |
| `EMBEDDING_API_KEY` | _(empty)_ | API key for the embedding provider. |
| `EMBEDDING_API_BASE_URL` | _(empty)_ | Base URL for embedding endpoints. |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Default embedding model name. |
| `OCR_MODEL_PROVIDER` | `openai` | Provider for OCR/VLM parsing (`openai`, `vllm`, `dots_ocr`). |
| `OCR_API_KEY` | _(empty)_ | API key for OCR provider. |
| `OCR_API_BASE_URL` | _(empty)_ | Base URL for OCR provider. |
| `OPENAI_OCR_MODEL` | `gpt-4o-mini` | OCR/VLM model name. |
| `DOTS_OCR_CACHE_FOLDER` | `./models/dots_ocr` | Local cache for dots_ocr weights. |
| `RERANKER_MODEL_NAME` | `Qwen/Qwen3-Reranker-0.6B` | Default reranker (chat mode uses listwise reranker if hosted). |
| `RERANKER_CACHE_FOLDER` | `./models/Qwen` | Cache path for reranker checkpoints. |
| `OPENAI_API_KEY` | _(empty)_ | Fallback key reused across OpenAI-compatible modules. |
| `OPENAI_BASE_URL` | _(empty)_ | Fallback base URL reused across modules. |
| `DEVICE` | `xxx` | Torch device (e.g. `cpu`, `cuda:0`). Set to `cpu` when GPU is not available. |
| `EMBEDDING_MODEL_NAME` | `Qwen/Qwen3-Embedding-0.6B` | Local huggingface embedding model when `*_PROVIDER=huggingface`. |
| `MODEL_PROFILE` | `api` | Chooses config profile (`api` or `local`). Impacts default JSON configs. |

## 2. Evidence Output Controls

| Variable | Default | Description |
| --- | --- | --- |
| `ENABLE_ALL_EVIDENCE` | `false` | When `true`, evidence payloads skip all trimming limits. |
| `CHAT_TOP_CHUNKS` | `5` | Maximum chunks returned in chat evidence. |
| `CHAT_TOP_TRIPLES` | `5` | Maximum graph triples returned in chat evidence. |
| `CHAT_TOP_SEED_ENTITIES` | `5` | Maximum seed entities surfaced in chat evidence. |
| `DEEPSEARCH_TOP_CHUNKS` | `10` | Maximum chunks returned in DeepSearch evidence. |
| `DEEPSEARCH_TOP_TRIPLES` | `30` | Maximum graph triples returned in DeepSearch evidence. |
| `DEEPSEARCH_TOP_SEED_ENTITIES` | `15` | Maximum seed entities surfaced in DeepSearch evidence. |
| `DEEPSEARCH_GRAPH_NODE_LIMIT` | `75` | Cap for DeepSearch graph snapshots (entity + chunk nodes). |
| `DEEPSEARCH_GRAPH_EDGE_LIMIT` | `200` | Cap for DeepSearch edge exports between the retained nodes. |

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
| `DEEPSEARCH_PLANNER_LLM_PROVIDER` ... `DEEPSEARCH_PLANNER_MAX_RETRIES` | _(empty)_ | Optional overrides for planner-specific LLM. Fill only when using a dedicated provider. |
| `DEEPSEARCH_PERSIST_PLAN` | `true` | Persist plan JSON to disk. |
| `DEEPSEARCH_PLAN_OUTPUT_DIR` | `./local/deepsearch_runs` | Folder for persisted plans. |
| `DEEPSEARCH_TOOL_ARTIFACT_DIR` | `./local/deepsearch_artifacts` | Output directory for tool telemetry. |
| `DEEPSEARCH_ALLOW_EXTERNAL_CHANNEL` | `false` | Enable web/external channels. |
| `DEEPSEARCH_EXTERNAL_SEARCH_ENABLED` | `false` | Toggle invocation of configured external search provider. |
| `TAVILY_API_KEY` | _(empty)_ | API key for Tavily (when external search enabled). |
| `DEEPSEARCH_WEB_PROVIDER` | _(empty)_ | MCP server name used for web search fallbacks. |
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
| `LOG_LEVEL` | `INFO` | Python logging level (`DEBUG`, `INFO`, etc.). |

## 8. File Storage & Parser Paths

| Variable | Default | Description |
| --- | --- | --- |
| `PARSER_OUTPUT_DIR` | `./data/parsed_files` | Base directory for parser outputs. |
| `RAGARC_RUNTIME_DIR` | `./local/runtime` | Fallback runtime directory when parser output dir is not writable. |
| `LOCAL_FILE_STORAGE_PATH` | `./local/files` | Root directory for locally stored uploads. |

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

## 10. Optional MinIO (Commented Section)

The `.env.example` includes commented placeholders for:
- `MINIO_ENDPOINT`, `MINIO_USERNAME`, `MINIO_PASSWORD`
- `MINIO_BUCKET`, `MINIO_SECURE`

Uncomment and configure them when integrating object storage for parsed files.

---

**Tip:** Copy `.env.example` to `.env`, fill in the API keys for the LLM provider you use, and everything else should function with Docker defaults. Adjust the remaining variables only when you need custom deployments or scopes. test.
