# 环境变量配置说明

项目通过根目录的 `.env` 控制所有行为。默认的 `.env.example` 已经适配本地开发（Docker 服务运行在 `localhost`），通常只需要填入模型/LLM 的 API Key。本文档按模块说明每一个参数的默认值与作用。

## 1. 模型与 LLM 提供方

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CHAT_MODEL_PROVIDER` | `openai` | 对话模型提供方（`openai`=OpenAI 兼容 API，`huggingface`=本地 Transformers）。 |
| `CHAT_MODEL_NAME` | _(空)_ | 可选：优先使用的对话模型名（填写后会覆盖 `OPENAI_CHAT_MODEL`）。 |
| `CHAT_MODEL_DEVICE` | `cpu` | HuggingFace 对话模型运行设备（仅当 `CHAT_MODEL_PROVIDER=huggingface` 时使用）。 |
| `CHAT_MODEL_CACHE_FOLDER` | _(空)_ | 可选：HuggingFace 对话模型权重/Tokenizer 缓存目录。 |
| `CHAT_API_KEY` | _(空)_ | 对话模型的 API Key（使用云端 API 时必填）。 |
| `CHAT_API_BASE_URL` | _(空)_ | OpenAI 兼容 API 的 Base URL（例如 `https://api.openai.com/v1`）。 |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | 兼容/默认的对话模型名（当 `CHAT_MODEL_NAME` 为空时使用）。 |
| `OPENAI_API_BASE` | _(空)_ | 可选：历史兼容的 OpenAI Base URL 别名。 |
| `EMBEDDING_MODEL_PROVIDER` | `openai` | 嵌入模型提供方（`openai`=OpenAI 兼容 API，`huggingface`=本地 SentenceTransformers）。 |
| `EMBEDDING_API_KEY` | _(空)_ | 嵌入模型的 API Key（使用云端 API 时必填）。 |
| `EMBEDDING_API_BASE_URL` | _(空)_ | 嵌入模型的 Base URL。 |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | 默认嵌入模型名称。 |
| `EMBEDDING_DEVICE` | `cpu` | HuggingFace 嵌入模型运行设备（仅当 `EMBEDDING_MODEL_PROVIDER=huggingface` 时使用）。 |
| `EMBEDDING_CACHE_FOLDER` | _(空)_ | 可选：HuggingFace 嵌入模型缓存目录。 |
| `EMBEDDING_DIMENSIONS` | _(空)_ | 本地 HuggingFace embedding 必填：嵌入向量维度；使用 OpenAI 兼容 API 时可留空，系统会自动探测维度（也可填入作为覆盖）。 |
| `OCR_MODEL_PROVIDER` | `openai` | OCR/VLM 提供方（`openai`、`vllm`、`dots_ocr` 等）。 |
| `OCR_API_KEY` | _(空)_ | OCR/VLM 的 API Key（使用云端 API 时必填）。 |
| `OCR_API_BASE_URL` | _(空)_ | OCR/VLM 的 Base URL。 |
| `OPENAI_OCR_MODEL` | `gpt-4o-mini` | 默认 OCR/VLM 模型名称。 |
| `DOTS_OCR_CACHE_FOLDER` | `./models/dots_ocr` | dots_ocr 模型缓存路径。 |
| `USE_CHINA_MIRROR` | `false` | 是否启用 HuggingFace 国内镜像（影响本地 embedding/reranker 等）。 |
| `RERANKER_MODEL_NAME` | `Qwen/Qwen3-Reranker-0.6B` | 默认本地 reranker 模型名（`MODEL_PROFILE=local` 时使用）。 |
| `RERANKER_CACHE_FOLDER` | `./models/Qwen` | reranker 缓存目录。 |
| `RERANKER_DEVICE` | `cpu` | reranker 运行设备。 |
| `OPENAI_API_KEY` | _(空)_ | 可选：全局备用 OpenAI Key（组件 Key 为空时复用）。 |
| `OPENAI_BASE_URL` | _(空)_ | 可选：全局备用 OpenAI Base URL。 |
| `DEVICE` | `cpu` | 可选：共享默认设备（当各组件设备变量为空时使用）。 |
| `EMBEDDING_MODEL_NAME` | `Qwen/Qwen3-Embedding-0.6B` | 使用本地 HuggingFace 嵌入时的模型名（`EMBEDDING_MODEL_PROVIDER=huggingface`）。 |
| `MODEL_PROFILE` | `api` | 选择配置档（`api` 或 `local`），影响默认 JSON 配置。 |

## 1.1 索引与存储路径

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `FILE_STORE_BASE_PATH` | `./data/file_store` | 文件原始内容存储目录（本地 blob store）。 |
| `PARSED_CONTENT_STORE_BASE_PATH` | `./data/parsed_content_store` | 解析结果存储目录。 |
| `CHUNK_STORE_BASE_PATH` | `./data/chunk_store` | Chunk 存储目录。 |
| `FAISS_INDEX_PATH` | `./data/unified_faiss_index` | 统一 FAISS 索引目录。 |
| `BM25_INDEX_PATH` | `./data/unified_bm25_index` | 统一 BM25 索引目录。 |
| `GRAPH_STORAGE_PATH` | `./data/graph_index_neo4j` | 图索引/向量缓存落盘目录（Neo4j HippoRAG）。 |
| `GRAPH_INDEX_NAME` | `index` | 图索引文件前缀名。 |

## 2. 证据输出控制

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ENABLE_ALL_EVIDENCE` | `false` | 为 `true` 时关闭所有截断，完整返回证据。 |
| `CHAT_TOP_CHUNKS` | `5` | Chat 证据中最多保留的 chunk 数量。 |
| `CHAT_TOP_TRIPLES` | `5` | Chat 证据中最多保留的图三元组数量。 |
| `CHAT_TOP_SEED_ENTITIES` | `5` | Chat 证据中最多保留的种子实体数量。 |
| `DEEPSEARCH_TOP_CHUNKS` | `10` | DeepSearch 证据中最多保留的 chunk 数量，同时也是报告附录中显示原文预览（前100字符）的数量。 |
| `DEEPSEARCH_TOP_TRIPLES` | `30` | DeepSearch 证据中最多保留的图三元组数量。 |
| `DEEPSEARCH_TOP_SEED_ENTITIES` | `15` | DeepSearch 证据中最多保留的种子实体数量。 |
| `DEEPSEARCH_GRAPH_NODE_LIMIT` | `75` | DeepSearch 图快照（实体 + chunk）的节点上限。 |
| `DEEPSEARCH_GRAPH_EDGE_LIMIT` | `200` | DeepSearch 图快照中最多保留的边数量。 |
| `DEEPSEARCH_MAX_REASONING_STEPS` | `32` | DeepSearch payload 中最多保留的 reasoning steps 数量。 |
| `DEEPSEARCH_MAX_STAGE_HISTORY` | `10` | DeepSearch payload 中最多保留的 stage_history 条数。 |
| `DEEPSEARCH_MAX_EXTERNAL_CALLS` | `5` | DeepSearch payload 中最多保留的 external_calls 条数。 |
| `DEEPSEARCH_MAX_TOOL_METADATA` | `5` | DeepSearch payload 中最多保留的 tool_results 条数。 |

## 3. 开发模式与租户

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DEVELOP_MODE` | `false` | 为 `true` 时 Docker 暴露端口，CLI 自动创建测试用户。 |
| `DEVELOP_OWNER_ID` | `00000000-0000-0000-0000-000000000001` | 开发模式下 CLI/测试默认的 owner UUID。 |
| `DEVELOP_OWNER_USERNAME` | `dev_cli_user` | 自动创建的测试用户用户名。 |
| `DEVELOP_OWNER_PASSWORD` | `dev-cli-password` | 测试用户密码。 |
| `ADMIN_OWNER_ID` | `00000000-0000-0000-0000-00000000ABCD` | 可查看所有租户数据的管理员 UUID，留空则禁用。 |

## 4. PostgreSQL

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `POSTGRES_HOST` | `localhost` | PostgreSQL 主机。 |
| `POSTGRES_PORT` | `5555` | Docker 内部端口。 |
| `POSTGRES_USER` | `postgres` | 数据库用户名。 |
| `POSTGRES_PASSWORD` | `123` | 数据库密码。 |
| `POSTGRES_DB` | `rag_arc` | 数据库名称。 |
| `POSTGRES_HOST_PORT` | `5555` | `EXPOSE_POSTGRES=true` 时映射到宿主机的端口。 |
| `EXPOSE_POSTGRES` | `false` | 是否对宿主机暴露 PostgreSQL。 |

## 5. Redis

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `REDIS_HOST` | `localhost` | Redis 主机。 |
| `REDIS_PORT` | `6379` | Redis 端口。 |
| `REDIS_DB` | `0` | Redis 逻辑库。 |
| `REDIS_PASSWORD` | _(空)_ | Redis 密码。 |
| `REDIS_HOST_PORT` | `6379` | `EXPOSE_REDIS=true` 时映射到宿主机的端口。 |
| `EXPOSE_REDIS` | `false` | 是否对宿主机暴露 Redis。 |

## 6. DeepSearch 配置

若无特殊需求，请保留默认值；只有在需要自定义规划器或工具链时才修改。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `DEEPSEARCH_DEFAULT_ADAPTER` | `hipporag` | 图适配器名称。 |
| `DEEPSEARCH_PLANNER_MODE` | `react` | 规划器模式。 |
| `DEEPSEARCH_GRAPH_STRATEGY` | `ppr_chain` | 图推理策略。 |
| `DEEPSEARCH_PLANNER_MAX_STEPS` | `6` | 规划器最大步数。 |
| `DEEPSEARCH_PLANNER_ENABLE_SUBQUESTION` | `true` | 是否允许拆分子问题。 |
| `DEEPSEARCH_PLANNER_DISABLE_LLM` | `false` | 禁用规划器 LLM（调试用）。 |
| `DEEPSEARCH_PLANNER_LLM_PROVIDER` | _(空)_ | 可选：规划器专用 LLM Provider（留空则复用全局对话配置）。 |
| `DEEPSEARCH_PLANNER_MODEL_NAME` | _(空)_ | 可选：规划器专用模型名。 |
| `DEEPSEARCH_PLANNER_MAX_TOKENS` | _(空)_ | 可选：规划器专用 max tokens。 |
| `DEEPSEARCH_PLANNER_TEMPERATURE` | _(空)_ | 可选：规划器专用 temperature。 |
| `DEEPSEARCH_PLANNER_API_KEY` | _(空)_ | 可选：规划器专用 API Key。 |
| `DEEPSEARCH_PLANNER_BASE_URL` | _(空)_ | 可选：规划器专用 Base URL。 |
| `DEEPSEARCH_PLANNER_ORGANIZATION` | _(空)_ | 可选：规划器专用 organization。 |
| `DEEPSEARCH_PLANNER_TIMEOUT` | _(空)_ | 可选：规划器专用请求超时。 |
| `DEEPSEARCH_PLANNER_MAX_RETRIES` | _(空)_ | 可选：规划器专用重试次数。 |
| `DEEPSEARCH_PERSIST_PLAN` | `true` | 是否落盘保存规划。 |
| `DEEPSEARCH_PLAN_OUTPUT_DIR` | `./local/deepsearch_runs` | 规划输出目录。 |
| `DEEPSEARCH_TOOL_ARTIFACT_DIR` | `./local/deepsearch_artifacts` | 工具执行日志目录。 |
| `DEEPSEARCH_ALLOW_EXTERNAL_CHANNEL` | `false` | 是否允许外部搜索。 |
| `DEEPSEARCH_EXTERNAL_SEARCH_ENABLED` | `false` | 启用外部搜索时需同时提供 API Key。 |
| `DEEPSEARCH_TELEMETRY_ENABLED` | `true` | 是否启用工具运行遥测（本地 artifacts）。 |
| `TAVILY_API_KEY` | _(空)_ | Tavily 搜索的 Key（启用外部搜索时必填）。 |
| `DEEPSEARCH_WEB_PROVIDER` | _(空)_ | 默认 web 搜索 MCP 名称。 |
| `DEEPSEARCH_TOOL_HINTS` | _(空)_ | JSON 字符串，覆盖规划器的工具提示。 |
| `DEEPSEARCH_TOOL_MCP_CONFIG_PATH` | _(空)_ | MCP 服务器 JSON 配置路径。 |
| `DEEPSEARCH_TOOL_MCP_ADAPTER_CONFIG` | _(空)_ | 适配器配置 JSON。 |
| `DEEPSEARCH_TOOL_MCP_ADAPTER_NAME` | _(空)_ | 使用默认配置时指定 adapter 名称。 |
| `DEEPSEARCH_TOOL_MCP_ADAPTER_PARAMS` | `{}` | 适配器参数（JSON 字符串）。 |
| `DEEPSEARCH_TOOL_MCP_SCOPE_ID` | _(空)_ | MCP 默认 Scope ID。 |
| `DEEPSEARCH_TOOL_MCP_SCOPE_TYPE` | `owner` | MCP Scope 类型。 |
| `DEEPSEARCH_TOOL_MCP_SCOPE_LABELS` | `[]` | MCP Scope 标签列表。 |
| `DEEPSEARCH_TOOL_MCP_SCOPE_ATTRIBUTES` | `{}` | MCP Scope 拓展属性。 |
| `DEEPSEARCH_TOOL_MCP_TOOLS` | _(空)_ | MCP 工具白名单，逗号分隔。 |
| `DEEPSEARCH_ALLOW_SEMANTIC_CHANNEL` | `true` | 是否启用语义通道。 |
| `DEEPSEARCH_CHAIN_DEPTH` | `4` | 图遍历层数。 |
| `DEEPSEARCH_ENABLE_FINANCE_HOOKS` | `false` | 启用金融场景特化逻辑。 |
| `DEEPSEARCH_MCP_SERVER_URI` | _(空)_ | 远程 MCP 服务地址。 |
| `DEEPSEARCH_MCP_API_KEY` | _(空)_ | MCP 远程访问 Key。 |
| `DEEPSEARCH_MCP_TRANSPORT` | `auto` | MCP 传输方式。 |
| `DEEPSEARCH_MCP_STDIO_COMMAND` | _(空)_ | stdio 方式时的启动命令。 |
| `DEEPSEARCH_MCP_STDIO_ENV` | `{}` | stdio 进程的环境变量（JSON）。 |
| `DEEPSEARCH_MCP_HEADERS` | `{}` | SSE/HTTP 传输附加 Header。 |
| `DEEPSEARCH_MCP_TIMEOUT` | `30` | MCP HTTP 连接超时。 |
| `DEEPSEARCH_MCP_READ_TIMEOUT` | `300` | MCP HTTP 读取超时。 |
| `DEEPSEARCH_MCP_PERSISTENT_SESSION` | `true` | 是否复用 HTTP 会话。 |
| `DEEPSEARCH_MCP_ENABLE_GRAPH_CONTEXT` | `true` | 是否附带图上下文。 |
| `DEEPSEARCH_MCP_GRAPH_CONTEXT_FIELD` | `__graph_context__` | MCP 请求中的上下文字段。 |
| `DEEPSEARCH_GAP_COVERAGE_THRESHOLD` | `0.7` | 覆盖率阈值，用于触发外部搜索。 |
| `DEEPSEARCH_GAP_CONFIDENCE_THRESHOLD` | `0.6` | 置信度阈值。 |
| `DEEPSEARCH_GAP_EXPECTED_MIN_CHUNKS` | `3` | 期望的最少证据数量。 |
| `DEEPSEARCH_CONSISTENCY_CHECK` | `true` | 启用 LLM 一致性检查，验证报告内容与证据是否一致。 |
| `DEEPSEARCH_PARALLEL_SECTIONS` | `false` | 并行生成报告章节（更快但消耗更多 API 调用）。 |

### MCP 配置示例

```bash
# 当需要把部分工具标记为 mcp_only/mcp_fallback 时，使用下列示例通过远程 MCP 路由
DEEPSEARCH_MCP_SERVER_URI="http://127.0.0.1:8765/mcp/tools"
DEEPSEARCH_MCP_TRANSPORT="sse"
DEEPSEARCH_MCP_HEADERS='{"Authorization": "Bearer your-mcp-token"}'
# 可选：限制 MCP 服务器对外暴露的工具集合
DEEPSEARCH_TOOL_MCP_TOOLS="graph.context_rollup,graph.think"
# 可选：为独立运行的 MCP 服务器注入默认图访问范围
DEEPSEARCH_TOOL_MCP_SCOPE_ID="00000000-0000-0000-0000-000000000001"
DEEPSEARCH_TOOL_MCP_SCOPE_TYPE="owner"
DEEPSEARCH_TOOL_MCP_SCOPE_LABELS='["demo", "shared"]'
```

## 7. 应用级别配置

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `JWT_SECRET_KEY` | `your-secret-key-change-this-in-production` | JWT 签名秘钥，生产环境务必替换。 |
| `HF_TOKEN` | _(空)_ | HuggingFace Token（下载受限模型时使用）。 |
| `HF_ENDPOINT` | _(空)_ | 可选：HuggingFace Endpoint 覆盖（例如 `https://hf-mirror.com`）。 |
| `LOG_LEVEL` | `INFO` | 日志等级。 |

## 8. 文件/解析路径

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PARSER_OUTPUT_DIR` | `./data/parsed_files` | 统一解析输出目录（native/dots_ocr/vlm_ocr 会落到子目录）。 |
| `NATIVE_PARSER_OUTPUT_DIR` | _(空)_ | 可选：原生解析器输出目录覆盖。 |
| `DOTSOCR_OUTPUT_DIR` | _(空)_ | 可选：dots_ocr 输出目录覆盖。 |
| `VLMOCR_OUTPUT_DIR` | _(空)_ | 可选：VLM OCR 输出目录覆盖。 |
| `OCR_MODEL_NAME` | _(空)_ | 可选：历史兼容的 OCR 模型名别名。 |
| `RAGARC_RUNTIME_DIR` | `./local/runtime` | 当解析目录不可写时的备用路径。 |
| `LOCAL_FILE_STORAGE_PATH` | `./local/files` | 本地文件存储根目录。 |

## 9. Neo4j 图数据库

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `NEO4J_URL` | `bolt://localhost:7687` | Neo4j 连接字符串。 |
| `NEO4J_USERNAME` | `neo4j` | Neo4j 用户名。 |
| `NEO4J_PASSWORD` | `12345678` | Neo4j 密码。 |
| `NEO4J_DATABASE` | `neo4j` | 数据库名称。 |
| `EXPOSE_NEO4J` | `false` | 是否开放 Neo4j Browser/Bolt 端口。 |
| `NEO4J_HTTP_PORT` | `7474` | 当 `EXPOSE_NEO4J=true` 时映射到宿主机的 HTTP 端口。 |
| `NEO4J_BOLT_PORT` | `7687` | 当 `EXPOSE_NEO4J=true` 时映射到宿主机的 Bolt 端口。 |

## 10. 可选的 MinIO 对象存储

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MINIO_USERNAME` | `ROOTNAME` | MinIO 用户名/Access Key（仅在启用 MinIO 集成时使用）。 |
| `MINIO_PASSWORD` | `CHANGEME123` | MinIO 密码/Secret Key。 |

`.env.example` 里还提供了以下（默认注释）的占位项：
- `MINIO_ENDPOINT`
- `MINIO_BUCKET`
- `MINIO_SECURE`

仅当需要接入对象存储时才取消注释并填写。

## 11. 构建 / 高级运行参数

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | _(空)_ | 可选：限制本地模型可见的 GPU。 |
| `PYTHONPATH` | _(空)_ | 可选：为子进程附加 Python 导入路径（例如 vLLM 启动器）。 |
| `UV_INSTALL_URL` | `https://astral.sh/uv/install.sh` | 可选：`build.sh` 使用的 `uv` 安装脚本地址。 |
| `UV_INDEX_URL` | `https://pypi.org/simple` | 可选：`build.sh` 使用的 Python 包索引。 |
| `PYTORCH_INDEX_URL` | _(空)_ | 可选：PyTorch wheel 索引覆盖（主要用于 GPU 构建）。 |

## 12. CLI 默认值

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CLI_OWNER_ID` | _(空)_ | 可选：CLI 命令固定使用的 owner id。 |
| `CLI_OWNER_ID_FILE` | _(空)_ | 可选：生成的 owner id 持久化路径（默认：`~/.rag_arc_owner_id`）。 |
| `DEFAULT_OWNER_ID` | _(空)_ | 可选：历史兼容的 owner id 别名。 |
| `RAG_ARC_OWNER_ID` | _(空)_ | 可选：历史兼容的 owner id 别名。 |

## 13. Quick Start / 测试钩子

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `QUICK_START_OWNER_ID` | _(空)_ | 可选：quick-start 示例使用的 owner id。 |
| `RAG_OUTPUT_DIR` | _(空)_ | 可选：RAG pipeline 输出目录。 |
| `DEEPSEARCH_EXPERIMENT_OUTPUT_DIR` | _(空)_ | 可选：DeepSearch 实验输出目录。 |
| `DEEPSEARCH_CITATION_ALIASES` | _(空)_ | 可选：引用别名映射（JSON）。 |
| `DEEPSEARCH_TOOL_AUDIT_LABEL` | _(空)_ | 可选：工具审计记录标签。 |
| `DEEPSEARCH_TOOL_MCP_AUDIT_LABEL` | _(空)_ | 可选：MCP 工具审计记录标签。 |
| `DEEPSEARCH_TOOL_MCP_INSTRUCTIONS` | _(空)_ | 可选：Planner 的 MCP 工具额外指令。 |
| `DEEPSEARCH_TOOL_MCP_SCOPE_OVERRIDE_POLICY` | _(空)_ | 可选：控制 MCP scope 覆盖时机的策略。 |
| `DEEPSEARCH_TOOL_MCP_SCOPE_OVERRIDE_TOKEN` | _(空)_ | 可选：授权 MCP scope 覆盖的 token。 |
| `DEEPSEARCH_RUN_LLM_INTEGRATION_TESTS` | `0` | 可选：设为 `1` 时运行 DeepSearch LLM 集成测试。 |

### 可选：本地模型 smoke tests（pytest）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RUN_RAGARC_GPT2_CHAT_TESTS` | `0` | 选择性开启：设为 `1` 时使用 `models/gpt2` 跑本地对话 smoke test。 |
| `RAGARC_GPT2_MODEL_DIR` | `./models/gpt2` | tiny-gpt2 本地目录。 |
| `RUN_RAGARC_LOCAL_EMBEDDING_TESTS` | `0` | 选择性开启：设为 `1` 时跑本地 embedding smoke test。 |
| `RAGARC_ST_MODEL_SNAPSHOTS` | `./models/all-MiniLM-L6-v2/models--sentence-transformers--all-MiniLM-L6-v2/snapshots` | SentenceTransformer 本地 snapshots 路径。 |
| `RUN_RAGARC_LOCAL_RERANK_TESTS` | `0` | 选择性开启：设为 `1` 时跑本地 rerank smoke test。 |
| `RAGARC_RERANKER_SNAPSHOTS` | `./models/Qwen/models--Qwen--Qwen3-Reranker-0.6B/snapshots` | reranker 本地 snapshots 路径。 |
| `RAGARC_ALLOW_LARGE_MODELS` | `0` | 安全开关：大模型 smoke test 需设为 `1`。 |

## 14. Azure OpenAI（可选）

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `API_VERSION` | _(空)_ | Azure OpenAI API Version（使用 Azure provider 时填写）。 |
| `AZURE_OPENAI_API_KEY` | _(空)_ | Azure OpenAI API Key。 |

## 15. 仅测试用占位变量（env substitution）

这些变量仅用于内部 env-substitution 测试，可保持为空。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `APP_NAME` | _(空)_ | 测试占位。 |
| `APP_VALUE` | _(空)_ | 测试占位。 |
| `BASE_URL` | _(空)_ | 测试占位。 |
| `EXISTING_VAR` | _(空)_ | 测试占位。 |
| `LIST_VAR` | _(空)_ | 测试占位。 |
| `MIXED_VAR` | _(空)_ | 测试占位。 |
| `NESTED_VAR` | _(空)_ | 测试占位。 |
| `STRING_VAR` | _(空)_ | 测试占位。 |
| `TEST_VAR` | _(空)_ | 测试占位。 |
| `VAR1` | _(空)_ | 测试占位。 |
| `VAR2` | _(空)_ | 测试占位。 |

## 16. 集成 / 测试开关

当所需服务/模型都可用时，可将其设为 `1`（或任意非空值）以选择性开启；留空则跳过对应套件。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `RUN_RAGARC_INTEGRATION_TESTS` | _(空)_ | 运行集成测试套件。 |
| `RUN_RAGARC_POSTGRES_TESTS` | _(空)_ | 运行依赖 Postgres 的测试套件。 |
| `RUN_RAGARC_CHAT_STORAGE_TESTS` | _(空)_ | 运行 chat storage 测试套件。 |
| `RUN_RAGARC_VECTOR_TESTS` | _(空)_ | 运行向量库相关测试套件。 |
| `RAGARC_E2E_TOKEN` | _(空)_ | `test/test_complete_e2e_api.py` 用于 API 鉴权的 token。 |

---

**使用建议**：复制 `.env.example` 为 `.env`，填入自己使用的模型/API Key，其余配置保持默认即可完成本地部署。仅当要接入其他数据库/服务或自定义 DeepSearch 行为时，再根据上表调整相应变量。
