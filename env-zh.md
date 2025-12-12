# 环境变量配置说明

项目通过根目录的 `.env` 控制所有行为。默认的 `.env.example` 已经适配本地开发（Docker 服务运行在 `localhost`），通常只需要填入模型/LLM 的 API Key。本文档按模块说明每一个参数的默认值与作用。

## 1. 模型与 LLM 提供方

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CHAT_MODEL_PROVIDER` | `openai` | 对话模型提供方（`openai` / `huggingface` 等）。 |
| `CHAT_API_KEY` | _(空)_ | 对话模型的 API Key（使用云端模型时必填）。 |
| `CHAT_API_BASE_URL` | _(空)_ | OpenAI 兼容 API 的 Base URL。 |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | 默认聊天模型名称。 |
| `EMBEDDING_MODEL_PROVIDER` | `openai` | 嵌入模型提供方。 |
| `EMBEDDING_API_KEY` | _(空)_ | 嵌入模型的 API Key。 |
| `EMBEDDING_API_BASE_URL` | _(空)_ | 嵌入模型的 Base URL。 |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | 默认嵌入模型。 |
| `OCR_MODEL_PROVIDER` | `openai` | OCR/VLM 模型提供方（`openai`、`vllm`、`dots_ocr` 等）。 |
| `OCR_API_KEY` | _(空)_ | OCR/VLM 模型的 API Key。 |
| `OCR_API_BASE_URL` | _(空)_ | OCR/VLM 的 Base URL。 |
| `OPENAI_OCR_MODEL` | `gpt-4o-mini` | 默认 OCR/VLM 模型名称。 |
| `DOTS_OCR_CACHE_FOLDER` | `./models/dots_ocr` | dots_ocr 模型缓存路径。 |
| `RERANKER_MODEL_NAME` | `Qwen/Qwen3-Reranker-0.6B` | 默认 reranker 模型。 |
| `RERANKER_CACHE_FOLDER` | `./models/Qwen` | reranker 缓存目录。 |
| `OPENAI_API_KEY` | _(空)_ | 全局备用 OpenAI Key。 |
| `OPENAI_BASE_URL` | _(空)_ | 全局备用 OpenAI Base URL。 |
| `DEVICE` | `xxx` | Torch 设备（如 `cpu`、`cuda:0`）。无 GPU 则填 `cpu`。 |
| `EMBEDDING_MODEL_NAME` | `Qwen/Qwen3-Embedding-0.6B` | 使用本地 HuggingFace 嵌入时的模型。 |
| `MODEL_PROFILE` | `api` | 选择配置档（`api` 或 `local`），影响默认 JSON 配置。 |

## 2. 证据输出控制

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `ENABLE_ALL_EVIDENCE` | `false` | 为 `true` 时关闭所有截断，完整返回证据。 |
| `CHAT_TOP_CHUNKS` | `5` | Chat 证据中最多保留的 chunk 数量。 |
| `CHAT_TOP_TRIPLES` | `5` | Chat 证据中最多保留的图三元组数量。 |
| `CHAT_TOP_SEED_ENTITIES` | `5` | Chat 证据中最多保留的种子实体数量。 |
| `DEEPSEARCH_TOP_CHUNKS` | `10` | DeepSearch 证据中最多保留的 chunk 数量。 |
| `DEEPSEARCH_TOP_TRIPLES` | `30` | DeepSearch 证据中最多保留的图三元组数量。 |
| `DEEPSEARCH_TOP_SEED_ENTITIES` | `15` | DeepSearch 证据中最多保留的种子实体数量。 |
| `DEEPSEARCH_GRAPH_NODE_LIMIT` | `75` | DeepSearch 图快照（实体 + chunk）的节点上限。 |
| `DEEPSEARCH_GRAPH_EDGE_LIMIT` | `200` | DeepSearch 图快照中最多保留的边数量。 |

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
| `DEEPSEARCH_PLANNER_LLM_PROVIDER` ~ `DEEPSEARCH_PLANNER_MAX_RETRIES` | _(空)_ | 当需要为规划器单独配置 LLM 时填入（提供方/模型/重试等）。 |
| `DEEPSEARCH_PERSIST_PLAN` | `true` | 是否落盘保存规划。 |
| `DEEPSEARCH_PLAN_OUTPUT_DIR` | `./local/deepsearch_runs` | 规划输出目录。 |
| `DEEPSEARCH_TOOL_ARTIFACT_DIR` | `./local/deepsearch_artifacts` | 工具执行日志目录。 |
| `DEEPSEARCH_ALLOW_EXTERNAL_CHANNEL` | `false` | 是否允许外部搜索。 |
| `DEEPSEARCH_EXTERNAL_SEARCH_ENABLED` | `false` | 启用外部搜索时需同时提供 API Key。 |
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
| `LOG_LEVEL` | `INFO` | 日志等级。 |

## 8. 文件/解析路径

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PARSER_OUTPUT_DIR` | `./data/parsed_files` | 解析输出目录。 |
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

## 10. 可选的 MinIO

`.env.example` 末尾提供了 MinIO 相关变量（默认注释）。需要使用对象存储时，取消注释并填写 `MINIO_ENDPOINT`、`MINIO_USERNAME`、`MINIO_PASSWORD`、`MINIO_BUCKET`、`MINIO_SECURE` 等。

---

**使用建议**：复制 `.env.example` 为 `.env`，填入自己使用的模型/API Key，其余配置保持默认即可完成本地部署。仅当要接入其他数据库/服务或自定义 DeepSearch 行为时，再根据上表调整相应变量。
