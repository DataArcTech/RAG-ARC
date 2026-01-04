# 环境变量（最小 `.env`）与全量参数说明

RAG-ARC 将配置拆成两层：

- **`.env`**：只放密钥/secret 与功能开关（推荐）。
- **`config/`**：所有可调参数（检索/分块/top_k/阈值/预算/路径等）。

为了兼容性，运行时仍支持大量环境变量作为“覆盖/override”开关。
本文档保留并解释 **所有** 支持的环境变量及其含义。

推荐用法：
1）让 `.env` 保持精简（只放密钥/开关）。
2）业务参数在 `config/`（JSON + Python）里调整。

---

## 0）最小 `.env`（推荐）

从模板开始：

```bash
cp .env.example .env
```

必填密钥（无默认值）：
- `OPENAI_API_KEY` / `OPENAI_BASE_URL`（或分别配置 `CHAT_*` / `EMBEDDING_*` / `OCR_*`）
- `JWT_SECRET_KEY`（建议 `openssl rand -hex 32` 生成）

可选功能开关（有默认值）：
- `TASK_QUEUE_MODE`、`MODEL_PROFILE`、`DEVELOP_MODE`、`ADMIN_OWNER_ID`

可选外部网页检索（仅在配置开启时需要）：
- `TAVILY_API_KEY`

可选基础设施连接信息（本地/Docker 默认值可用；仅在需要连接远端服务时配置）：
- PostgreSQL：`POSTGRES_HOST`、`POSTGRES_PORT`、`POSTGRES_USER`、`POSTGRES_PASSWORD`、`POSTGRES_DB`
- Redis：`REDIS_HOST`、`REDIS_PORT`、`REDIS_DB`、`REDIS_PASSWORD`
- Neo4j：`NEO4J_URL`、`NEO4J_USERNAME`、`NEO4J_PASSWORD`、`NEO4J_DATABASE`

---

## 0.1）高级配置（在 `config/` 修改，不建议堆在 `.env`）

推荐修改位置：

- `config/json_configs/rag_inference*.json`：聊天/RAG 主流程、检索器、重排器、模型选择等。
- `config/json_configs/knowledge*.json`：解析、分块、索引/建图流程等。
- `config/json_configs/deepsearch_service.json`：DeepSearch 的 planner/tools/report/quality gate 等。
- `config/output_limits.py`：API 返回裁剪与证据上限（payload 限制）。
- `config/core/deepsearch/*_defaults.py`：DeepSearch 的 loop/tool/report 默认参数（运行时读取）。

密钥如何在配置里引用：
- JSON 支持 `${ENV_VAR}` 占位符（例如 `${OPENAI_API_KEY}`），因此密钥放 `.env`，参数放 `config/`。

## 1. 模型与 LLM 提供方

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `CHAT_MODEL_PROVIDER` | `openai` | 对话模型提供方（`openai`=OpenAI 兼容 API，`huggingface`=本地 Transformers）。 |
| `CHAT_MODEL_NAME` | _(空)_ | 可选：优先使用的对话模型名；当 `CHAT_MODEL_PROVIDER=huggingface` 时可填写 HuggingFace repo id 或本地模型路径。 |
| `CHAT_MODEL_DEVICE` | `cpu` | HuggingFace 对话模型运行设备（仅当 `CHAT_MODEL_PROVIDER=huggingface` 时使用）。 |
| `CHAT_MODEL_CACHE_FOLDER` | _(空)_ | 可选：HuggingFace 对话模型权重/Tokenizer 缓存目录。 |
| `CHAT_API_KEY` | _(空)_ | **必填**（当 `CHAT_MODEL_PROVIDER=openai`）：对话模型 API Key。 |
| `CHAT_API_BASE_URL` | _(空)_ | **必填**（当 `CHAT_MODEL_PROVIDER=openai`）：OpenAI 兼容 API Base URL（例如 `https://api.openai.com/v1`）。 |
| `OPENAI_CHAT_MODEL` | `gpt-4o-mini` | 兼容/默认的对话模型名（当 `CHAT_MODEL_NAME` 为空时使用）。 |
| `OPENAI_API_BASE` | _(空)_ | 可选：历史兼容的 OpenAI Base URL 别名。 |
| `EMBEDDING_MODEL_PROVIDER` | `openai` | 嵌入模型提供方（`openai`=OpenAI 兼容 API，`huggingface`=本地 SentenceTransformers）。 |
| `EMBEDDING_API_KEY` | _(空)_ | **必填**（当 `EMBEDDING_MODEL_PROVIDER=openai`）：嵌入模型 API Key。 |
| `EMBEDDING_API_BASE_URL` | _(空)_ | **必填**（当 `EMBEDDING_MODEL_PROVIDER=openai`）：嵌入模型 Base URL。 |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | 默认嵌入模型名称。 |
| `EMBEDDING_DEVICE` | `cpu` | HuggingFace 嵌入模型运行设备（仅当 `EMBEDDING_MODEL_PROVIDER=huggingface` 时使用）。 |
| `EMBEDDING_CACHE_FOLDER` | _(空)_ | 可选：HuggingFace 嵌入模型缓存目录。 |
| `EMBEDDING_DIMENSIONS` | _(空)_ | 可选：嵌入向量维度覆盖。留空时系统可自动探测并缓存维度。 |
| `OCR_MODEL_PROVIDER` | `openai` | OCR/VLM 提供方（`openai`、`vllm`、`dots_ocr` 等）。 |
| `OCR_API_KEY` | _(空)_ | OCR/VLM 的 API Key（使用云端 API 时必填）。 |
| `OCR_API_BASE_URL` | _(空)_ | OCR/VLM 的 Base URL。 |
| `OPENAI_OCR_MODEL` | `gpt-4o-mini` | 默认 OCR/VLM 模型名称。 |
| `DOTS_OCR_CACHE_FOLDER` | `./models/dots_ocr` | dots_ocr 模型缓存路径。 |
| `DOTS_OCR_LOADING_METHOD` | `huggingface` | DotsOCR 加载方式（`huggingface` 本地 Transformers，`vllm` 为 OpenAI 兼容服务）。 |
| `DOTS_OCR_USE_CHINA_MIRROR` | `false` | 下载 dots_ocr 权重时是否使用 HuggingFace 镜像。 |
| `DOTS_OCR_USE_SNAPSHOT_DOWNLOAD` | `false` | 是否使用 HuggingFace `snapshot_download` 目录结构（可避免动态模块问题）。 |
| `DOTS_OCR_DEVICE` | `cpu` | DotsOCR 运行设备（默认回退到 `DEVICE`）。 |
| `DOTS_OCR_MODEL_PATH` | `rednote-hilab/dots.ocr` | dots_ocr 的 HuggingFace repo id（当 `DOTS_OCR_LOADING_METHOD=huggingface`）。 |
| `DOTS_OCR_BASE_URL` | `http://localhost:8000/v1` | vLLM/OpenAI 兼容 dots_ocr 服务 base URL（当 `DOTS_OCR_LOADING_METHOD=vllm`）。 |
| `DOTS_OCR_API_KEY` | _(空)_ | vLLM/OpenAI 兼容 dots_ocr 服务 API key（当 `DOTS_OCR_LOADING_METHOD=vllm`）。 |
| `DOTS_OCR_VLLM_MODEL_NAME` | `model` | vLLM/OpenAI 兼容服务暴露的模型名。 |
| `DOTS_OCR_MAX_COMPLETION_TOKENS` | `16384` | OCR 生成的最大 completion tokens。 |
| `DOTS_OCR_TEMPERATURE` | `0.1` | OCR 生成温度。 |
| `DOTS_OCR_TOP_P` | `1.0` | OCR 生成 top-p。 |
| `USE_CHINA_MIRROR` | `false` | 是否启用 HuggingFace 国内镜像（影响本地 embedding/reranker 等）。 |
| `RERANKER_MODEL_NAME` | `Qwen/Qwen3-Reranker-0.6B` | 默认本地 reranker 模型名（`MODEL_PROFILE=local` 时使用）。 |
| `RERANKER_CACHE_FOLDER` | `./models/Qwen` | reranker 缓存目录。 |
| `RERANKER_DEVICE` | `cpu` | reranker 运行设备。 |
| `OPENAI_API_KEY` | _(空)_ | 全局备用 Key（当各组件 `*_API_KEY` 为空时复用）。只要任一 OpenAI 兼容模块启用且未单独配置 `*_API_KEY`，则该项 **必填**。 |
| `OPENAI_BASE_URL` | _(空)_ | 全局备用 Base URL（当各组件 `*_API_BASE_URL` 为空时复用）。只要任一 OpenAI 兼容模块启用且未单独配置 `*_API_BASE_URL`，则该项 **必填**。 |
| `DEVICE` | `cpu` | 可选：共享默认设备（当各组件设备变量为空时使用）。 |
| `EMBEDDING_MODEL_NAME` | `Qwen/Qwen3-Embedding-0.6B` | 嵌入模型名称；当 `EMBEDDING_MODEL_PROVIDER=huggingface` 时可填写 HuggingFace repo id 或本地模型路径。 |
| `MODEL_PROFILE` | `api` | 选择配置档（`api` 或 `local`），影响默认 JSON 配置。 |
| `MINILM_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | `download_models.py` 下载 MiniLM 时使用的默认 repo id。 |
| `MINILM_CACHE_FOLDER` | `./models/all-MiniLM-L6-v2` | `download_models.py` 下载 MiniLM 的缓存目录。 |

## 1.1 索引与存储路径

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `FILE_STORE_BASE_PATH` | `./data/file_store` | 文件原始内容存储目录（本地 blob store；相对路径按项目根目录解析）。 |
| `PARSED_CONTENT_STORE_BASE_PATH` | `./data/parsed_content_store` | 解析结果存储目录（相对路径按项目根目录解析）。 |
| `CHUNK_STORE_BASE_PATH` | `./data/chunk_store` | Chunk 存储目录（相对路径按项目根目录解析）。 |
| `LOCAL_BLOB_STORE_BASE_PATH` | `./data/files` | `LOCAL_FILE_STORAGE_PATH` 的历史别名（仅在 JSON 未提供 `base_path` 时才会使用）。 |
| `FAISS_INDEX_PATH` | `./data/unified_faiss_index` | 统一 FAISS 索引目录。 |
| `BM25_INDEX_PATH` | `./data/unified_bm25_index` | 统一 BM25 索引目录。 |
| `GRAPH_STORAGE_PATH` | `./data/graph_index_neo4j` | 图索引/向量缓存落盘目录（Neo4j HippoRAG）。 |
| `GRAPH_INDEX_NAME` | `index` | 图索引文件前缀名。 |
| `KG_SCHEMA_PATH` | `./kg_schema.yml` | Neo4j HippoRAG 的 KG schema YAML 路径（谓词治理 + 方向敏感集合）。 |

补充说明：
- **FAISS 指纹保护**：FAISS 的 `.pkl` 元数据会写入 `embedding_fingerprint`（provider/model/dim）。当切换 embedding 模型/维度时，建议设置新的 `FAISS_INDEX_PATH`（推荐）或清理重建索引；否则系统会 fail-fast，避免“静默索引污染”。
- **E2E 隔离**：真实服务测试建议把上述路径指向隔离目录（例如 `./local/e2e_*`），避免污染默认的 `./data/*`。

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
| `DEEPSEARCH_WEAVER_EVIDENCE_PREVIEW_CHARS` | `180` | DeepSearch Weaver trace 渲染时的证据预览字符上限。 |
| `DEEPSEARCH_WEAVER_EVIDENCE_SAMPLE_COUNT` | `3` | DeepSearch Weaver trace 渲染时展示的证据样本数量。 |
| `DEEPSEARCH_GRAPH_EXPORT_MAX_EDGES` | `2000` | DeepSearch 子图可视化导出 edges 上限（Neo4j exporter）。 |
| `KNOWLEDGE_GRAPH_EXPORT_MAX_NODES` | `1000` | `/knowledge/graph/export*` 的 max_nodes 上限，用于防止导出过大导致资源消耗过高。 |
| `KNOWLEDGE_GRAPH_EXPORT_MAX_EDGES` | `5000` | `/knowledge/graph/export*` 的 max_edges 上限，用于防止导出过大导致资源消耗过高。 |
| `GRAPH_EXPORT_CHUNK_CONTENT_PREVIEW_CHARS` | `240` | 图导出 payload 中 chunk 内容预览的最大字符数（用于可视化预览，避免返回过大）。 |
| `GRAPH_EXPORT_EDGE_FETCH_FACTOR` | `10` | exporter 抓取 edges 后再采样的倍率（fetch_limit = max_edges * factor）。 |
| `GRAPH_EXPORT_EDGE_FETCH_MAX` | `50000` | exporter edges 抓取绝对上限（防止 Neo4j edge query 过大）。 |
| `SEMANTIC_UNIT_MAX_MATCHED_SLICES` | `3` | 语义单元归并时最多附带的命中 slice 数。 |
| `TABLE_MAX_MERGED_ROWS` | `30` | 表格归并回 anchor 时最多拼接的数据行数。 |
| `SEMANTIC_UNIT_MAX_MERGED_SLICE_CHARS` | `1200` | code/list 归并时每个 slice 追加到 `anchor.content` 的最大字符数。 |
| `SEMANTIC_UNIT_MAX_MERGED_TOTAL_CHARS` | `3000` | 归并追加到 `anchor.content` 的总字符预算。 |

## 2.1 语义单元分块控制项

当 knowledge 配置选择 `semantic_unit_chunker`（例如 `config/json_configs/knowledge_semantic_unit.json`）时，这些参数用于控制父子分块策略。

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `SEMANTIC_CHUNKING_LEVEL` | `basic` | 分块级别：`disabled`/`basic`/`standard`/`advanced`。 |
| `TABLE_SMALL_MAX_TOKENS` | _(空)_ | 表格大小阈值覆盖（留空则使用代码默认值）。 |
| `TABLE_SLICE_MAX_TOKENS` | _(空)_ | 表格 slice 目标 token 上限覆盖。 |
| `TABLE_SLICE_OVERLAP_ROWS` | _(空)_ | 表格 slice 行 overlap 覆盖。 |
| `CODE_SMALL_MAX_TOKENS` | _(空)_ | 代码块大小阈值覆盖。 |
| `CODE_SLICE_MAX_TOKENS` | _(空)_ | 🔶 预留：目前不产出 code slice（fenced code block 不拆分），该参数仅用于后续按函数/类边界切分时的目标 token 预算。 |
| `CODE_SLICE_OVERLAP_LINES` | _(空)_ | 🔶 预留：目前不产出 code slice（fenced code block 不拆分），该参数仅用于后续切分时的行 overlap。 |
| `LIST_SMALL_MAX_TOKENS` | _(空)_ | 列表大小阈值覆盖。 |
| `LIST_SLICE_MAX_TOKENS` | _(空)_ | 列表 slice 目标 token 上限覆盖。 |
| `LIST_SLICE_OVERLAP_ITEMS` | _(空)_ | 列表 slice item overlap 覆盖。 |

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

## 5.1 Celery / 长任务队列（Celery + Redis）

当 `TASK_QUEUE_MODE=celery` 时，以下长任务会由 Celery worker 执行并可跨进程扩展：
- knowledge 文件索引 / 删除
- DeepSearch `run_async`（进度 SSE 支持 `last_event_id` 重放）
- knowledge 导出任务：`/knowledge/graph/export_async`、`/knowledge/mindmap/export_async`

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `TASK_QUEUE_MODE` | `celery` | 切换后台任务模式：`inprocess`（进程内）或 `celery`（分布式 worker）。 |
| `CELERY_BROKER_URL` | _(空)_ | Broker URL（留空则使用 `redis://REDIS_HOST:REDIS_PORT/REDIS_DB`）。 |
| `CELERY_RESULT_BACKEND` | _(空)_ | Result backend（默认同 broker；建议主要用 RedisTaskQueue 的 result key）。 |
| `CELERY_QUEUE_INDEXING` | `indexing` | 索引/删除任务的队列名。 |
| `CELERY_QUEUE_DEEPSEARCH` | `deepsearch` | DeepSearch 队列名。 |
| `CELERY_QUEUE_EXPORT` | `export` | 导出任务队列名（图谱/思维导图）。留空则复用 `CELERY_QUEUE_INDEXING`。 |
| `CELERY_LOGLEVEL` | `info` | Worker 日志级别（`./start.sh` 自动启动 worker 时使用）。 |
| `CELERY_WORKER_CONCURRENCY` | `2` | Worker 并发（进程/线程数；`./start.sh` 自动启动 worker 时使用）。 |
| `CELERY_WORKER_POOL` | `prefork` | Worker pool 实现（`./start.sh` 自动启动 worker 时使用）。 |
| `CELERY_TASK_IGNORE_RESULT` | `true` | 是否忽略 Celery 原生 result backend 写入（长任务建议 `true`）。 |
| `CELERY_RESULT_EXPIRES_SECONDS` | `3600` | Celery result backend 的过期时间（秒）。 |
| `CELERY_TASK_ACKS_LATE` | `true` | 任务结束后再 ack（提高可靠性，但需结合幂等/锁）。 |
| `CELERY_ACKS_ON_FAILURE_OR_TIMEOUT` | `true` | 失败/超时时是否 ack（与 acks_late 配合）。 |
| `CELERY_REJECT_ON_WORKER_LOST` | `true` | worker 丢失时是否让任务重入队。 |
| `CELERY_WORKER_PREFETCH_MULTIPLIER` | `1` | 每 worker 预取任务倍数（长任务建议 `1`）。 |
| `CELERY_TASK_SOFT_TIME_LIMIT_SECONDS` | `0` | Soft time limit（秒，`0` 表示不启用）。 |
| `CELERY_TASK_TIME_LIMIT_SECONDS` | `0` | Hard time limit（秒，`0` 表示不启用）。 |
| `CELERY_VISIBILITY_TIMEOUT_SECONDS` | `86400` | Redis broker visibility timeout（秒；需大于最长任务耗时）。 |
| `MQ_NAMESPACE` | `rag-arc:mq` | RedisTaskQueue 命名空间前缀。 |
| `MQ_TASK_RUN_TTL_SECONDS` | `86400` | TaskRun KV 的 TTL（秒）。 |
| `MQ_PROGRESS_TTL_SECONDS` | `86400` | 进度流（per-run stream/seq_map 等）的 TTL（秒）。 |
| `MQ_RESULT_TTL_SECONDS` | `86400` | 结果 key 的 TTL（秒）。 |
| `MQ_RESULT_MAX_INLINE_BYTES` | `262144` | 结果 JSON 存入 Redis 的最大字节数；超过后自动外置存储（local/MinIO），Redis 仅存引用 envelope（`0` 表示禁用外置）。 |
| `MQ_RESULT_STORE` | `local` | 结果外置存储后端：`local` 或 `minio`。 |
| `MQ_RESULT_LOCAL_DIR` | `local/mq_results` | `local` 外置结果的基础目录。 |
| `MQ_RESULT_MINIO_ENDPOINT` | _(空)_ | `minio` 外置结果的 MinIO endpoint（TODO：尚未实现）。 |
| `MQ_RESULT_MINIO_BUCKET` | _(空)_ | `minio` 外置结果的 bucket（TODO：尚未实现）。 |
| `MQ_STREAM_MAXLEN` | `20000` | Redis Streams 最大长度（近似裁剪）。 |
| `MQ_FAILFAST_ON_REDIS_DOWN` | _(空)_ | Redis 不可用时是否 fail-fast：为空则 `celery` 模式默认 fail-fast，`inprocess` 模式默认 best-effort。 |
| `FILE_OP_LOCK_TTL_SECONDS` | `21600` | 文件操作分布式锁 TTL（秒，索引/删除共用）。 |
| `CELERY_TASK_MAX_RETRIES` | `3` | 任务异常时最大重试次数。 |
| `CELERY_TASK_RETRY_COUNTDOWN_SECONDS` | `5` | 任务异常重试的等待秒数。 |
| `CELERY_TASK_LOCK_MAX_RETRIES` | `30` | 获取 file lock 失败时的最大重试次数。 |
| `CELERY_TASK_LOCK_RETRY_COUNTDOWN_SECONDS` | `2` | 获取 file lock 失败时的重试等待秒数。 |
| `MQ_AUTO_START_WORKERS` | `true` | 当 `TASK_QUEUE_MODE=celery` 时，`./start.sh` 自动启动 `rag-arc-worker-*` 容器。 |
| `MQ_SYNC_TO_POSTGRES_ENABLED` | `true` | `./start.sh` 启动 `rag-arc-mq-sync` 守护进程，将 Redis Streams 归档到 Postgres。 |
| `MQ_SYNC_POLL_INTERVAL_SECONDS` | `2` | 同步轮询间隔（秒，daemon 模式）。 |
| `MQ_SYNC_BATCH_SIZE` | `2000` | 每次同步最多读取的 stream 条目数。 |
| `MQ_SYNC_BLOCK_MS` | `1000` | 每次同步的 Redis XREAD 阻塞时间（毫秒；`0` 表示不阻塞）。 |
| `MQ_STARTUP_HEALTHCHECK` | `true` | `./start.sh` 启动时执行一段 MQ 健康检查（写入小事件、跑一次同步、校验表存在）。 |

### 5.1.1 运行方式（本地/测试）

- 当 `TASK_QUEUE_MODE=celery` 且 `MQ_AUTO_START_WORKERS=true` 时，`./start.sh` 会自动启动 Celery worker 容器。
- 停止 Celery worker：`./stop.sh`（或 `bash scripts/mq_tools/stop_mq_workers_local.sh`）。
- 手动启动（非 Docker）：`bash scripts/mq_tools/start_mq_workers_local.sh`（读取 `.env`，日志输出到 `log/mq_workers/`）。
- 可选：将 Redis Streams 归档到 Postgres：`uv run python scripts/mq_tools/message_queue_sync.py --daemon`（或 `--once` 单次同步）。

## 6. DeepSearch 配置

若无特殊需求，请保留默认值；只有在需要自定义规划器或工具链时才修改。

说明：DeepSearch 工具参数（包括确定性计算工具 `code.python` 的 `allowed_imports`、超时与输出/内存上限）统一通过 `config/json_configs/deepsearch_service.json` → `tool_manager.enabled_tools[...]` 配置，而不是单独的环境变量。

### code.python 工具参数（JSON 配置）

配置位置：`config/json_configs/deepsearch_service.json` → `tool_manager.enabled_tools["code.python"].params`。

| Key | 默认值 | 说明 |
| --- | --- | --- |
| `allowed_imports` | `["json","math","decimal","fractions","statistics","datetime","numpy","pandas","scipy"]` | `code.python` 子进程允许导入的顶级包白名单（不在白名单的 import 会抛 `ImportError`）。 |
| `timeout_seconds` | `6.0` | Python 子进程执行的 wall time 超时（秒）。 |
| `max_code_chars` | `12000` | `extra.code` 最大字符数，超出会直接失败并返回 `code_too_large`。 |
| `max_stdout_chars` | `8000` | stdout 最大捕获字符数（超过会截断）。 |
| `max_stderr_chars` | `8000` | stderr 最大捕获字符数（超过会截断）。 |
| `max_result_chars` | `12000` | `result` 序列化后的 `result_text` 最大字符数（超过会截断）。 |
| `max_memory_mb` | `1024` | 子进程内存上限（尽力：可用时设置 `RLIMIT_AS/RLIMIT_DATA`）。 |
| `emit_result_evidence` | `true` | 是否将执行结果写入一条 evidence（source=`code.python`），供下一轮 `<think>` 通过 `context_evidences` 读取。 |
| `disable_file_io` | `true` | 是否在子进程内阻断 `open()`（默认禁用文件 I/O）。 |

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
| `DEEPSEARCH_ARTIFACT_DIR` | _(空)_ | 可选：DeepSearch 运行 artifacts 根目录（每次 run 会创建 `run_id/` 子目录，写入 plan/reasoning/report/state 等 JSON/Markdown）。 |
| `DEEPSEARCH_TOOL_ARTIFACT_DIR` | `./local/deepsearch_artifacts` | 工具执行日志/产物目录。 |
| `DEEPSEARCH_ALLOW_EXTERNAL_CHANNEL` | `false` | 规划器是否允许生成 `web` 步骤（当未设置 `DEEPSEARCH_EXTERNAL_SEARCH_ENABLED` 时生效）。 |
| `DEEPSEARCH_EXTERNAL_SEARCH_ENABLED` | `false` | 运行时覆盖外部搜索开关（默认由配置 `external_channel.enabled` + `gap_detection.enable_external_on_gap` 决定）。 |
| `DEEPSEARCH_SECTIONWISE_WRITER` | `false` | 启用“分节写作 + Memory Bank 检索 + recency retain_k”模式。 |
| `DEEPSEARCH_BUDGET_TIER` | _(空)_ | 可选的复杂度→预算覆盖开关（`low` / `default`）；为空时将基于问题内容做启发式预算分配。 |
| `DEEPSEARCH_TELEMETRY_ENABLED` | `true` | 是否启用工具运行遥测（本地 artifacts）。 |
| `TAVILY_API_KEY` | _(空)_ | Tavily 搜索的 Key（启用外部搜索时必填）。 |
| `DEEPSEARCH_WEB_PROVIDER` | _(空)_ | 外部搜索路由提示（`tavily` / `tool` / `mcp`；其他值会回退到 `tavily`）。 |
| `DEEPSEARCH_EXTERNAL_CACHE_MODE` | `auto` | 外部搜索录制/回放模式：`off` / `record` / `replay` / `auto`。 |
| `DEEPSEARCH_EXTERNAL_CACHE_DIR` | `./local/deepsearch_artifacts/external_cache` | 外部搜索缓存目录。 |
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
| `DEEPSEARCH_TOOL_CONTEXT_MAX_EVIDENCES` | `5` | 工具调用时传入的 `context_evidences` 最大条数（recency 保留最近 K 条，防止 context 爆炸）。 |
| `DEEPSEARCH_TOOL_CONTEXT_MAX_CHARS` | `800` | 工具调用时每条 evidence 的最大字符数（超出会截断）。 |
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
| `DEEPSEARCH_QUALITY_LOOP_ENABLED` | `false` | 启用“研究 → 质量门槛 → 迭代”闭环（会在报告后进行质量评估并触发补证据/重写）。 |
| `DEEPSEARCH_QUALITY_LOOP_MAX_ROUNDS` | `2` | 质量闭环最多迭代轮数（包含首次）。 |
| `DEEPSEARCH_QUALITY_LOOP_MIN_CITATION_SENTENCE_COVERAGE` | `0.6` | 报告句子中必须包含有效引用的最低比例。 |
| `DEEPSEARCH_QUALITY_LOOP_REQUIRE_CONSISTENCY` | `true` | 当一致性检查出现问题时是否直接判定未通过。 |
| `DEEPSEARCH_QUALITY_LOOP_MAX_UNCITED_SENTENCES` | `6` | 最多输出多少条“缺引用句子”作为修复目标（用于驱动补检索/重写）。 |
| `DEEPSEARCH_QUALITY_LOOP_MAX_ACTIONS` | `6` | 质量门槛最多产出的后续动作数量。 |
| `DEEPSEARCH_QUALITY_LOOP_ENABLE_LLM_JUDGE` | `true` | 启用基于 Rubric 的 LLM Judge（仅在确定性检查失败或存在缺口时调用）。 |
| `DEEPSEARCH_QUALITY_LOOP_JUDGE_TEMPERATURE` | `0.0` | Judge 的 temperature。 |
| `DEEPSEARCH_QUALITY_LOOP_JUDGE_MAX_RETRIES` | `1` | Judge 调用的重试次数。 |
| `DEEPSEARCH_QUALITY_LOOP_TRIGGER_EXTERNAL_ON_FAILURE` | `true` | 允许质量门槛在失败时请求外部搜索动作（仍需外部搜索开关开启）。 |

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
| `JWT_SECRET_KEY` | _(空)_ | JWT 签名秘钥。留空时 API 会自动生成并写入 `RAGARC_RUNTIME_DIR`（默认：`./local/runtime/jwt_secret_key`）；生产环境建议显式配置。 |
| `HF_TOKEN` | _(空)_ | HuggingFace Token（下载受限模型时使用）。 |
| `HF_ENDPOINT` | _(空)_ | 可选：HuggingFace Endpoint 覆盖（例如 `https://hf-mirror.com`）。 |
| `LOG_LEVEL` | `INFO` | 日志等级。 |
| `RAGARC_DEPENDENCY_CHECK_MODE` | `warn` | 应用启动依赖检查模式（Postgres/Redis/Neo4j）：`off`/`warn`/`strict`。注意：当前 API 启动在该变量未设置时默认走 `strict`。 |
| `RAGARC_INDEXING_DEPENDENCY_CHECK_MODE` | `strict` | 知识库索引任务依赖检查模式（`/knowledge/*` 索引与 Celery 任务使用）：`off`/`warn`/`strict`。单测为保持 hermetic 默认设置为 `off`。 |

## 8. 文件/解析路径

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PARSER_OUTPUT_DIR` | `./data/parsed_files` | 统一解析输出目录（native/dots_ocr/vlm_ocr 会落到子目录）。 |
| `NATIVE_PARSER_OUTPUT_DIR` | _(空)_ | 可选：原生解析器输出目录覆盖。 |
| `DOTSOCR_OUTPUT_DIR` | _(空)_ | 可选：dots_ocr 输出目录覆盖。 |
| `VLMOCR_OUTPUT_DIR` | _(空)_ | 可选：VLM OCR 输出目录覆盖。 |
| `OCR_MODEL_NAME` | _(空)_ | 可选：历史兼容的 OCR 模型名别名。 |
| `RAGARC_RUNTIME_DIR` | `./local/runtime` | 当首选目录不可写时的运行时兜底根目录。 |
| `LOCAL_FILE_STORAGE_PATH` | `./data/files` | 当 JSON 未提供 `base_path` 时，`local_blob_store` 的默认根目录（相对路径按项目根目录解析）。 |

## 9. Neo4j 图数据库

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `NEO4J_URL` | `bolt://localhost:7687` | Neo4j 连接字符串。 |
| `NEO4J_USERNAME` | `neo4j` | Neo4j 用户名。 |
| `NEO4J_PASSWORD` | _(空)_ | Neo4j 密码。 |
| `NEO4J_DATABASE` | `neo4j` | 数据库名称。 |
| `EXPOSE_NEO4J` | `false` | 是否开放 Neo4j Browser/Bolt 端口。 |
| `NEO4J_HTTP_PORT` | `7474` | 当 `EXPOSE_NEO4J=true` 时映射到宿主机的 HTTP 端口。 |
| `NEO4J_BOLT_PORT` | `7687` | 当 `EXPOSE_NEO4J=true` 时映射到宿主机的 Bolt 端口。 |

## 10. 可选的 MinIO 对象存储

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `MINIO_USERNAME` | `ROOTNAME` | MinIO 用户名/Access Key（仅在启用 MinIO 集成时使用）。 |
| `MINIO_PASSWORD` | `CHANGEME123` | MinIO 密码/Secret Key。 |

MinIO 常用变量（仅在启用对象存储集成时才需要设置）：
- `MINIO_ENDPOINT`
- `MINIO_BUCKET`
- `MINIO_SECURE`

默认本地/Docker 部署不需要配置这些项。

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
| `RUN_RAGARC_MQ_STRESS_TESTS` | _(空)_ | 可选：设为 `1` 时运行真实 Redis 的消息队列轻压测（`test/stress/test_mq_stress_real_redis.py`）。 |
| `RAGARC_E2E_TOKEN` | _(空)_ | `test/test_complete_e2e_api.py` 用于 API 鉴权的 token。 |

---

**使用建议**：复制 `.env.example` 为 `.env`，填入必需密钥（`OPENAI_API_KEY`/`OPENAI_BASE_URL`（或按模块配置 `*_API_KEY`/`*_API_BASE_URL`）与 `JWT_SECRET_KEY`）即可完成本地部署。高级参数优先在 `config/`（JSON/Python）里改；只有确实需要覆盖时再使用环境变量。
