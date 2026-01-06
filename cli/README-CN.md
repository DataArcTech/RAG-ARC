# RAG-ARC CLI 使用指南

CLI 提供从“文档接入 → 索引/建图 → 检索问答”的完整算法链路，无需启动 FastAPI。所有命令通过 `uv run rag-arc ...` 执行，并复用 `.env` 以及既有依赖（PostgreSQL、Redis、Neo4j、MinIO 或本地存储）。

## 前置条件
1. 执行 `uv sync` 安装依赖，并在 `.env` 中设置 `DEVELOP_MODE=true`（或分别开启 `EXPOSE_POSTGRES/EXPOSE_REDIS/EXPOSE_NEO4J`），以便 Docker 服务暴露到本地端口供 CLI 访问；同时 CLI 会默认使用 `DEVELOP_OWNER_ID`（可在 `.env` 配置）。
2. 在宿主机安装 libpq 依赖，Debian/Ubuntu 可运行 `sudo apt install -y libpq5 libpq-dev`，否则 `psycopg` 无法连接 PostgreSQL。
3. 与 Web 模式相同方式启动 PostgreSQL/Redis/Neo4j（Docker 脚本或本地进程），确认暴露端口与 `.env` 一致。
4. 开启 `DEVELOP_MODE` 时 CLI 会自动创建一个占位用户（用户名/密码来自 `DEVELOP_OWNER_USERNAME`/`DEVELOP_OWNER_PASSWORD`），仅用于算法测试，用户/权限功能会被跳过。
5. 如需切换配置，可提前设置 `MODEL_PROFILE` 及各类 provider 环境变量。

## 常用命令

| 类别 | 命令 | 说明 |
| --- | --- | --- |
| 文档导入 | `uv run rag-arc ingest-file ./doc.pdf --owner-id <UUID>` | 上传并完成切分、索引、建图（单文件，推荐始终指定 `--owner-id`）。 |
| 文档导入 | `uv run rag-arc ingest-folder ./docs --pattern '*.pdf' --owner-id <UUID>` | 按文件夹批量导入，默认递归子目录。 |
| 知识管理 | `uv run rag-arc list-files --json --owner-id <UUID>` | 列出当前 Owner 下的文件，可按状态/分页过滤。 |
| 知识管理 | `uv run rag-arc delete-file FILE_ID --owner-id <UUID>` | 仅标记删除（元数据操作，不会清理索引/存储）。 |
| 知识管理 | `uv run rag-arc trigger-index FILE_ID [FILE_ID ...] --owner-id <UUID>` | 对既有文件重新触发索引/建图。 |
| 图工具 | `uv run rag-arc export-graph --output graph.json --owner-id <UUID>` | 导出完整图谱（Neo4j/igraph）到终端或 JSON 文件。 |
| 检索问答 | `uv run rag-arc chat "什么是RAG-ARC？" --owner-id <UUID>` | 多路径检索 + 重排 + LLM 的完整对话。 |
| 检索问答 | `uv run rag-arc pipeline "什么是RAG-ARC？" --skip-llm --subgraph --owner-id <UUID>` | 仅查看改写/检索/重排（可导出子图）。 |
| 图问答 | `uv run rag-arc graph-qa "X和Y之间有什么关系?" --owner-id <UUID>` | 仅走图检索链路，并返回子图元数据。 |
| DeepSearch | `uv run rag-arc deepsearch "请分析 Singapore American School..." --with-evidence --json` | 执行 Graph DeepSearch 流程。`--with-evidence` 返回 chunk/seed/triple，`--json` 将裁剪后的报告写到 `local/cli/<owner>/`（如需完整原始结果再加上 `--save-raw`）。 |
| MCP | `uv run rag-arc tool-mcp-server --transport stdio` | 启动 DeepSearch 工具 MCP 服务器（配置位于 `config/json_configs/deepsearch_tool_mcp_server.json`，SSE 默认端口 8765，默认路径 `/mcp/tools`）。 |
| MCP | `uv run rag-arc chat-mcp-server --transport stdio` | 启动聊天/鉴权 MCP 服务器（实现见 `api/mcp/server.py`，SSE/HTTP 默认 `127.0.0.1:8785/mcp/chat`）。 |

> 注意：CLI 的 `delete-file` 命令定位为轻量测试，仅更新文件状态/元数据，不会执行耗时的索引、向量、图谱和 Blob 清理。如需完整的后台删除流程，请调用 HTTP API `DELETE /knowledge/{file_id}`。

强烈建议每次运行都显式传入 `--owner-id <UUID>`，以便复用同一租户/用户的数据。未指定时 CLI 会优先使用环境变量或缓存的默认 UUID，行为可能因机器而异。

> ℹ️ ToolManager 默认在 CLI/HTTP 进程内执行全部内建 DeepSearch 工具；只有当你在配置里提供 `mcp_client`、为某个工具开启 `mcp_only`/`mcp_fallback`，或通过 `remote_tools` 注册完全远程的描述符时，才需要事先启动 MCP 工具服务器。

## 使用提示
- `--json` 适用于 list/chat/pipeline/graph-qa/export-graph，方便获得结构化输出。
- `ingest-folder` 支持 `--limit`、`--pattern`、`--no-recursive` 控制导入范围，每个文件失败会即时打印原因。
- `trigger-index` 与 `export-graph` 直接操作配置文件中声明的图存储（默认 Neo4j）。运行前请确认相关服务处于可访问状态。
- 默认的 owner ID 会缓存到 `~/.rag_arc_owner_id`，若需多人共享或固定某个租户，可使用 `--owner-id ...`，或在环境变量中设置 `CLI_OWNER_ID` / `RAG_ARC_OWNER_ID` / `DEFAULT_OWNER_ID`。
- 终端中展示的 chunk 预览默认截取前 50 个字符；如需查看完整内容请使用 `--json` 输出。
- `chat`/`pipeline`/`graph-qa` 提供 `--with-evidence` 选项，可让终端/JSON 输出附带与问题相关的 chunk、图三元组以及种子实体信息（该选项会自动开启 `--subgraph` 以导出 HippoRAG 子图）。
- DeepSearch 命令依赖 `deepsearch_service` 配置（见 `config/json_configs/deepsearch_service.json`），请确保 CLI/服务端在启动时成功注册该模块。
- 可通过 `.env` 中的证据控制变量（`CHAT_TOP_CHUNKS`、`CHAT_TOP_TRIPLES`、`CHAT_TOP_SEED_ENTITIES`、`DEEPSEARCH_TOP_CHUNKS`、`DEEPSEARCH_TOP_TRIPLES`、`DEEPSEARCH_TOP_SEED_ENTITIES`、`DEEPSEARCH_GRAPH_NODE_LIMIT`）调节输出规模；设置 `ENABLE_ALL_EVIDENCE=true` 可关闭所有裁剪。
- DeepSearch `--json` 会同时生成裁剪后的报告和 `_raw` 备份，均保存在 `local/cli/<owner>/`，方便日后排查。
