# 配置说明

RAG-ARC 使用“环境变量 + JSON 配置文件”的单一事实来源（Single Source of Truth）：

- **运行时密钥 / 部署相关配置**：环境变量（`.env`，参考 `.env.example`）。
- **可调参数（阈值/预算/工具选择/路径/特性开关）**：`config/json_configs/` 下的 JSON。
- LLM JSON 输出的全局重试默认：`config/core/llm_json_retry_defaults.py`（支持 env 覆盖，详见 `config/env-*.md`）。

## 入口配置

- DeepSearch service：`config/json_configs/deepsearch_service.json`
- RAG inference（HippoRAG Q&A）：`config/json_configs/rag_inference.json`
- Knowledge pipelines：`config/json_configs/knowledge.json`

`MODEL_PROFILE`（见 `.env.example`）控制默认加载的配置 profile（例如 `api` vs `local`）。

## 环境变量文档

- English：`config/env-en.md`
- 中文：`config/env-zh.md`

补充说明：
- 当 `EMBEDDING_MODEL_PROVIDER=openai` 时，`OPENAI_EMBEDDING_MODEL` 优先级高于 `EMBEDDING_MODEL_NAME`。
- DeepSearch 的 web search 需要 `TAVILY_API_KEY`，并且在 `config/json_configs/deepsearch_service.json` 里启用外部检索通道。

## 虚拟路径（`io://...`）与本地路径

许多“路径类”配置支持两种写法：
- `io://...` 虚拟路径（推荐：更可移植；通过 IOManager 映射到 LocalDB/MinIO），或
- 本地文件系统路径（便于单测隔离、一次性脚本/调试）。

常见例子（以 `config/env-*.md` 为准）：
- 解析产物目录：`PARSER_OUTPUT_DIR`、`NATIVE_PARSER_OUTPUT_DIR`、`MINERU_SHARED_CACHE_DIR`
- DeepSearch artifacts：`DEEPSEARCH_TOOL_ARTIFACT_DIR`
- MQ 结果外置：`MQ_RESULT_STORE`、`MQ_RESULT_LOCAL_DIR`（如需完全避免 IOManager 依赖，可设 `MQ_RESULT_STORE=local`）

