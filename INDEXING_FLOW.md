# 索引（Ingest + Index）流程说明

本文描述 RAG-ARC 在本项目中的「文档导入（ingest）」与「索引（index）」全链路：从本地文件/上传文件进入系统，到解析、切分、向量/关键词索引、图谱索引完成，并给出常用 CLI/API 命令与排障要点。

## 1. 关键概念

- **owner_id**：租户/用户隔离的主键（UUID）。同一个 `owner_id` 下的文件、chunk、索引与图谱数据相互可见；不同 `owner_id` 互相隔离。
- **FileStatus**：文件处理状态（见 `encapsulation/data_model/orm_models.py`）：
  - `STORED`：文件元数据写入 + blob 已存储
  - `PARSED` / `CHUNKED`：中间态（解析/切分完成）
  - `INDEXED`：所有 indexer 执行完成
  - `FAILED`：流程失败
  - `DELETED`：软删除（仅元数据标记）

## 2. 依赖与环境准备

索引流程会用到以下依赖（取决于配置）：

- PostgreSQL：存储用户、文件元数据、解析结果元数据、chunk 元数据等
- Redis：聊天/会话缓存（索引本身不一定必须，但本项目常见启动会初始化）
- Neo4j：图谱索引（HippoRAG/PrunedHippoRAG）
- LLM/Embedding/OCR Provider：解析 OCR、抽取三元组、生成向量（可能触发外部调用与限流）

推荐命令（项目根目录执行）：

```bash
uv sync
uv sync --extra dev
```

启动基础设施（Docker）：

```bash
./build.sh
./start.sh
```

`.env` 里至少确保数据库/Neo4j/Redis 地址可达，并配置模型 provider（不要提交密钥；生产用 `.env` 管理）。

## 3. owner_id（共享/固定租户）约定

- CLI 中**强烈建议**每次都显式传 `--owner-id <UUID>`，避免落到随机/缓存的 owner，导致“看不到刚导入的文件”。
- 本项目提供一个共享 owner，用于 chatbot 与统一共享文档：
  - `.env`：`CHATBOT_SHARED_DOCUMENT_OWNER_ID=<UUID>`

## 4. ingest-folder / ingest-file：端到端流程

以 CLI 为例（见 `cli/rag.py`）：

### 4.1 上传（store）

1. 读取文件 bytes
2. `FileStorage.upload_file(...)` 执行：
   - 归一化文件名（会带项目相对路径前缀，例如 `RAG-ARC/local/files/...`）
   - 在同一 `owner_id` 下做重复检测：
     - 同名文件：拒绝
     - 同内容（content_hash）：拒绝
   - 写入 PostgreSQL：`file_metadata`（`STORED`）
   - 写入 blob store（默认本地文件系统路径，`LOCAL_FILE_STORAGE_PATH`）

### 4.2 解析（parse）

由 `Knowledge.file_index.index_file(file_id)` 驱动：

- PDF/图片类可能走 OCR（由 `.env` 与 `config/json_configs/knowledge*.json` 控制）
- 输出解析结果（文本、结构化片段）并落库/落盘

### 4.3 切分（chunk）

将解析后的内容按 chunker 配置切分（例如 token chunker：`chunk_size`/`chunk_overlap`），并持久化 chunk 元数据。

### 4.4 索引（index）

按 `config/json_configs/knowledge.json` 中的 `indexer_configs` 顺序执行，常见包含：

- **图谱索引**（Neo4j）：抽取实体/关系（可能调用 LLM），写入图数据库并生成/更新图向量索引
- **向量索引**（Faiss）：对 chunk 生成 embedding，写入本地 faiss index（`./data/unified_faiss_index/`）
- **关键词索引**（BM25）：对 chunk 建立 BM25 索引（本地路径由配置决定）

完成后文件状态更新为 `INDEXED`。

## 5. 常用 CLI 命令

### 5.1 批量导入本地目录

```bash
uv run rag-arc ingest-folder local/files --pattern '*.pdf' --owner-id <UUID>
```

### 5.2 导入单文件

```bash
uv run rag-arc ingest-file ./path/to/doc.pdf --owner-id <UUID>
```

### 5.3 查看文件列表/状态

```bash
uv run rag-arc list-files --json --owner-id <UUID>
```

### 5.4 触发重新索引（仅 STORED/FAILED 会被调度）

```bash
uv run rag-arc trigger-index <FILE_ID> --owner-id <UUID>
```

## 6. HTTP API（如启用 FastAPI）

API 的上传/索引路径与 CLI 的核心模块共用（`application/knowledge/module.py` / `core/file_management/*`），区别在于：

- API 侧会走鉴权/权限校验与异步调度（视具体路由实现）
- `DELETE /knowledge/{file_id}` 通常只做元数据标记，完整清理需要异步/后台任务（取决于实现）

## 7. 常见排障

### 7.1 “Upload failed: File metadata ... already exists”

这通常意味着写入 `file_metadata` 时主键冲突（极少见）或数据库里存在残留/并发写入问题。若你遇到稳定复现：

- 确认目标 `owner_id` 对应的 `user` 记录存在（`file_metadata.owner_id` 外键依赖）
- 检查数据库是否正确连到预期实例（`.env` 的 `POSTGRES_HOST/PORT/DB`）

### 7.2 “File with name ... already exists / same content already exists”

同一 `owner_id` 下的重复导入会被拒绝：

- 同名：直接拒绝
- 同内容：基于归一化 `content_hash` 判定（避免重复占用索引空间）

### 7.3 embedding/LLM 限流（HTTP 429）

索引图谱与向量会调用 embedding/LLM，可能触发 provider 限流：

- 等待后重试（或降低并发/批大小：见 `knowledge.json` 中 extractor 的并发与 batch 配置）
- 或更换/升级配额

### 7.4 Faiss/BM25/Graph 路径提示未设置

`framework.register` 打印 “Environment variable ... is not set” 时通常表示配置里引用了 `${VAR}`，但 `.env` 未提供：

- 部分变量有默认值（例如本地 blob store 默认 `./data/files`）
- 但索引落盘路径建议显式设置，以便在不同部署环境保持一致

