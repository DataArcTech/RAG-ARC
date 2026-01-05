## MinerU 文件解析服务（独立部署）

本目录提供一个可**独立部署**的 MinerU 多模态文件解析服务（FastAPI Server + Client），用于在 GPU 机器上运行解析，并通过 HTTP/SSH 隧道在本地调用；同时也可以作为 RAG-ARC 的远程解析后端。

- **与 RAG-ARC 运行环境/依赖隔离**：MinerU 的安装与配置请参考上游官方教程，本目录只负责“封装成服务”。
- 适用于“本地算力不足，把解析服务迁移到任意服务器上运行”的场景。

---

## 功能说明

**Server（`mineru_server`）**

- 提供 HTTP API：上传 PDF/图片，调用上游 MinerU 进行解析。
- 输出：Markdown、`content_list` JSON、图片等资源文件。
- 可选：使用 OpenAI 兼容的多模态 Chat 接口为图片生成可检索图注（用于检索/切分/召回更友好）。
- 任务输出落盘：`--output-dir/<task_id>/...`，并提供清单与文件下载接口。

**Client**

- `mineru_client.py`：通用客户端 + CLI（解析/下载主要产物/同步整个任务目录）。
- `mineru_server/client.py` + `mineru_server/cli.py client`：更轻量的“同步任务目录”客户端。

---

## 上游 MinerU 环境（必须先配好）

本服务**不会替你安装/配置 MinerU**。请务必按上游官方教程完成：

- CUDA / 依赖 / 模型 / backend 配置
- 确认在同一个 Python 环境里可以正常 `import mineru` 并运行解析

本服务运行时会调用：
- `mineru.cli.common.do_parse` / `aio_do_parse`
- `mineru.utils.enum_class.MakeMode`

---

## 快速开始

### 1) 启动 Server（建议在 GPU 机器）

在仓库根目录执行：

```bash
python mineru/mineru_main.py server --host 0.0.0.0 --port 8899
```

生产建议（可迁移、可落盘、避免写到仓库目录）：

```bash
python mineru/mineru_main.py server \
  --host 0.0.0.0 --port 8899 \
  --output-dir /data/mineru_outputs \
  --temp-dir /tmp/mineru_temp
```

开启 LLM 图注（仅当 `--caption-mode` 使用 LLM 时需要）：

```bash
export CHAT_API_BASE_URL="https://api.openai.com/v1"
export CHAT_API_KEY="sk-xxx"
export OPENAI_CHAT_MODEL="gpt-4o-mini"  # 必须支持图片输入（多模态）

python mineru/mineru_main.py server \
  --caption-mode content_list_then_llm
```

健康检查：

```bash
curl http://127.0.0.1:8899/health
```

### 2) 本地 Client 调用

配置服务地址（避免硬编码，方便随时迁移服务机器）：

```bash
export MINERU_SERVER_URL="http://<server-ip>:8899"
```

解析并下载产物：

```bash
python mineru/mineru_client.py parse \
  --file /path/to/demo.pdf \
  --output-dir ./mineru_client_outputs
```

多文件压测：

```bash
python mineru/mineru_client.py bench \
  --files /path/a.pdf /path/b.pdf \
  --concurrency 2 \
  --output-dir ./mineru_client_outputs \
  --no-download
```

---

## RAG-ARC 集成方式

RAG-ARC 可通过 `.env` 把 PDF/图片解析切到该服务：

- 设置 `PARSER_PARSE_MODE=mineru`
- 设置 `MINERU_SERVER_URL=http://<server-ip>:8899`（可选 `MINERU_TIMEOUT_S=900`）

解析产物会被镜像到 `PARSER_OUTPUT_DIR`（默认 `./data/parsed_files/mineru/...`），与 RAG-ARC 原有解析输出目录结构保持一致。

## 通过 SSH 隧道调用（推荐）

当 GPU 机器不对外开端口时：

```bash
ssh -L 8899:127.0.0.1:8899 <user>@<gpu-host>
export MINERU_SERVER_URL="http://127.0.0.1:8899"
python mineru/mineru_client.py parse --file demo.pdf --output-dir ./mineru_client_outputs
```

---

## Server HTTP API 说明

Base URL：`http://<host>:<port>`

### 接口列表

- `GET /health`：健康检查 + 当前配置摘要。
- `GET /config`：返回配置（敏感字段会脱敏）。
- `POST /parse`：单文件解析（multipart 上传）。
- `POST /parse/batch`：多文件解析（multipart 上传）。
- `GET /task/{task_id}/manifest`：列出 `output_dir/<task_id>/...` 下的文件清单。
- `GET /task/{task_id}/file/{rel_path}`：按相对路径下载文件（推荐，避免同名冲突）。
- `GET /download/{task_id}/{filename}`：按文件名搜索下载（历史接口，若目录下存在重名文件可能有歧义）。

### `POST /parse` 表单参数

- `backend`：默认从服务配置读取（推荐 `vlm-transformers`）。
- `parse_method`：默认 `auto`。
- `lang`：默认 `ch`。
- `formula_enable`：`true|false`。
- `table_enable`：`true|false`。
- `start_page`：起始页（0-based）。
- `end_page`：结束页（包含，选填）。
- `output_format`：`mm_md | md_only | content_list`。

返回体包含：
- `task_id`, `status`, `processing_time`
- 各类产物的绝对路径（`*_path`）与**任务相对路径**（`*_rel_path`，用于稳定下载）
- `images_metadata`（每张图包含 `task_rel_path`，可直接走 `/task/.../file/...` 下载）

---

## CLI 使用说明

### Server CLI

```bash
python mineru/mineru_main.py server --help
```

常用参数：
- 网络：`--host`, `--port`, `--workers`, `--max-jobs`
- 存储：`--output-dir`, `--temp-dir`, `--modelscope-cache-dir`, `--hf-home`, `--mineru-home`
- MinerU 默认参数：`--backend`, `--parse-method`, `--lang`, `--no-formula`, `--no-table`, `--model-source`, `--device`
- vLLM（仅 vLLM backend 生效）：`--vllm-gpu-mem-util`, `--vllm-enforce-eager`, `--vllm-max-model-len`, `--vllm-swap-space-gb`, `--vllm-cpu-offload-gb`
- 图注：`--caption-mode`, `--chat-api-base-url`, `--chat-api-key`, `--chat-api-key-file`, `--chat-model`, `--caption-context`, `--caption-context-file`, `--up`, `--down`

### Client CLI

```bash
python mineru/mineru_client.py --help
```

---

## 代码调用（Python）

### 最小：解析 + 同步整个任务目录

```python
from pathlib import Path
from mineru_server.client import MinerUServerClient

client = MinerUServerClient(base_url="http://127.0.0.1:8899", timeout_s=900)
result = client.parse(
    file_path=Path("demo.pdf"),
    backend="vlm-transformers",
    parse_method="auto",
    lang="ch",
    formula_enable=True,
    table_enable=True,
    start_page=0,
    end_page=None,
    output_format="mm_md",
)
task_root = client.sync_task(result["task_id"], Path("./mineru_client_outputs"))
print(task_root)
```

### 下载主要产物（md/json/images）

```python
from pathlib import Path
from mineru_client import MinerUClient

client = MinerUClient(base_url="http://127.0.0.1:8899", timeout=900)
parse_result = client.parse_file(Path("demo.pdf"))
downloaded = client.download_artifacts(parse_result, Path("./mineru_client_outputs"))
print(downloaded["document_dir"])
```

---

## 环境变量说明

### Client

- `MINERU_SERVER_URL`：客户端默认服务地址（例如 `http://127.0.0.1:8899`）。

### Server 的 `.env` 加载机制

- `MINERU_DOTENV_PATH`：指定 `.env` 文件绝对/相对路径（优先级最高）。

不指定时，Server 默认按以下顺序加载：
1) `mineru/.env`（推荐：服务独立配置）
2) `<cwd>/.env`

### LLM 图注（仅 `--caption-mode` 使用 LLM 时需要）

- `CHAT_API_BASE_URL`：OpenAI 兼容 base url（例如 `https://api.openai.com/v1`）。
- `CHAT_API_KEY`：API Key（多 worker 建议用 `CHAT_API_KEY_FILE`）。
- `CHAT_API_KEY_FILE`：包含 API Key 的文件路径。
- `OPENAI_CHAT_MODEL`：模型名（必须支持图片输入）。

可选：
- `CAPTION_CONTEXT`：追加到图注 prompt 前的固定上下文。
- `CAPTION_CONTEXT_FILE`：从文件读取固定上下文。
- `CAPTION_UP`：图片上文 token 窗口（默认 `500`）。
- `CAPTION_DOWN`：图片下文 token 窗口（默认 `500`）。

### 缓存目录（建议显式配置，便于迁移与磁盘管理）

- `MODELSCOPE_CACHE`：ModelScope 缓存目录（默认 `~/.cache/modelscope/hub`）。
- `HF_HOME`：HuggingFace 缓存目录（默认 `~/.cache/huggingface`）。
- `XDG_CACHE_HOME`：会影响默认缓存根目录。

注意：Server 会根据自身配置设置上游 MinerU 的运行环境变量（例如 `MINERU_DEVICE_MODE`, `MINERU_OUTPUT_DIR`, `HF_HOME` 等），以保证可复现与可迁移。

---

## 输出目录结构

Server 输出根目录：`--output-dir`（默认：`mineru/mineru_outputs`）

每次请求输出：

```
<output-dir>/<task_id>/<doc_name>/<method_dir>/
  <doc_name>.md
  <doc_name>_content_list.json
  images/
```

Client 可把整个任务目录同步到：

```
<client-output-dir>/mineru_outputs/<task_id>/...
```

---

## 可迁移部署建议（随时搬到任意机器）

- 生产环境不要依赖仓库相对路径：显式传入绝对 `--output-dir`、`--temp-dir`。
- Client 统一通过 `MINERU_SERVER_URL` 配置服务地址，避免写死 IP。
- 多 worker 场景建议使用 `CHAT_API_KEY_FILE` 管理密钥（避免把 key 写进配置文件/历史记录）。
- `--workers > 1` 时会在 `mineru/.tmp/` 写入 config JSON 供 worker 进程读取，请保证该目录可写且权限受控。
