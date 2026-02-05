## MinerU 文件解析服务（独立部署）

本目录提供一个独立的 FastAPI 服务，封装上游 MinerU 解析能力。
可部署在 GPU 机器上，通过 `MINERU_SERVER_URL` 或 SSH 隧道远程调用。

---

## 功能概述

**Server（`mineru_server`）**
- HTTP API：解析 PDF/图片。
- 输出 Markdown、content_list JSON、图片等资源文件。
- 可选：调用 OpenAI 兼容多模态接口生成图片图注。
- 任务产物落盘到 `--output-dir/<task_id>/...` 并提供下载接口。

**Client**
- `mineru_client.py`：简单 CLI（解析 / 下载 / 同步任务目录）。
- `mineru_server/client.py` + `mineru_server/cli.py client`：轻量同步工具。

---

## 前置条件（上游 MinerU 环境）

本服务**不会安装或配置 MinerU**，请先按上游文档准备环境（CUDA、模型、后端等）。
确保同一 Python 环境中 `import mineru` 可用。

---

## 快速开始

### 1) 启动服务（建议在 GPU 机器）

```bash
python mineru_main.py server --host 0.0.0.0 --port 8899
```

生产建议路径：

```bash
python mineru_main.py server \
  --host 0.0.0.0 --port 8899 \
  --output-dir /data/mineru_outputs \
  --temp-dir /tmp/mineru_temp
```

开启 LLM 图注：

```bash
export CHAT_API_BASE_URL="https://api.openai.com/v1"
export CHAT_API_KEY="sk-xxx"
export OPENAI_CHAT_MODEL="gpt-4o-mini"

python mineru_main.py server \
  --caption-mode content_list_then_llm
```

健康检查：

```bash
curl http://127.0.0.1:8899/health
```

### 2) 客户端调用

```bash
export MINERU_SERVER_URL="http://<server-ip>:8899"

python mineru_main.py client \
  --base-url "$MINERU_SERVER_URL" \
  --file /path/to/demo.pdf \
  --output-dir ./mineru_client_outputs
```

### 3) Python 调用（import client）

Python 里调用服务有两种常见方式：
- 直接用 HTTP（`requests`）；
- 或者 import 本仓库里自带的轻量 client（需要把 `` 加到 `PYTHONPATH`）。

示例：

```python
from pathlib import Path
from mineru_server.client import MinerUServerClient

client = MinerUServerClient(base_url="http://127.0.0.1:8899", timeout_s=3600)
result = client.parse(
    file_path=Path("demo/pdfs/demo3.pdf"),
    backend="vlm-transformers",
    parse_method="auto",
    lang="ch",
    formula_enable=True,
    table_enable=True,
    start_page=0,
    end_page=None,
    output_format="mm_md",
    wait=True,
)
print(result["status"], result.get("processing_time"))
```

---

## 外部 vLLM 多卡加速

**说明：**本服务内置 vLLM 仅单卡。若要多卡加速，需要启动外部 vLLM（OpenAI 兼容）并让本服务走 `vlm-http-client`。

### A) 启动 vLLM 服务

推荐脚本：

```bash
./scripts/start_vllm.sh --gpus 4,6 --tp 2 --port 30000
```

脚本常用参数：

```bash
# 指定模型路径
./scripts/start_vllm.sh --model-path /path/to/vlm_model --gpus 0,1 --tp 2 --port 30000

# 使用 HuggingFace 缓存的 MinerU2.5
./scripts/start_vllm.sh --model-key mineru2.5 --model-source hf --gpus 0,1 --tp 2 --port 30000

# 单卡
./scripts/start_vllm.sh --gpus 0 --tp 1 --port 30000

# 透传 vLLM 额外参数（放在 -- 后）
./scripts/start_vllm.sh --gpus 0,1 --tp 2 --port 30000 -- --max-model-len 8192
```

等价手动命令：

```bash
conda activate mineru

CUDA_VISIBLE_DEVICES=4,6 \
vllm serve /home/dataarc/.cache/modelscope/hub/models/OpenDataLab/MinerU2.5-2509-1.2B \
  --served-model-name MinerU2.5-2509-1.2B \
  --port 30000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8
```

注意：
- `--tensor-parallel-size` 必须与 GPU 数量一致。
- vLLM 预分配显存；如 OOM 请调低 `--gpu-memory-utilization`。
- vLLM 仅用于 **VLM 模型**；OCR 模型（如 PaddleOCR）**不走 vLLM**。

### 停止 vLLM / MinerU（并释放显存）

如果你用 `start_vllm_pool.sh` 启动 vLLM，每个实例会写：
- pid：`/tmp/vllm_<port>.pid`
- log：`/tmp/vllm_<port>.log`

推荐用停止脚本：

```bash
# 停掉指定端口的 vLLM
./scripts/stop_mineru_vllm.sh --vllm-ports 30000,30001 --clean

# 停掉指定端口的 MinerU server
./scripts/stop_mineru_vllm.sh --mineru-ports 8897,8898,8899 --clean
```

如果你已经通过 `lsof -i:<port>` 找到 LISTEN 进程并 `kill -9`，但显存仍然被占用，
通常是 `VLLM::EngineCore` 子进程还活着。此时按 vLLM 用到的 GPU 强制清掉 EngineCore：

```bash
# 示例：vLLM 用的是 GPU 5 和 6
./scripts/stop_mineru_vllm.sh --kill-enginecore-gpus 5,6
```

### B) 无需改客户端（推荐）

启动 vLLM 后，用 `vlm-http-client` + `--server-url` 启动本服务：

```bash
python mineru_main.py server \
  --backend vlm-http-client \
  --server-url http://127.0.0.1:30000
```

已有客户端仍然调用 `/parse`，无需改代码。

### C) 网关 + 服务池（推荐提高吞吐）

如果你启动了**多个** vLLM 实例（不同 GPU/端口），可以只启动**一个** MinerU server 作为网关，
并配置 `--server-urls` 地址池：

先启动 vLLM 池（默认每张卡一个 vLLM 实例）：

```bash
conda activate mineru
./scripts/start_vllm_pool.sh --gpus 4,6 --base-port 30000
```

再启动 MinerU 网关：

```bash
python mineru_main.py server \
  --backend vlm-http-client \
  --server-urls http://127.0.0.1:30000,http://127.0.0.1:30001 \
  --max-jobs 2 \
  --max-pending 16
```

网关会按“当前 in-flight 数量最少”的策略选择后端；当某个后端报错时，会进入短暂冷却期并被暂时避开。

**TP 与池模式的区别（很重要）：**
- `start_vllm.sh --gpus 4,6 --tp 2`：启动 *一个* 模型实例，切到两张卡上；**单个请求会同时用两张卡**。
- `start_vllm_pool.sh --gpus 4,6`：启动 *两个* 模型实例（每个 tp=1）；并发请求可被路由到**不同 GPU**。

---

## 命令行参数（完整）

### Server CLI

```bash
python mineru_main.py server --help
```

**网络与容量**
- `--host`（默认 `0.0.0.0`）
- `--port`（默认 `8899`）
- `--workers`（默认 `1`）
- `--max-jobs`（默认 `1`）：单 worker 最大并发
- `--max-pending`（默认 `0`）：排队+运行的总上限，`0` 表示无限
  - 环境变量：`MINERU_SERVER_MAX_PENDING_JOBS`
- `--enforce-backend`（默认 `false`）：忽略请求里传入的 `backend`，始终使用服务端默认 `--backend`（网关模式）
  - 环境变量：`MINERU_SERVER_ENFORCE_BACKEND=1`

**MinerU 默认参数**
- `--backend`（默认 `vlm-transformers`）
- `--parse-method`（默认 `auto`）
- `--lang`（默认 `ch`）
- `--formula-enable` / `--no-formula`（默认开启）
- `--table-enable` / `--no-table`（默认开启）

**外部 VLM 服务（http-client 后端）**
- `--server-url`：OpenAI 兼容服务地址
  - 环境变量：`MINERU_VLM_SERVER_URL`
  - 仅在 `vlm-http-client` / `hybrid-http-client` 生效
- `--server-urls`：OpenAI 兼容服务地址池（网关模式；按请求路由）
  - 可重复传入
  - 环境变量：`MINERU_VLM_SERVER_URLS`

**设备与模型来源**
- `--model-source`（默认 `modelscope`）
- `--device`（默认 `cuda`）
- `--virtual-vram-gb`（默认 `None`）

**vLLM 参数（仅进程内 vLLM）**
- `--vllm-gpu-mem-util`（默认 `0.5`）
- `--vllm-enforce-eager`（默认 `false`）
- `--vllm-max-model-len`（默认 `None`）
- `--vllm-swap-space-gb`（默认 `4.0`）
- `--vllm-cpu-offload-gb`（默认 `0.0`）

**存储路径**
- `--output-dir`（默认 `mineru_outputs`）
- `--temp-dir`（默认 `.tmp/mineru_temp`）
- `--modelscope-cache-dir`（默认 `~/.cache/modelscope/hub`）
- `--hf-home`（默认 `~/.cache/huggingface`）
- `--mineru-home`（默认 `..`）

**图注（LLM caption）**
- `--caption-mode`（默认 `content_list_then_llm`）
- `--chat-api-base-url`（env: `CHAT_API_BASE_URL`）
- `--chat-api-key`（env: `CHAT_API_KEY`）
- `--chat-api-key-file`（env: `CHAT_API_KEY_FILE`）
- `--chat-model`（env: `OPENAI_CHAT_MODEL`）
- `--chat-timeout-s`（默认 `60`）
- `--caption-max-images`（默认 `0`，<=0 表示不限）
- `--caption-context`（env: `CAPTION_CONTEXT`）
- `--caption-context-file`（env: `CAPTION_CONTEXT_FILE`）
- `--up`（默认 `500`）
- `--down`（默认 `500`）

### Client CLI

```bash
python mineru_main.py client --help
```

- `--base-url`（默认 `MINERU_SERVER_URL` 或 `http://127.0.0.1:8899`）
- `--file`（必填）
- `--output-dir`（默认 `./mineru_client_outputs`）
- `--backend`（默认 `vlm-transformers`）
- `--parse-method`（默认 `auto`）
- `--lang`（默认 `ch`）
- `--formula-enable` / `--no-formula`
- `--table-enable` / `--no-table`
- `--start-page`（默认 `0`）
- `--end-page`（默认 `None`）
- `--output-format`（默认 `mm_md`，可选 `mm_md|md_only|content_list`）
- `--timeout`（默认 `900`）

---

## HTTP API（Server）

Base URL：`http://<host>:<port>`

Endpoints：
- `GET /health`
- `GET /config`
- `POST /parse`
- `GET /parse/status/{task_id}`
- `POST /parse/batch`
- `GET /task/{task_id}/manifest`
- `GET /task/{task_id}/file/{rel_path}`
- `GET /download/{task_id}/{filename}`（历史接口）

`POST /parse` 表单参数：
- `backend`, `parse_method`, `lang`, `formula_enable`, `table_enable`
- `start_page`, `end_page`, `output_format`
- `wait`（true 表示阻塞等待）

---

## SSH 隧道（远程 GPU）

```bash
ssh -CNg -L 8899:127.0.0.1:8899 <user>@<gpu-host>
export MINERU_SERVER_URL="http://127.0.0.1:8899"
python mineru_main.py client --base-url "$MINERU_SERVER_URL" --file demo.pdf --output-dir ./mineru_client_outputs
```

---

## RAG-ARC 集成

- `PARSER_PARSE_MODE=mineru`
- `MINERU_SERVER_URL=http://<server-ip>:8899`（可选 `MINERU_TIMEOUT_S=900`）
- 可选页范围：`MINERU_START_PAGE=0`, `MINERU_END_PAGE=1`

解析产物会落盘到 `PARSER_OUTPUT_DIR`（默认 `./data/parsed_files/mineru/<file_id>/...`）。

### 证据资源 URL（RAG-ARC API）
- `document_url`: `GET /knowledge/{file_id}/download`
- Markdown 图片链接 `images/...` 会被改写为 `GET /knowledge/{file_id}/mineru-assets/images/...`
