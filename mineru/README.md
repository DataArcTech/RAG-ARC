## MinerU Parsing Service (Standalone)

This directory provides a standalone FastAPI service that wraps the upstream MinerU parser.
It can run on a GPU machine and be accessed remotely via `MINERU_SERVER_URL` or an SSH tunnel.

---

## What It Does

**Server (`mineru_server`)**
- HTTP API to parse PDF/images using upstream MinerU.
- Outputs Markdown, content_list JSON, and extracted assets (images).
- Optional image captioning via an OpenAI-compatible multimodal chat endpoint.
- Stores task outputs under `--output-dir/<task_id>/...` and provides download endpoints.

**Clients**
- `mineru_client.py`: simple HTTP client + CLI (parse, download artifacts, sync full task directory).
- `mineru_server/client.py` + `mineru_server/cli.py client`: minimal sync helper.

---

## Prerequisites (Upstream MinerU)

This service **does not install or configure upstream MinerU**. Please follow the official upstream tutorial
(CUDA, models, backends, etc.). Once `import mineru` works in the same Python environment, this service can call:
- `mineru.cli.common.do_parse` / `aio_do_parse`
- `mineru.utils.enum_class.MakeMode`

---

## Quick Start

### 1) Start the Server (GPU machine)

```bash
python mineru_main.py server --host 0.0.0.0 --port 8899
```

Recommended production paths:

```bash
python mineru_main.py server \
  --host 0.0.0.0 --port 8899 \
  --output-dir /data/mineru_outputs \
  --temp-dir /tmp/mineru_temp
```

If you enable LLM captioning:

```bash
export CHAT_API_BASE_URL="https://api.openai.com/v1"
export CHAT_API_KEY="sk-xxx"
export OPENAI_CHAT_MODEL="gpt-4o-mini"

python mineru_main.py server \
  --caption-mode content_list_then_llm
```

Health check:

```bash
curl http://127.0.0.1:8899/health
```

### 2) Client (run anywhere)

```bash
export MINERU_SERVER_URL="http://<server-ip>:8899"

python mineru_main.py client \
  --base-url "$MINERU_SERVER_URL" \
  --file /path/to/demo.pdf \
  --output-dir ./mineru_client_outputs
```

### 3) Python Usage (import client)

If you want to call the server from Python, you can either:
- use plain HTTP (`requests`), or
- import the lightweight client in this repo (make sure `` is on `PYTHONPATH`).

Example:

```bash
PYTHONPATH=service python - <<'PY'
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
PY
```

---

## External vLLM Acceleration (Multi-GPU)

**Why:** In-process vLLM inside this service is single-GPU. For true multi-GPU acceleration, run an external
vLLM OpenAI-compatible server and point this service to it via `--backend vlm-http-client --server-url ...`.

### A) Start a vLLM Server

Recommended helper script:

```bash
./scripts/start_vllm.sh --gpus 4,6 --tp 2 --port 30000
```

Script usage (common flags):

```bash
# Use a different model path
./scripts/start_vllm.sh --model-path /path/to/vlm_model --gpus 0,1 --tp 2 --port 30000

# Use HuggingFace cache for MinerU2.5
./scripts/start_vllm.sh --model-key mineru2.5 --model-source hf --gpus 0,1 --tp 2 --port 30000

# Single GPU
./scripts/start_vllm.sh --gpus 0 --tp 1 --port 30000

# Pass extra vLLM args (after --)
./scripts/start_vllm.sh --gpus 0,1 --tp 2 --port 30000 -- --max-model-len 8192
```

Manual equivalent:

```bash
conda activate mineru

CUDA_VISIBLE_DEVICES=4,6 \
vllm serve /home/dataarc/.cache/modelscope/hub/models/OpenDataLab/MinerU2.5-2509-1.2B \
  --served-model-name MinerU2.5-2509-1.2B \
  --port 30000 \
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.8
```

Notes:
- `--tensor-parallel-size` must match the number of visible GPUs.
- vLLM pre-allocates VRAM; adjust `--gpu-memory-utilization` if you see OOM.
- vLLM serves **VLM models only**; OCR models (e.g., PaddleOCR) are **not** served by vLLM.

### Stopping vLLM / MinerU (and freeing VRAM)

If you start vLLM via `start_vllm_pool.sh`, each instance writes:
- pid: `/tmp/vllm_<port>.pid`
- log: `/tmp/vllm_<port>.log`

Use the stop helper:

```bash
# stop vLLM on ports
./scripts/stop_mineru_vllm.sh --vllm-ports 30000,30001 --clean

# stop MinerU servers on ports
./scripts/stop_mineru_vllm.sh --mineru-ports 8897,8898,8899 --clean
```

If you already `kill -9` the LISTEN process (via `lsof -i:<port>`), VRAM can still be occupied by
`VLLM::EngineCore` child processes. In that case, identify the GPUs used and kill EngineCore on them:

```bash
# example: vLLM used GPU 5 and 6
./scripts/stop_mineru_vllm.sh --kill-enginecore-gpus 5,6
```

### B) No-Client-Change Option (Recommended)

Start vLLM, then start this service pointing to it:

```bash
python mineru_main.py server \
  --backend vlm-http-client \
  --server-url http://127.0.0.1:30000
```

Existing clients keep calling `/parse` with no changes.

### C) Gateway + Service Pool (Recommended for Throughput)

If you run **multiple** vLLM instances (each on different GPUs/ports), you can run a **single** MinerU server
as a gateway and configure a server URL pool:

Start a vLLM pool (one vLLM instance per GPU):

```bash
conda activate mineru
./scripts/start_vllm_pool.sh --gpus 4,6 --base-port 30000
```

Then start the MinerU gateway:

```bash
python mineru_main.py server \
  --backend vlm-http-client \
  --server-urls http://127.0.0.1:30000,http://127.0.0.1:30001 \
  --max-jobs 2 \
  --max-pending 16
```

The gateway routes each parse job to the least-busy endpoint (tracked by in-flight requests). When an endpoint
errors, it is temporarily put into cooldown and avoided.

**TP vs pool (important):**
- `start_vllm.sh --gpus 4,6 --tp 2` starts *one* model sharded across two GPUs: a single request uses **both** GPUs.
- `start_vllm_pool.sh --gpus 4,6` starts *two* models (tp=1 each): concurrent requests can be routed to **different** GPUs.

---

## CLI Reference (All Options)

### Server CLI

```bash
python mineru_main.py server --help
```

**Networking & capacity**
- `--host` (default: `0.0.0.0`)
- `--port` (default: `8899`)
- `--workers` (default: `1`) — uvicorn worker processes
- `--max-jobs` (default: `1`) — max concurrent parse jobs per worker
- `--max-pending` (default: `0`) — max queued+running jobs, `0` = unlimited
  - env: `MINERU_SERVER_MAX_PENDING_JOBS`
- `--enforce-backend` (default: `false`) — ignore request-level backend override (gateway mode)
  - env: `MINERU_SERVER_ENFORCE_BACKEND=1`

**MinerU defaults**
- `--backend` (default: `vlm-transformers`)
- `--parse-method` (default: `auto`)
- `--lang` (default: `ch`)
- `--formula-enable` / `--no-formula` (default: enabled)
- `--table-enable` / `--no-table` (default: enabled)

**External VLM server (http-client backends)**
- `--server-url` — OpenAI-compatible server URL
  - env: `MINERU_VLM_SERVER_URL`
  - used with `vlm-http-client` / `hybrid-http-client`
- `--server-urls` — comma-separated OpenAI-compatible server URL pool (gateway mode; per-request routing)
  - can be repeated
  - env: `MINERU_VLM_SERVER_URLS`

**Device & model sources**
- `--model-source` (default: `modelscope`)
- `--device` (default: `cuda`)
- `--virtual-vram-gb` (default: `None`)

**vLLM knobs (in-process vLLM only)**
- `--vllm-gpu-mem-util` (default: `0.5`)
- `--vllm-enforce-eager` (default: `false`)
- `--vllm-max-model-len` (default: `None`)
- `--vllm-swap-space-gb` (default: `4.0`)
- `--vllm-cpu-offload-gb` (default: `0.0`)

**Storage paths**
- `--output-dir` (default: `mineru_outputs`)
- `--temp-dir` (default: `.tmp/mineru_temp`)
- `--modelscope-cache-dir` (default: `~/.cache/modelscope/hub`)
- `--hf-home` (default: `~/.cache/huggingface`)
- `--mineru-home` (default: repo `..`)

**Captioning (LLM image captions)**
- `--caption-mode` (default: `content_list_then_llm`)
- `--chat-api-base-url` (env: `CHAT_API_BASE_URL`)
- `--chat-api-key` (env: `CHAT_API_KEY`)
- `--chat-api-key-file` (env: `CHAT_API_KEY_FILE`)
- `--chat-model` (env: `OPENAI_CHAT_MODEL`)
- `--chat-timeout-s` (default: `60`)
- `--caption-max-images` (default: `0`, <=0 means unlimited)
- `--caption-context` (env: `CAPTION_CONTEXT`)
- `--caption-context-file` (env: `CAPTION_CONTEXT_FILE`)
- `--up` (default: `500`) — context tokens above image ref
- `--down` (default: `500`) — context tokens below image ref

### Client CLI

```bash
python mineru_main.py client --help
```

- `--base-url` (default: `MINERU_SERVER_URL` or `http://127.0.0.1:8899`)
- `--file` (required)
- `--output-dir` (default: `./mineru_client_outputs`)
- `--backend` (default: `vlm-transformers`)
- `--parse-method` (default: `auto`)
- `--lang` (default: `ch`)
- `--formula-enable` / `--no-formula` (default: enabled)
- `--table-enable` / `--no-table` (default: enabled)
- `--start-page` (default: `0`)
- `--end-page` (default: `None`)
- `--output-format` (default: `mm_md`, choices: `mm_md|md_only|content_list`)
- `--timeout` (default: `900`)

---

## HTTP API (Server)

Base URL: `http://<host>:<port>`

Endpoints:
- `GET /health`
- `GET /config`
- `POST /parse`
- `GET /parse/status/{task_id}`
- `POST /parse/batch`
- `GET /task/{task_id}/manifest`
- `GET /task/{task_id}/file/{rel_path}`
- `GET /download/{task_id}/{filename}` (legacy)

`POST /parse` form fields:
- `backend`, `parse_method`, `lang`, `formula_enable`, `table_enable`
- `start_page`, `end_page`, `output_format`
- `wait` (if true, block until parse completes)

---

## SSH Tunnel (Remote GPU)

```bash
ssh -CNg -L 8899:127.0.0.1:8899 <user>@<gpu-host>
export MINERU_SERVER_URL="http://127.0.0.1:8899"
python mineru_main.py client --base-url "$MINERU_SERVER_URL" --file demo.pdf --output-dir ./mineru_client_outputs
```

---

## RAG-ARC Integration

- `PARSER_PARSE_MODE=mineru`
- `MINERU_SERVER_URL=http://<server-ip>:8899` (optional `MINERU_TIMEOUT_S=900`)
- Optional backend override: `MINERU_BACKEND=vlm-http-client`
- Optional page range: `MINERU_START_PAGE=0`, `MINERU_END_PAGE=1`

Parsed outputs are mirrored under `PARSER_OUTPUT_DIR` (default `./data/parsed_files/mineru/<file_id>/...`).

### Evidence asset URLs (RAG-ARC API)
- `document_url`: `GET /knowledge/{file_id}/download`
- Markdown image links `images/...` are rewritten to `GET /knowledge/{file_id}/mineru-assets/images/...`
