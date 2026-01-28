## MinerU Parsing Service (Standalone)

This directory contains a standalone FastAPI service + clients that wrap the upstream [MinerU](https://github.com/opendatalab/MinerU) multimodal parser.

- It is **deployable independently from RAG-ARC** (imports, environment, deployment).
- It is designed to run on a separate GPU machine and be accessed from your laptop via `MINERU_SERVER_URL` or an SSH tunnel.
- RAG-ARC can use it as a remote PDF/image parsing backend via `.env` (see “RAG-ARC Integration” below).

---

## What It Does

**Server (`mineru_server`)**

- Exposes an HTTP API to parse PDF/images using upstream MinerU.
- Produces Markdown, JSON content lists, and extracted assets (images).
- Optionally generates image captions for retrieval using an OpenAI-compatible multimodal chat endpoint.
- Stores all task outputs under `--output-dir/<task_id>/...` and provides download endpoints.

**Clients**

- `mineru_client.py`: a simple HTTP client + CLI that can (1) trigger parsing, (2) download primary artifacts, and (3) mirror the full server task directory.
- `mineru_server/client.py` + `mineru_server/cli.py client`: a minimal “sync task outputs” helper.

---

## Prerequisites (Upstream MinerU)

This service **does not install or configure upstream MinerU for you**. Please follow the official upstream tutorial to create a working MinerU environment (CUDA, models, backends, etc.).

Once the upstream MinerU import works in the same Python environment, this service can call:
- `mineru.cli.common.do_parse` / `aio_do_parse`
- `mineru.utils.enum_class.MakeMode`

---

## Quick Start

### 1) Server (recommended: run on a GPU machine)

From the repository root:

```bash
# run directly (your environment must already have MinerU installed)
python mineru/mineru_main.py server --host 0.0.0.0 --port 8899
```

Recommended production settings (portable paths, not inside the repo):

```bash
python mineru/mineru_main.py server \
  --host 0.0.0.0 --port 8899 \
  --output-dir /data/mineru_outputs \
  --temp-dir /tmp/mineru_temp
```

If you enable LLM captioning:

```bash
export CHAT_API_BASE_URL="https://api.openai.com/v1"
export CHAT_API_KEY="sk-xxx"
export OPENAI_CHAT_MODEL="gpt-4o-mini"

python mineru/mineru_main.py server \
  --caption-mode content_list_then_llm
```

Tip: use `--caption-max-images` to cap how many images are sent to the LLM for captioning; set it to `0` (or any negative value) to remove the limit.
Captioning defaults to `content_list_then_llm` with no image cap (`caption_max_images=0`). Any multimodal chat model that supports image inputs can be used.

Health check:

```bash
curl http://127.0.0.1:8899/health
```

---

## Multi-GPU vLLM Acceleration (Recommended for High Throughput)

This service supports vLLM backends, but **in-process vLLM currently runs on a single GPU only** because
we do not pass vLLM tensor-parallel settings from this service. For **true multi-GPU acceleration**, run
an external vLLM OpenAI-compatible server with tensor parallelism and use MinerU's `vlm-http-client`
backend (or extend this service to pass `server_url`).

### A) Start a multi-GPU vLLM server (OpenAI-compatible)

Example (2 GPUs, H800 80GB, ModelScope cache):

```bash
# Recommended: use the helper script in service/scripts
./service/scripts/start_vllm.sh --gpus 4,6 --tp 2 --port 30000
```

Script usage (common flags):

```bash
# Use a different model path
./service/scripts/start_vllm.sh --model-path /path/to/vlm_model --gpus 0,1 --tp 2 --port 30000

# Use HuggingFace cache for MinerU2.5
./service/scripts/start_vllm.sh --model-key mineru2.5 --model-source hf --gpus 0,1 --tp 2 --port 30000

# Single GPU
./service/scripts/start_vllm.sh --gpus 0 --tp 1 --port 30000

# Pass extra vLLM args (after --)
./service/scripts/start_vllm.sh --gpus 0,1 --tp 2 --port 30000 -- --max-model-len 8192
```

Equivalent manual command:

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
- vLLM only serves VLM models; OCR models (e.g., PaddleOCR) are **not** served by vLLM.

### B) Use MinerU http-client backend (example)

The upstream MinerU API supports `vlm-http-client` with a `server_url` (OpenAI-compatible):

```python
from pathlib import Path
from mineru.cli.common import aio_do_parse, read_fn

await aio_do_parse(
    output_dir="/tmp/mineru_outputs",
    pdf_file_names=["demo3"],
    pdf_bytes_list=[read_fn(Path("demo/pdfs/demo3.pdf"))],
    p_lang_list=["ch"],
    backend="vlm-http-client",
    server_url="http://127.0.0.1:30000",
)
```

If you want to keep using this service API, you can add a `server_url` parameter and pass it through to
`run_mineru_parse()` → `aio_do_parse()`; that enables multi-GPU vLLM without changing your client.

### Acceleration effect (example on this machine)

Hardware: 2 × NVIDIA H800 80GB, vLLM 0.11.0, `--gpu-memory-utilization 0.8`, warm run.

- PDF: `(详)盛利2-至尊 产品手册(英文版).pdf` (126 pages)
- Wall time: **~62s**
- VLM inference time: **~49s** (≈ **2.5 pages/s**)

Small documents can be dominated by overhead; for best gains, use larger PDFs or higher concurrency.

### 2) Client (run on your laptop)

Set the server URL:

```bash
export MINERU_SERVER_URL="http://<server-ip>:8899"
```

Parse a file and download artifacts:

```bash
python mineru/mineru_client.py parse \
  --file /path/to/demo.pdf \
  --output-dir ./mineru_client_outputs
```

Benchmark multiple files:

```bash
python mineru/mineru_client.py bench \
  --files /path/a.pdf /path/b.pdf \
  --concurrency 2 \
  --output-dir ./mineru_client_outputs \
  --no-download
```

---

## Remote Usage via SSH Tunnel

If the GPU machine is not directly reachable:

```bash
ssh -L 8899:127.0.0.1:8899 <user>@<gpu-host>
export MINERU_SERVER_URL="http://127.0.0.1:8899"
python mineru/mineru_client.py parse --file demo.pdf --output-dir ./mineru_client_outputs
```

---

## RAG-ARC Integration

RAG-ARC can route PDF/image parsing to this service:

- Set `PARSER_PARSE_MODE=mineru`
- Set `MINERU_SERVER_URL=http://<server-ip>:8899` (and optionally `MINERU_TIMEOUT_S=900`)
- Optional page range (0-based, inclusive end): `MINERU_START_PAGE=0`, `MINERU_END_PAGE=1`

Parsed outputs are mirrored under `PARSER_OUTPUT_DIR` (default `./data/parsed_files/mineru/<file_id>/...`), where `<file_id>` is the RAG-ARC file ID used to avoid filename collisions.

### Evidence asset URLs (RAG-ARC API)

When `include_evidence=true`, RAG-ARC returns:
- `document_url`: `GET /knowledge/{file_id}/download`
- For MinerU-parsed Markdown image links like `![...](images/xxx.jpg)`, the backend rewrites them to `GET /knowledge/{file_id}/mineru-assets/images/...` so frontends can render images directly (auth required).

## HTTP API (Server)

Base URL: `http://<host>:<port>`

### Endpoints

- `GET /health`: server health + effective runtime info.
- `GET /config`: returns config (secrets redacted).
- `POST /parse`: parse a single file (multipart upload).
- `GET /parse/status/{task_id}`: poll parse status for async parse jobs.
- `POST /parse/batch`: parse multiple files (multipart upload).
- `GET /task/{task_id}/manifest`: list files under `output_dir/<task_id>/...`.
- `GET /task/{task_id}/file/{rel_path}`: download a specific file by relative path (collision-free).
- `GET /download/{task_id}/{filename}`: legacy “search by filename” download (may be ambiguous if duplicates exist).

### `POST /parse` parameters (form fields)

- `backend`: default from server config (`vlm-transformers` recommended).
- `parse_method`: default `auto`.
- `lang`: default `ch`.
- `formula_enable`: `true|false`.
- `table_enable`: `true|false`.
- `start_page`: `0`-based start page.
- `end_page`: inclusive end page (optional).
- `output_format`: `mm_md | md_only | content_list`.
- `wait`: if `true`, block until parsing completes (default: `false`, returns immediately).

The response includes (for async, poll `GET /parse/status/{task_id}` until `status=success`):
- `task_id`, `status`, `processing_time`
- absolute paths (`markdown_path`, `images_dir`, ...) and **task-relative paths** (`*_rel_path`) for robust downloads
- `images_metadata` with `task_rel_path` for each image

---

## CLI Reference

### Server CLI

```bash
python mineru/mineru_main.py server --help
```

Key options:
- Networking: `--host`, `--port`, `--workers`, `--max-jobs`, `--max-pending`
- Storage: `--output-dir`, `--temp-dir`, `--modelscope-cache-dir`, `--hf-home`, `--mineru-home`
- MinerU defaults: `--backend`, `--parse-method`, `--lang`, `--no-formula`, `--no-table`, `--model-source`, `--device`
- vLLM knobs (vLLM backends only): `--vllm-gpu-mem-util`, `--vllm-enforce-eager`, `--vllm-max-model-len`, `--vllm-swap-space-gb`, `--vllm-cpu-offload-gb`
- Captioning: `--caption-mode`, `--caption-max-images`, `--chat-api-base-url`, `--chat-api-key`, `--chat-api-key-file`, `--chat-model`, `--caption-context`, `--caption-context-file`, `--up`, `--down`

### Client CLI (simple downloader)

```bash
python mineru/mineru_client.py --help
```

---

## Python Usage (Code Integration)

### Minimal parse + sync full task directory

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

By default, the client polls `/parse/status/{task_id}` until completion. Pass `wait=False` to return immediately with a pending status.

### Download primary artifacts (md/json/images)

```python
from pathlib import Path
from mineru_client import MinerUClient

client = MinerUClient(base_url="http://127.0.0.1:8899", timeout=900)
parse_result = client.parse_file(Path("demo.pdf"))
downloaded = client.download_artifacts(parse_result, Path("./mineru_client_outputs"))
print(downloaded["document_dir"])
```

---

## Environment Variables

### Client

- `MINERU_SERVER_URL`: default server base URL for clients (e.g. `http://127.0.0.1:8899`).

### Server dotenv loading

- `MINERU_DOTENV_PATH`: explicit path to a `.env` file.

If not set, the server loads `.env` in this order:
1) `mineru/.env` (service-local, recommended)
2) `<cwd>/.env`

### LLM captioning (required only when `--caption-mode` uses LLM)

- `CHAT_API_BASE_URL`: OpenAI-compatible base URL (e.g. `https://api.openai.com/v1`).
- `CHAT_API_KEY`: API key (prefer `CHAT_API_KEY_FILE` for multi-worker deployments).
- `CHAT_API_KEY_FILE`: path to a file that contains the API key.
- `OPENAI_CHAT_MODEL`: multimodal model name (must support image input).

Optional:
- `CAPTION_CONTEXT`: extra fixed text prepended to the caption prompt.
- `CAPTION_CONTEXT_FILE`: file version of `CAPTION_CONTEXT`.
- `CAPTION_UP`: tokens above image anchor (default `500`).
- `CAPTION_DOWN`: tokens below image anchor (default `500`).

### Caches (optional but recommended for portability)

- `MODELSCOPE_CACHE`: ModelScope cache root (otherwise defaults to `~/.cache/modelscope/hub`).
- `HF_HOME`: HuggingFace cache root (otherwise defaults to `~/.cache/huggingface`).
- `XDG_CACHE_HOME`: changes the base cache root used by defaults.

Note: the server sets the upstream MinerU runtime env vars (e.g. `MINERU_DEVICE_MODE`, `MINERU_OUTPUT_DIR`, `HF_HOME`, ...) from its own config for reproducibility.

---

## Output Layout

Server output root: `--output-dir` (default: `mineru/mineru_outputs`)

Each request is stored under:

```
<output-dir>/<task_id>/<doc_name>/<method_dir>/
  <doc_name>.md
  <doc_name>_content_list.json
  images/
```

The client can mirror the entire task directory to:

```
<client-output-dir>/mineru_outputs/<task_id>/...
```

---

## Portability Notes (Move Server to Any Machine)

- Do not rely on repo-relative defaults in production; pass absolute `--output-dir` and `--temp-dir`.
- Configure the client via `MINERU_SERVER_URL` (no hard-coded IPs).
- Store secrets in files (`CHAT_API_KEY_FILE`) instead of inline env vars if you use multiple workers.
- For `--workers > 1`, a config JSON is written under `mineru/.tmp/` to share settings with worker processes.
