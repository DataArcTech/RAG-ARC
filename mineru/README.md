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

Health check:

```bash
curl http://127.0.0.1:8899/health
```

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

Parsed outputs are mirrored under `PARSER_OUTPUT_DIR` (default `./data/parsed_files/mineru/...`) to match the existing RAG-ARC parser artifact layout.

## HTTP API (Server)

Base URL: `http://<host>:<port>`

### Endpoints

- `GET /health`: server health + effective runtime info.
- `GET /config`: returns config (secrets redacted).
- `POST /parse`: parse a single file (multipart upload).
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

The response includes:
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
- Networking: `--host`, `--port`, `--workers`, `--max-jobs`
- Storage: `--output-dir`, `--temp-dir`, `--modelscope-cache-dir`, `--hf-home`, `--mineru-home`
- MinerU defaults: `--backend`, `--parse-method`, `--lang`, `--no-formula`, `--no-table`, `--model-source`, `--device`
- vLLM knobs (vLLM backends only): `--vllm-gpu-mem-util`, `--vllm-enforce-eager`, `--vllm-max-model-len`, `--vllm-swap-space-gb`, `--vllm-cpu-offload-gb`
- Captioning: `--caption-mode`, `--chat-api-base-url`, `--chat-api-key`, `--chat-api-key-file`, `--chat-model`, `--caption-context`, `--caption-context-file`, `--up`, `--down`

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
