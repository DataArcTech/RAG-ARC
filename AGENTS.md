# Repository Guidelines

## Project Structure

- `api/`: FastAPI HTTP API, routers, MCP servers (`api/mcp/`).
- `application/`: business workflows (RAG inference, knowledge, accounts).
- `core/`: core RAG building blocks (parsing/chunking, retrieval, rerank, prompts, utils).
- `encapsulation/` + `framework/`: adapters, shared abstractions, config plumbing.
- `config/`: JSON + Python configuration (`config/json_configs/` is the main entry point).
- `test/`: pytest suite (`test_*.py`).
- Runtime/artifacts: `data/`, `local/`, and `models/` (often large; don’t commit generated contents).

## Build, Test, and Development Commands

- Install deps (creates venv): `uv sync`
- Install dev deps (pytest, httpx): `uv sync --extra dev`
- Run API locally: `uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
- Run CLI (Typer entrypoint): `uv run rag-arc --help`
- Run tests: `uv run pytest`
- Docker (recommended): `./build.sh` (build images) then `./start.sh` (start services) and `./stop.sh` (stop).

## Coding Style & Naming

- Python 3.11+ (see `pyproject.toml`); use 4-space indentation.
- Keep modules layered: `core/` should stay framework-agnostic; put service wiring in `application/`/`api/`.
- Prefer explicit types for request/response models (Pydantic v2) and stable, descriptive names.

## Testing Guidelines

- Framework: `pytest` + `pytest-asyncio` (async tests supported).
- Naming: `test_*.py` and `test_*` functions.
- Integration tests are opt-in and may require Postgres/Redis/Neo4j and model access; enable via env flags like `RUN_RAGARC_INTEGRATION_TESTS=1` (see `README.md`).

## Commits & Pull Requests

- Follow existing history prefixes: `feat: ...`, `fix: ...`, `optimize: ...` (imperative, concise subject).
- PRs should include: summary, how to run/verify (`uv run pytest` or specific test), and any config changes (`.env` / `config/json_configs/*`).

## Security & Configuration

- Never commit secrets from `.env`; update `.env.example` when adding new config knobs.
- Changing model/provider settings for Docker deployments typically requires rebuilding the image (`./build.sh`).
