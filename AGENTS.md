# Repository Guidelines

## Project Structure & Module Organization
RAG-ARC follows a layered layout: `framework/` handles base module/config loading, `encapsulation/` wraps provider and database adapters, `core/` implements chunking, retrieval, rerank, and prompt logic, and `application/` assembles `rag_inference`, `knowledge`, and `account` flows. `api/` (FastAPI + MCP) and `cli/` (Typer `rag-arc`) sit above, configs live under `config/`, artifacts in `assets/`, `models/`, `data/`, and mirrored suites in `test/` track each package.

## Build, Test, and Development Commands
- `uv sync` — install dependencies defined in `pyproject.toml`.
- `uv run uvicorn main:app --reload` — run FastAPI locally (add `--host 0.0.0.0 --port 8000` when sharing).
- `./build.sh && ./start.sh` — build/start the Docker stack; `./stop.sh` halts services and `./cleanup.sh` resets data.
- `uv run rag-arc ingest-folder ./example/docs --owner-id <UUID>` — CLI ingestion; swap `chat`, `graph-qa`, or `export-graph` to test other flows without HTTP.

## Coding Style & Naming Conventions
Target Python 3.11, four-space indentation, and PEP 8 spacing. Type public functions, validate IO with Pydantic, and keep business logic in `core/` or `application/` instead of routers. Use `snake_case` modules/functions, `PascalCase` classes, `UPPER_SNAKE_CASE` constants/env keys, verb-noun Typer commands (`rag-arc trigger-index`), and name JSON configs after their workflow (e.g., `rag_inference.json`).

## Testing Guidelines
Run Pytest via `uv run pytest`; scope it to folders like `test/core` for quicker loops. Extend scenario suites such as `test/test_complete_e2e_api.py` and `test/test_user_isolation_e2e.py` whenever ingestion, retrieval, auth, or isolation logic changes. Every feature must add at least one assertion or fixture that exercises the dense, sparse, or graph path it touches.

## Commit & Pull Request Guidelines
History favors concise summaries (`Dev/glk (#18)`, `Bug fix; change mirror source for uv install in China`), so keep commits imperative, single-purpose, under 72 characters, and reference issues or PRs only when relevant. Squash WIP noise before review. Pull requests should state intent, list validation commands (`uv run pytest`, CLI/Docker smoke tests), link issues, and flag schema, config, or manual steps.

## Security & Configuration Tips
Create `.env` from `.env.example`, but never commit populated secrets; keep them in your shell or a secret manager. Mention edits to provider toggles (`MODEL_PROFILE`, `*_MODEL_PROVIDER`) or the JSON files in `config/`, and rebuild Docker images whenever those values change. Use `./clean-docker-data.sh` only when you accept wiping Postgres/Redis/Neo4j volumes and cached models.
