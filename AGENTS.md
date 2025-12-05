# Repository Guidelines

## Project Structure & Module Organization
RAG-ARC keeps features inside `application/` (rag_inference, knowledge, account) while primitives sit in `core/` and infrastructure adapters in `encapsulation/`. HTTP routes live under `api/routers`, registry/config glue stays in `framework/` and `config/`, and CLI tools reside in `cli/` as the `rag-arc` Typer app. Tests mirror this hierarchy inside `test/`; media, Docker volumes, and cached weights stay in `assets/`, `data/`, and `models/`.

## Build, Test, and Development Commands
Install dependencies with `uv sync`, then run the API locally via `uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload`. `./build.sh` scaffolds `.env` and builds Docker images, `./start.sh` brings up PostgreSQL/Redis/Neo4j plus the app, and `./stop.sh` or `./cleanup.sh` remove them. Exercise pipelines through `uv run rag-arc ingest-folder ./example/docs --owner-id <uuid>` or `uv run rag-arc pipeline "What is RAG-ARC?"`. When switching to local models, fetch weights with `uv run python download_models.py --components embedding reranker` before launching services.

## Coding Style & Naming Conventions
Target Python 3.11, four-space indentation, and snake_case modules/functions; exported classes remain PascalCase. FastAPI routers should expose type-hinted handlers backed by `pydantic.BaseModel` schemas, and any new provider or retriever must be registered in `framework/register.py` plus a descriptive JSON entry under `config/json_configs/`. Store custom prompts in `core/prompts/` and prefer succinct docstrings over inline commentary except for tricky algorithms.

## Testing Guidelines
Pytest drives coverage; always run `uv run pytest` (or a focused command such as `uv run pytest test/test_complete_e2e_api.py -k chat`) before opening a PR. Place unit-level coverage in the matching folder under `test/core`, `test/encapsulation`, or `test/api`, and extend e2e suites (`test/test_user_isolation_e2e.py`, etc.) whenever HTTP or graph workflows change. Keep fixtures idempotent and call `test/cleanup_db.py` if PostgreSQL/Redis/Neo4j state persists between runs.

## Commit & Pull Request Guidelines
Commits favor concise, imperative subjects such as `replace pip with uv` or `Dev/glk (#18)` and should reference issues or PRs in parentheses when relevant. Pull requests summarize behavior changes, flag `.env` or migration updates, and attach CLI/API evidence for new endpoints. Verify `uv run pytest` plus the impacted CLI commands (ingest, chat, pipeline) before requesting review.

## Configuration & Security Notes
Secrets belong in `.env` only; never commit provider keys or database passwords. Toggle between API-hosted and local pipelines via `MODEL_PROFILE`, `CHAT_MODEL_PROVIDER`, and the `*_CONFIG_PATH` variables, then rebuild containers with `./build.sh && ./start.sh` so settings propagate. Keep overrides under `config/json_configs/` and scrub temporary artifacts from `data/`, `local/`, and `models/` before raising a pull request.
