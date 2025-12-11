# RAG-ARC CLI

The CLI lets you exercise the full RAG pipeline (ingestion → indexing/graph build → retrieval/Q&A) without starting the FastAPI server. All commands run inside your existing Python environment via `uv run rag-arc ...` and reuse the same `.env` + backing services (PostgreSQL, Redis, Neo4j, MinIO/local storage).

## Prerequisites
1. Install dependencies with `uv sync` and make sure `.env` points to reachable services. When the backing services run inside Docker, set `DEVELOP_MODE=true` (or the individual `EXPOSE_*` flags) in `.env` so PostgreSQL/Redis/Neo4j ports are exposed to `localhost`. In develop mode the CLI automatically uses `DEVELOP_OWNER_ID` (default `00000000-0000-0000-0000-000000000001`) unless you override `--owner-id`.
2. Ensure the OS has libpq available; on Debian/Ubuntu run `sudo apt install -y libpq5 libpq-dev` before running ingestion commands (needed by `psycopg`).
3. Start the infrastructure (PostgreSQL/Redis/Neo4j/etc.) the same way you would for the API (Docker scripts or local processes) and confirm the exposed ports match the `.env` values.
4. In develop mode the CLI automatically creates a placeholder user (username/password from `DEVELOP_OWNER_USERNAME`/`DEVELOP_OWNER_PASSWORD`) so you can focus purely on algorithm testing; user/permission flows are intentionally bypassed.
5. Optional: set `MODEL_PROFILE` / provider env vars before launching commands.

## Command overview

| Category | Command | Description |
| --- | --- | --- |
| Ingestion | `uv run rag-arc ingest-file ./doc.pdf --owner-id <UUID>` | Upload + chunk + index + build graph for a single file (always pass a stable `--owner-id`). |
| Ingestion | `uv run rag-arc ingest-folder ./docs --pattern '*.pdf' --owner-id <UUID>` | Bulk ingest every file in a folder, recursive by default. |
| Knowledge mgmt | `uv run rag-arc list-files --json --owner-id <UUID>` | List files accessible to the owner (filter by status/limit/offset). |
| Knowledge mgmt | `uv run rag-arc delete-file FILE_ID --owner-id <UUID>` | Mark a file as deleted (metadata only, no cleanup). |
| Knowledge mgmt | `uv run rag-arc trigger-index FILE_ID [FILE_ID ...] --owner-id <UUID>` | Re-run indexing for one or more files. |
| Graph tooling | `uv run rag-arc export-graph --output graph.json --owner-id <UUID>` | Export the entire graph (Neo4j or igraph) to stdout or a JSON file. |
| Retrieval | `uv run rag-arc chat "What is RAG-ARC?" --owner-id <UUID>` | Full pipeline (multi-path retrieval + rerank + LLM). |
| Retrieval | `uv run rag-arc pipeline "What is RAG-ARC?" --skip-llm --subgraph --owner-id <UUID>` | Inspect rewrite/retrieval/rerank without calling the LLM. |
| Graph QA | `uv run rag-arc graph-qa "Explain relation between X and Y" --owner-id <UUID>` | Run graph-only question answering and return subgraph metadata. |
| MCP | `uv run rag-arc tool-mcp-server --transport stdio` | Launch the DeepSearch tool MCP server (config at `config/json_configs/deepsearch_tool_mcp_server.json`, SSE port 8765). |
| MCP | `uv run rag-arc chat-mcp-server --transport stdio` | Launch the chat/auth MCP server defined in `api/mcp/server.py` (SSE/HTTP default to `127.0.0.1:8785/mcp/chat`). |

> The CLI's `delete-file` command is meant for lightweight testing and therefore only updates file metadata/status. For the real asynchronous cleanup pipeline (chunks, indexes, blobs, graph), call the REST API `DELETE /knowledge/{file_id}` instead.

Always pass `--owner-id <UUID>` when you want to reuse the same tenant/user data. If omitted, the CLI falls back to the cached owner UUID (or generates one on first use), which might differ between environments.

## Tips
- `--json` is supported on list/chat/pipeline/graph-qa/export-graph to emit structured output.
- `ingest-folder` respects `--limit`, `--pattern`, and `--no-recursive` to control scope, and fails fast per file.
- `trigger-index` and `export-graph` run against the same graph store configured in `config/json_configs/*` (Neo4j for API profile by default). Ensure those services are accessible before running the commands.
- The CLI caches a default owner ID in `~/.rag_arc_owner_id`. Override it via `--owner-id ...` or by setting `CLI_OWNER_ID`/`RAG_ARC_OWNER_ID` in the environment when you want to share the same tenant across machines.
