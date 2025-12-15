# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Run Commands

```bash
# Install dependencies (creates virtual environment automatically)
uv sync

# Install with dev dependencies (for testing)
uv sync --extra dev

# Start FastAPI server
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Run all tests
uv run pytest

# Run specific test file
uv run pytest test/deepsearch/test_planner.py

# Run tests with verbose output
uv run pytest -v --tb=short

# CLI commands (bypass HTTP layer)
uv run rag-arc chat "What is RAG-ARC?"
uv run rag-arc pipeline "query" --skip-llm --json
uv run rag-arc ingest-folder ./docs --owner-id UUID
uv run rag-arc deepsearch "query" --json

# MCP servers
uv run rag-arc tool-mcp-server --transport stdio  # DeepSearch tools (port 8765)
uv run rag-arc chat-mcp-server --transport stdio  # Chat workflow (port 8785)
```

## Architecture Overview

RAG-ARC uses a **layered architecture** with strict separation of concerns:

```
api/          -> FastAPI routes, MCP server endpoints
application/  -> Business logic (RAG inference, knowledge management, DeepSearch)
core/         -> Core algorithms (retrieval, rerank, chunking, graph extraction)
encapsulation/-> External service wrappers (databases, LLMs, embeddings)
config/       -> Pydantic config classes + JSON config files
framework/    -> Base classes (AbstractConfig, AbstractModule, Register)
cli/          -> Typer CLI entry point
```

### Configuration & Registration System

The framework uses a **config-driven factory pattern**:

1. **AbstractConfig** (`framework/config.py`): Base class requiring `type: Literal["TAG"]` discriminator
2. **AbstractModule** (`framework/module.py`): Base class for runtime modules
3. **Register** (`framework/register.py`): Singleton registry that loads JSON configs, substitutes `${ENV_VAR}` placeholders, and builds module instances via `config.build()`

Component initialization flow:
- `main.py` calls `app_registration.initialize()`
- `initialize()` registers modules from `config/json_configs/*.json`
- Routes access modules via `registrator.get_object("module_name")`

### Key Subsystems

**Multi-Path Retrieval** (`core/retrieval/`):
- Dense: FAISS vector search
- Sparse: Tantivy BM25
- Graph: Neo4j with Pruned HippoRAG (subgraph PPR)
- Fusion via RRF in `multipath.py`

**Graph Adapter** (`core/graph_adapter/`):
- Abstracts Neo4j/igraph backends for DeepSearch
- `scope_provider.py` configures owner isolation

**DeepSearch** (`core/deepsearch/`, `application/rag_inference/deepsearch/`):
- Multi-step reasoning over knowledge graphs
- Tools in `core/deepsearch/tools/` (fast/hybrid/heavy categories)
- Tool orchestration via MCP (`encapsulation/deepsearch/tooling/manager.py`)

**Document Pipeline** (`core/file_management/`):
- Parsers: native, dots_ocr, vlm_ocr
- Chunkers: token, semantic, recursive, markdown_header
- Indexing: FAISS, BM25, graph (NetworkX/Neo4j)

### Database Connections

All in `encapsulation/database/`:
- PostgreSQL: metadata, users, sessions (`relational_db/postgresql.py`)
- Redis: caching (`cache_db/redis_db.py`)
- Neo4j: knowledge graphs (`graph_db/neo4j.py`, `pruned_hipporag_neo4j.py`)
- FAISS: dense vectors (`vector_db/faiss.py`)
- MinIO/Local: file blobs (`file_db/`)

### Environment & Profiles

Set `MODEL_PROFILE=api` (default) or `MODEL_PROFILE=local` to switch between OpenAI API and local HuggingFace models. Provider-specific env vars (`CHAT_MODEL_PROVIDER`, `EMBEDDING_MODEL_PROVIDER`, etc.) override profile defaults.

For local development with Docker services, set `DEVELOP_MODE=true` in `.env` to expose PostgreSQL/Redis/Neo4j ports to localhost.

## Testing Notes

- Tests require `.env` configuration with valid API keys
- Integration tests gated by env flags: `RUN_RAGARC_INTEGRATION_TESTS=1`, `RUN_RAGARC_POSTGRES_TESTS=1`, etc.
- Use `pytest.mark.integration` for tests requiring external services
- `asyncio_mode = "auto"` is configured in `pyproject.toml`
