# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-ARC is a modular Retrieval-Augmented Generation (RAG) framework built on Python 3.11+ with FastAPI. It implements multi-path retrieval (dense/sparse/graph), graph-based knowledge extraction (GraphRAG with pruned HippoRAG), and a DeepSearch capability for complex reasoning over knowledge graphs.

## Common Commands

### Development Setup
```bash
# Install dependencies (uv auto-creates venv)
uv sync

# Install with dev dependencies (pytest, httpx)
uv sync --extra dev

# Download local models (only needed for MODEL_PROFILE=local)
uv run python download_models.py
```

### Running the Application
```bash
# Start FastAPI server
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# CLI commands (requires .env and backing services)
uv run rag-arc ingest-file ./doc.pdf --owner-id <UUID>
uv run rag-arc chat "What is RAG-ARC?" --owner-id <UUID>
uv run rag-arc export-graph --output graph.json --owner-id <UUID>
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest test/deepsearch/test_planner.py

# Run with verbose output
uv run pytest -v

# Run with short traceback
uv run pytest --tb=short
```

Integration tests require environment flags:
- `RUN_RAGARC_INTEGRATION_TESTS=1`: GPU/model-heavy suites
- `RUN_RAGARC_POSTGRES_TESTS=1`: PostgreSQL integration tests
- `RUN_RAGARC_CHAT_STORAGE_TESTS=1`: Chat storage tests (PostgreSQL + Redis)
- `RUN_RAGARC_VECTOR_TESTS=1`: Faiss/Qwen vector tests

### Docker Deployment
```bash
# Build images (one-time setup)
./build.sh

# Start all services (PostgreSQL, Redis, Neo4j, RAG-ARC)
./start.sh

# Stop containers (keeps data)
./stop.sh

# Clean Docker resources but keep local data
./cleanup.sh

# Full cleanup including data directories
./clean-docker-data.sh
```

## Architecture

### Layered Design
RAG-ARC follows a strict layered architecture with dependency flow downward:

```
api/                    # FastAPI routes, MCP servers
  ↓
application/            # Business logic modules (rag_inference, knowledge, account, deepsearch)
  ↓
core/                   # Core capabilities (retrieval, rerank, file_management, query_rewrite)
  ↓
encapsulation/          # Abstraction layer (database, llm, data_model)
  ↓
framework/              # Framework primitives (module, register, config, decorators)
```

### Component Registration System
The framework uses a centralized registration system (`framework/register.py`) that:
- Loads JSON configs from `config/json_configs/` at startup
- Substitutes environment variables (`${VAR_NAME}` syntax)
- Builds module instances via `AbstractConfig.build()` pattern
- Stores singletons in `Register.registrations` dict

Key registration happens in `app_registration.py`:
```python
registrator.register(config_path, app_name, config_type)
module = registrator.get_object(app_name)  # Retrieve registered instance
```

### Shared Module Pattern
The `@shared_module` decorator (`framework/shared_module_decorator.py`) enables instance reuse:
- Same config parameters → same instance (reduces memory/GPU usage)
- Different parameters → new instance
- Used for LLM clients, embedding models, retrievers, tokenizers

Example: Multiple retrievers sharing the same embedding model will use a single GPU-loaded instance.

### Multi-Path Retrieval Fusion
`core/retrieval/multipath.py` orchestrates multiple retrieval paths:
1. **Dense Retrieval**: FAISS vector similarity (GPU/CPU)
2. **Sparse Retrieval**: BM25 via Tantivy (full-text search)
3. **Graph Retrieval**: Neo4j-based with Pruned HippoRAG (Personalized PageRank on subgraphs)

Fusion methods (configured via `fusion_method`):
- `rrf`: Reciprocal Rank Fusion (default)
- `weighted_sum`: Custom weights per retriever
- `rank_fusion`: Rank-based combination

Graph retriever captures subgraph metadata in chunk metadata (`_subgraph_info`) for downstream use.

### Graph Adapter System
`core/graph_adapter/` provides pluggable graph backends:
- **Base**: `GraphAdapter` abstract interface (`base.py`)
- **HippoRAG**: Default implementation with subgraph PPR (`hipporag.py`)
- **Registry**: `GraphAdapterRegistry` for registering new adapters (`registry.py`)
- **Scope Provider**: Global scope configuration for graph isolation (`scope_provider.py`)

To add a new graph backend, implement `GraphAdapter` and register it via the registry.

### DeepSearch on Graph
DeepSearch (`application/rag_inference/deepsearch/service.py`) implements multi-stage reasoning:
1. **Planner**: Breaks down complex queries into sub-tasks
2. **Graph Adapter**: Executes graph-based retrieval and exploration
3. **Tool Manager**: Coordinates deterministic tools (Fast), LLM-heavy tools (Heavy), and hybrid tools
4. **Gap Detection**: Identifies knowledge gaps and triggers external search (optional, requires `DEEPSEARCH_EXTERNAL_SEARCH_ENABLED=true`)
5. **Report Composer**: Synthesizes evidence into structured reports

Tool categorization (see `docs-proj/deepsearch_on_graph_execution.md`):
- **F-Tools (Fast Deterministic)**: Sub-millisecond pattern/chunk scans
- **H-Tools (Heavy Cognitive)**: LLM-driven chain exploration with "think" windows
- **X-Tools (Hybrid Bridge)**: Deterministic context + LLM explanation

### Document Processing Pipeline
`core/file_management/index_manager.py` coordinates:
1. **Storage**: Files stored in local filesystem or MinIO (configurable via `file_db.type`)
2. **Parsing**: Multi-format support (PDF, DOCX, PPTX, Excel, HTML)
   - Native parsers for standard formats
   - OCR parsers (DOTS-OCR, VLM-based) for scanned documents
3. **Chunking**: Strategies in `core/file_management/chunker/`
   - Token-based, semantic, recursive, markdown header-based
   - Semantic unit chunking (`semantic_unit_chunker.py`) preserves tables/code/lists
4. **Indexing**: Multi-index coordination
   - FAISS (dense vectors)
   - Tantivy (BM25 sparse)
   - Neo4j (graph triples)

### Configuration Profiles
Two built-in profiles (controlled by `MODEL_PROFILE` env var):
- **api** (default): Uses OpenAI-compatible APIs for chat/embedding/OCR
  - Configs: `rag_inference.json`, `knowledge.json`
- **local**: Uses local HuggingFace models
  - Configs: `rag_inference_local.json`, `knowledge_local.json`

Override specific providers via env vars:
- `CHAT_MODEL_PROVIDER`, `EMBEDDING_MODEL_PROVIDER`, `OCR_MODEL_PROVIDER`
- Falls back to `OPENAI_API_KEY`/`OPENAI_BASE_URL` when component keys are empty

When changing providers in Docker, rebuild the image: `./build.sh && ./start.sh`

### User Isolation & Admin Access
- All data (files, chunks, graph) is scoped by `owner_id` (UUID)
- PostgreSQL, FAISS, BM25, and Neo4j queries filter by owner
- Admin access: Set `ADMIN_OWNER_ID` in `.env` to enable cross-owner operations via `include_all_owners=true` or `target_owner_id=<UUID>` params

### Evidence & SSE Streaming
HTTP endpoints support evidence bundles (`include_evidence=true`):
- Chunks, triples, seed entities, graph metadata
- Controlled by env vars: `CHAT_TOP_CHUNKS`, `CHAT_TOP_TRIPLES`, `DEEPSEARCH_TOP_CHUNKS`, etc.
- `ENABLE_ALL_EVIDENCE=true` disables trimming

SSE streaming (`/rag_inference/stream_chat/{session_id}`):
- OpenAI-compatible streaming format
- Evidence delivered via `delta.tool_calls[].function.name == "rag_arc_payload"`
- Progress updates via `delta.tool_calls[].function.name == "rag_arc_progress"`

## Important Implementation Details

### Semantic Unit Chunker
`core/file_management/chunker/semantic_unit_chunker.py` preserves structural units:
- **Tables**: Tracked as anchors with slices (rows)
- **Code blocks**: Preserved with language metadata
- **Lists**: Maintained with hierarchy

Configuration via `SEMANTIC_CHUNKING_LEVEL`:
- `disabled`: Skip semantic processing
- `basic`: Basic structure detection
- `standard`: Enhanced table/code/list handling
- `advanced`: Maximum structure preservation

Anchor backfill in multipath retrieval ensures related structural units are included even if not directly retrieved.

### Graph Construction
Knowledge graph built from chunks via:
1. **Entity Extraction**: LLM identifies entities and relations
2. **Triple Formation**: (subject, predicate, object) triples stored in Neo4j
3. **PPR Indexing**: Personalized PageRank computed on demand for query-relevant subgraphs
4. **Incremental Updates**: New documents extend existing graph without full rebuild

Subgraph PPR (vs. full-graph) improves efficiency and precision by limiting traversal to query-relevant nodes.

### Asynchronous Indexing
`core/file_management/index_manager.py` uses async patterns:
- File upload → indexing trigger (non-blocking)
- Status tracking: `PENDING` → `PROCESSING` → `COMPLETED` or `FAILED`
- Soft deletion: `delete-file` CLI marks status as `DELETED` without cleanup
- Full deletion: HTTP `DELETE /knowledge/{file_id}` triggers async cleanup of chunks, indexes, blobs, graph

### Database Layer
`encapsulation/database/` abstracts storage:
- **relational_db**: PostgreSQL via SQLAlchemy (users, files, sessions, messages)
- **cache_db**: Redis for session state and caching
- **graph_db**: Neo4j for knowledge graph
- **vector_db**: FAISS for dense embeddings
- **bm25_indexer**: Tantivy for sparse retrieval

All use connection pooling and scope-based isolation (owner_id filtering).

### LLM Factory Pattern
`encapsulation/llm/` implements provider-agnostic LLM clients:
- **openai_llm**: OpenAI API (and compatible providers like DeepSeek)
- **huggingface_llm**: Local transformers models
- **vllm_llm**: vLLM for high-throughput inference

Factory selection based on `provider` field in config:
```python
if provider == "openai":
    return OpenAILLM(config)
elif provider == "huggingface":
    return HuggingFaceLLM(config)
```

## File Organization Conventions

### Config Files
- `config/json_configs/*.json`: Module configurations (loaded at startup)
- `config/application/*.py`: Pydantic config schemas
- `config/core/*.py`: Core component configs
- `.env`: Runtime environment variables (do NOT commit secrets)

### Test Structure
```
test/
├── core/                    # Core module tests
│   ├── retrieval/          # Retrieval tests (multipath, graph, BM25, FAISS)
│   └── file_management/    # Parsing, chunking tests
├── encapsulation/          # Database, LLM abstraction tests
├── deepsearch/             # DeepSearch planner, tool tests
└── test_complete_e2e_api.py  # End-to-end HTTP API tests
```

Use pytest markers: `@pytest.mark.integration` for tests requiring external services.

### Data Directories
- `./data/`: Persistent storage (PostgreSQL, Neo4j, Redis data when using Docker)
- `./local/`: Runtime/cache data (file chunks, test outputs)
- `./models/`: Local model weights (Qwen, DOTS-OCR, MiniLM)

These are `.gitignore`d; never commit data or models.

## Development Workflows

### Adding a New Retriever
1. Create class in `core/retrieval/` implementing `BaseRetriever`
2. Add config schema in `config/core/retrieval/`
3. Register in multipath config's `retrievers` list
4. Test isolation with `test/core/retrieval/test_*.py` pattern

### Adding a New LLM Provider
1. Implement in `encapsulation/llm/` following factory pattern
2. Add provider enum to config schemas
3. Update `app_registration.py` provider resolution logic
4. Test with both API and local profiles

### Adding a New Chunker
1. Implement in `core/file_management/chunker/` with `ChunkerConfig`
2. Add to `parser_combinator.py` strategy selection
3. Write tests in `test/core/file_management/chunker/`
4. Document in knowledge config JSON

### Extending DeepSearch Tools
1. Define tool in `application/rag_inference/deepsearch/` or `encapsulation/deepsearch/`
2. Categorize as F-Tool, H-Tool, or X-Tool (see architecture section)
3. Register in tool manager's `build_builtin_tools`
4. Add MCP descriptor if exposing via MCP server
5. Test with `test/deepsearch/test_tools.py`

## Configuration Tips

### Switching Between API and Local Profiles
```bash
# In .env
MODEL_PROFILE=local  # Use local HuggingFace models

# Or for API mode (default)
MODEL_PROFILE=api    # Use OpenAI-compatible APIs
```

After changing, rebuild Docker: `./build.sh && ./start.sh`

### Configuring Multi-Path Retrieval Weights
Edit `config/json_configs/rag_inference.json`:
```json
{
  "retriever": {
    "fusion_method": "weighted_sum",
    "weights": [0.4, 0.3, 0.3],  // [dense, sparse, graph]
    "retrievers": [...]
  }
}
```

### Enabling External Search in DeepSearch
```bash
# In .env
DEEPSEARCH_EXTERNAL_SEARCH_ENABLED=true
TAVILY_API_KEY=<your-key>
```

Gap detector will trigger Tavily web search when graph coverage is insufficient.

### Debug Mode for Services
```bash
# In .env
DEVELOP_MODE=true         # Expose Docker service ports to localhost
LOG_LEVEL=DEBUG           # Enable verbose logging
```

## Key Files Reference

- `main.py`: FastAPI app entry point, router registration
- `app_registration.py`: Component initialization, config loading
- `framework/register.py`: Core registration system
- `core/retrieval/multipath.py`: Multi-path retrieval orchestration
- `core/file_management/index_manager.py`: Document processing coordinator
- `core/graph_adapter/hipporag.py`: HippoRAG graph implementation
- `application/rag_inference/deepsearch/service.py`: DeepSearch orchestrator
- `encapsulation/database/bm25_indexer.py`: Tantivy BM25 implementation
- `api/routers/`: HTTP endpoint definitions (knowledge, rag_inference, deepsearch, auth)
- `cli/rag.py`: CLI command implementations

## Notes for Future Developers

### Maintain Layered Dependencies
Never import from upper layers (e.g., `core/` should not import from `application/`). Encapsulation layer provides abstractions for cross-cutting concerns (database, LLM).

### Profile Consistency
When adding new model-dependent features, support both API and local profiles. Add provider selection in `app_registration.py` and create both `config.json` and `config_local.json`.

### Owner Isolation
All new data models must include `owner_id` field and filter by it in queries. PostgreSQL, vector stores, and graph queries must respect owner boundaries unless `ADMIN_OWNER_ID` is used.

### Async Patterns
New indexing/processing operations should follow the async pattern:
1. Accept request, return immediately with `PENDING` status
2. Process in background (or via task queue)
3. Update status to `COMPLETED`/`FAILED`
4. Provide status query endpoint

### Environment Variable Substitution
Config JSONs support `${VAR_NAME}` substitution. Use this for paths, API keys, and provider selection. See `framework/register.py:_substitute_env_vars()`.

### Graph Adapter Extensibility
When implementing new graph backends (e.g., LightRAG, GraphSearch), extend `GraphAdapter` base class and register via `GraphAdapterRegistry`. Existing DeepSearch tools will work with any compliant adapter.
