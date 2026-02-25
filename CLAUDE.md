# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-ARC is a modular Retrieval-Augmented Generation framework with multi-path retrieval (Dense/BM25/Graph), graph extraction, and fusion ranking. The system uses a layered architecture with factory patterns, singleton management, and a component registry system.

## Development Commands

### Environment Setup
```bash
# Install dependencies (creates virtual environment automatically)
uv sync

# Install with development dependencies (for tests)
uv sync --extra dev

# Download local models (only needed for local profile)
uv run python download_models.py
```

### Running the Service
```bash
# Start FastAPI server
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Docker deployment
./build.sh   # Build images (one-time setup)
./start.sh   # Start all services
./stop.sh    # Stop containers (keeps data)
```

### Testing
```bash
# Run all tests
uv run pytest

# Run with verbose output
uv run pytest -v

# Run specific test file
uv run pytest test/core/test_rag_pipeline.py

# Run with short traceback
uv run pytest --tb=short
```

**Important**: Tests require `.env` configuration. Integration tests are opt-in via environment flags:
- `RUN_RAGARC_INTEGRATION_TESTS=1` - GPU/model-heavy tests
- `RUN_RAGARC_POSTGRES_TESTS=1` - PostgreSQL integration tests
- `RUN_RAGARC_CHAT_STORAGE_TESTS=1` - Chat storage tests
- `RUN_RAGARC_VECTOR_TESTS=1` - FAISS/vector tests

### CLI Commands (No HTTP Server)
```bash
# Ingest documents (upload + parse + index + graph build)
uv run rag-arc ingest-folder ./docs --owner-id <UUID>

# Parse only (no indexing)
uv run rag-arc parse-folder ./docs --owner-id <UUID>

# List files
uv run rag-arc list-files --owner-id <UUID> --json

# Re-run indexing for existing files
uv run rag-arc trigger-index FILE_ID_1 FILE_ID_2

# Chat with RAG system
uv run rag-arc chat "What is RAG-ARC?" --owner-id <UUID>

# Graph-only QA
uv run rag-arc graph-qa "What relations exist?" --json

# Export graph
uv run rag-arc export-graph --output graph.json

# DeepSearch
uv run rag-arc deepsearch "Question?" --with-evidence --json

# MCP servers
uv run rag-arc tool-mcp-server --transport stdio
uv run rag-arc chat-mcp-server --transport stdio
```

**Note**: CLI requires backing services (PostgreSQL/Redis/Neo4j) to be running. Set `DEVELOP_MODE=true` in `.env` to expose Docker service ports to localhost.

## Architecture

### Layered Structure

1. **Framework Layer** (`framework/`)
   - `register.py`: Component registry (singleton) that loads JSON configs and instantiates modules
   - `module.py`: Base `AbstractModule` class for all business modules
   - `config.py`: Base `AbstractConfig` class for configuration
   - `singleton_decorator.py`: Singleton pattern implementation
   - `shared_module_decorator.py`: Shared instance management for retrievers/embeddings
   - `thread_pool.py`: Global thread pool for async operations

2. **Configuration Layer** (`config/`)
   - `json_configs/`: JSON configuration files with `${ENV_VAR}` placeholder support
     - `rag_inference.json` / `rag_inference_local.json`: RAG pipeline config
     - `knowledge.json` / `knowledge_local.json`: Document processing config
     - `deepsearch_service.json`: DeepSearch configuration
     - `account.json`, `session.json`, `chat_message.json`: User management
   - `application/`: Python config classes (Pydantic models)
   - `core/`: Core module configurations
   - `env-en.md` / `env-zh.md`: Full environment variable reference

3. **Core Layer** (`core/`)
   - `file_management/`: Document parsing (PDF/DOCX/PPT/Excel) and chunking strategies
   - `retrieval/`: Dense (FAISS), BM25 (Tantivy), Graph (Neo4j HippoRAG), and hybrid fusion
   - `rerank/`: Re-ranking algorithms (listwise, LLM-based, Qwen local)
   - `query_rewrite/`: Query rewriting and expansion
   - `graph_adapter/`: Graph database adapters (Neo4j, igraph)
   - `prompts/`: Prompt templates for LLM interactions

4. **Encapsulation Layer** (`encapsulation/`)
   - `llm/`: LLM provider abstractions (OpenAI, HuggingFace, vLLM)
   - `database/`: Database interfaces (PostgreSQL, Redis, Neo4j, MinIO)
   - `data_model/`: Pydantic models and schemas
   - `message_queue/`: Task queue abstractions (Celery, in-process)
   - `web_search/`: Web search integration (Tavily)

5. **Application Layer** (`application/`)
   - `rag_inference/`: RAG inference pipeline and chat logic
   - `knowledge/`: Knowledge management (file upload, indexing, deletion)
   - `account/`: User authentication and authorization
   - `deepsearch/`: DeepSearch service (graph-first reasoning)
   - `intent_routing/`: Semantic intent routing

6. **API Layer** (`api/`)
   - `routers/`: FastAPI route definitions
   - `middleware/`: Request/response middleware
   - `mcp/`: MCP server implementations

### Key Design Patterns

**Component Registry Pattern**:
- `app_registration.py` initializes all modules at startup
- `Register` singleton loads JSON configs and instantiates modules via factory pattern
- Modules are retrieved via `registrator.get_object("module_name")`

**Factory Pattern**:
- LLM, Embedding, and Retriever components use factory methods
- Configuration-driven instantiation via JSON configs

**Shared Instance Management**:
- Retrievers and embedding models use `@shared_module` decorator
- Prevents duplicate model loading and improves performance
- Instances are reused across requests based on configuration fingerprint

**Owner-Scoped Data Isolation**:
- All data (files, chunks, graphs) is scoped to `owner_id` (UUID)
- Multi-tenant isolation at database and index level
- Admin can access cross-tenant data via `ADMIN_OWNER_ID`

## Configuration System

### Two-Layer Configuration

1. **`.env` file**: Secrets and feature switches only
   - Required: `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `JWT_SECRET_KEY`
   - Optional: Infrastructure overrides (PostgreSQL/Redis/Neo4j connection details)
   - Feature flags: `MODEL_PROFILE`, `DEVELOP_MODE`, `TASK_QUEUE_MODE`, `bench_mode`

2. **`config/` directory**: All tunable parameters
   - Retrieval top-k, thresholds, weights
   - Chunking strategies and sizes
   - Model selection and GPU assignment
   - Graph construction parameters

### Model Profiles

Switch between API and local models via `MODEL_PROFILE` in `.env`:

- `MODEL_PROFILE=api` (default): Uses OpenAI-compatible APIs
  - Config: `rag_inference.json`, `knowledge.json`
  - Providers: `openai` for chat/embedding/OCR

- `MODEL_PROFILE=local`: Uses local HuggingFace models
  - Config: `rag_inference_local.json`, `knowledge_local.json`
  - Providers: `huggingface` for chat/embedding, `dots_ocr_parser` for OCR
  - Requires model download: `uv run python download_models.py`

**Important**: When using Docker, rebuild after changing `MODEL_PROFILE`:
```bash
./build.sh  # Rebuild with new .env settings
./start.sh  # Restart services
```

### Environment Variable Substitution

JSON configs support `${ENV_VAR}` placeholders:
```json
{
  "api_key": "${OPENAI_API_KEY}",
  "base_url": "${OPENAI_BASE_URL}"
}
```

The `Register` class automatically substitutes environment variables when loading configs.

## Multi-Path Retrieval System

RAG-ARC uses three parallel retrieval paths with Reciprocal Rank Fusion (RRF):

1. **Dense Retrieval** (FAISS)
   - GPU-accelerated vector similarity search
   - Supports flat, IVF, and HNSW index types
   - Two-stage retrieval for HNSW (ANN prefetch + exact rescoring)
   - Owner-scoped isolation

2. **Sparse Retrieval** (BM25 via Tantivy)
   - Full-text search with BM25 scoring
   - Owner-scoped isolation
   - Supports query variants for mixed-language corpora

3. **Graph Retrieval** (Neo4j HippoRAG)
   - Subgraph PPR (Personalized PageRank on relevant subgraphs)
   - Query-aware pruning for efficiency
   - Incremental updates without full reconstruction
   - Optional dense-seeded file prior to reduce cross-product drift

**Fusion Configuration**:
- Weights controlled via `.env`: `RAG_RETRIEVAL_WEIGHT_DENSE`, `RAG_RETRIEVAL_WEIGHT_BM25`, `RAG_RETRIEVAL_WEIGHT_GRAPH`
- Default: Graph retrieval disabled for normal RAG (weight=0.0) to reduce latency
- Graph is still built and used by DeepSearch
- Dynamic quota allocation via `RAG_RETRIEVAL_DYNAMIC_QUOTA_ENABLED`

## Document Processing Pipeline

### Stages

1. **Upload**: Store file in MinIO or local filesystem
2. **Parse**: Extract text/images from documents
   - Native: PDF text extraction (no OCR)
   - DotsOCR: Local OCR
   - MinerU: Remote parsing service (recommended for complex PDFs)
3. **Chunk**: Split text into chunks
   - Token-based, semantic, recursive, markdown header-based
   - Configurable via `SEMANTIC_CHUNKING_LEVEL`
4. **Index**: Build indexes
   - FAISS (dense vectors)
   - Tantivy (BM25)
   - Neo4j (graph)
5. **Graph Build**: Extract entities/relations and build knowledge graph

### Parse/Index Decoupling

Files can be parsed separately from indexing:
- `ingest_mode=parse`: Upload + parse only (status: `PARSED`)
- `ingest_mode=index`: Full pipeline (status: `INDEXED`)
- `trigger-index`: Re-index existing `PARSED` files without re-parsing

This is useful when:
- Parsing is expensive (MinerU with VLM)
- You want to parse once and index multiple times with different parameters
- Sharing parsed output across environments

## GraphRAG Implementation

Based on HippoRAG2 with enhancements:

1. **Subgraph PPR**: Compute Personalized PageRank on relevant subgraphs (not full graph)
2. **Query-Aware Pruning**: Dynamically adjust neighbor retention based on entity relevance
3. **Incremental Updates**: Update graph without full reconstruction
4. **Dense-Seeded File Prior**: Boost file passage weights when dense chunks concentrate on single file

**Knowledge Graph Maintenance**:
- L0 (hot-path): Materialize `EntityMention` during ingest for disambiguation
- L1 (background): Disambiguate same-name entities using mention context
- Creates `EntityIdentity` cluster centers with `RESOLVED_TO` relationships
- Configurable via `kg_maintenance` in `knowledge*.json`

## PageIndex (Long Document Navigation)

DeepSearch uses PageIndex for efficient navigation of long documents:

1. **Document-level routing**: `locate` aggregates relevant files
2. **Section-level routing**: `toc.tree` / `section.select` for ToC navigation
3. **Page-level reading**: `read.pages` for full context

Controlled via `.env`:
- `PAGEINDEX_ENABLED=true`
- `SECTION_INDEX_ENABLED=true`
- `PAGEINDEX_TOC_CHECK_PAGE_NUM=20`
- `PAGEINDEX_MAX_PAGE_NUM_EACH_NODE=10`

## DeepSearch

Graph-first reasoning system with think→explore→report loop:

- **Think**: Plan exploration strategy
- **Explore**: Execute tools (locate, graph traversal, web search)
- **Report**: Synthesize findings

**Tools**:
- `locate`: Document-level routing
- `toc.tree` / `section.select`: Section navigation
- `read.pages`: Full page reading
- `web.search`: Real-time web search (Tavily)
- `code.python`: Deterministic math/finance verification

**Configuration**:
- `config/json_configs/deepsearch_service.json`
- Tool manager with MCP routing support
- Evidence bundle (chunks/triples/seeds) via `include_evidence=true`

## Testing Guidelines

### Test Structure
- Unit tests: `test/core/`, `test/framework/`, `test/encapsulation/`
- Integration tests: Marked with `@pytest.mark.integration`
- API tests: `test/api/`

### Running Integration Tests
Integration tests require external services and are opt-in:
```bash
# Enable all integration tests
export RUN_RAGARC_INTEGRATION_TESTS=1
export RUN_RAGARC_POSTGRES_TESTS=1
export RUN_RAGARC_CHAT_STORAGE_TESTS=1
export RUN_RAGARC_VECTOR_TESTS=1

# Run tests
uv run pytest
```

### Test Dependencies
- PostgreSQL, Redis, Neo4j must be running
- Set connection details in `.env`
- For E2E tests, set `RAGARC_E2E_TOKEN` (JWT bearer token)

## Common Development Patterns

### Adding a New Retriever
1. Create retriever class in `core/retrieval/`
2. Implement `retrieve()` method returning `List[Chunk]`
3. Add factory method in retriever config
4. Register in `config/json_configs/rag_inference*.json`
5. Update fusion weights if needed

### Adding a New Parser
1. Create parser class in `core/file_management/parsers/`
2. Implement `parse()` method returning markdown
3. Register in `ParserFactory`
4. Add to `knowledge*.json` parser config

### Adding a New Chunking Strategy
1. Create chunker class in `core/file_management/chunking/`
2. Implement `chunk()` method returning `List[Chunk]`
3. Register in `ChunkerFactory`
4. Add to `knowledge*.json` chunker config

### Accessing Registered Modules
```python
from app_registration import registrator

# Get module instance
rag_inference = registrator.get_object("rag_inference")
knowledge = registrator.get_object("knowledge")
deepsearch = registrator.get_object("deepsearch_service")

# Call module methods
result = await rag_inference.chat(query="...", owner_id=uuid.UUID(...))
```

## Important Notes

### Owner ID Management
- Always pass `owner_id` (UUID) for data isolation
- CLI caches default owner in `~/.rag_arc_owner_id`
- Override via `--owner-id` flag or `CLI_OWNER_ID` env var
- Admin access via `ADMIN_OWNER_ID` in `.env`

### Deletion Behavior
- CLI `delete-file`: Marks file as `DELETED` (metadata only)
- API `DELETE /knowledge/{file_id}`: Full async cleanup (chunks, indexes, blobs, graph)
- Use API for production deletions

### Dependency Health Checks
- Startup checks: PostgreSQL, Redis, Neo4j, MinerU (if configured)
- Mode: `RAGARC_DEPENDENCY_CHECK_MODE` (off/warn/strict)
- Indexing checks: `RAGARC_INDEXING_DEPENDENCY_CHECK_MODE`

### Logging
- Logs written to `./log/` directory
- Daily rotation with size limits (10MB per file, 30 days retention)
- Beijing timezone (UTC+8) for timestamps
- Request ID tracking via `X-Request-ID` header

### Security
- JWT-based authentication
- Role-based access control (RBAC)
- Document-level permissions (VIEW/EDIT)
- Bcrypt password hashing
- Owner-scoped data isolation

## Troubleshooting

### Module Registration Failures
Check logs from `app_registration.initialize()` for config validation errors. Common issues:
- Missing environment variables in JSON placeholders
- Invalid JSON syntax
- Missing model files for local profile
- Database connection failures

### FAISS Index Issues
- Ensure `FAISS_INDEX_PATH` is consistent between indexing and retrieval
- For HNSW, enable two-stage retrieval: `FAISS_TWO_STAGE_ENABLED=true`
- Check GPU availability for `faiss-gpu-cu12`

### Graph Retrieval Issues
- Verify Neo4j connection: `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`
- Check graph index path: `GRAPH_STORAGE_PATH`
- Ensure graph weight > 0: `RAG_RETRIEVAL_WEIGHT_GRAPH=1.0`

### MinerU Parsing Failures
- Check `MINERU_SERVER_URL` is reachable
- Increase timeout: `MINERU_TIMEOUT_S=900`
- Enable fallback: `MINERU_FALLBACK_TO_NATIVE_ON_FAILURE=true`
- Check MinerU server logs for VLM issues

### CLI Not Working
- Ensure `DEVELOP_MODE=true` to expose Docker ports
- Check backing services are running
- Verify `.env` connection details match Docker setup
- Install libpq: `sudo apt install -y libpq5 libpq-dev`
