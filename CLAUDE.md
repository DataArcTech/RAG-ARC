# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RAG-ARC is a modular Retrieval-Augmented Generation framework with multi-path retrieval, graph structure extraction (GraphRAG), and fusion ranking. The system supports both standard RAG and graph-enhanced RAG modes with a focus on enterprise deployment.

## Build, Test & Run Commands

### Development Setup
```bash
# Install dependencies (uv automatically creates virtual environment)
uv sync

# Configure environment
cp .env.example .env
# Edit .env with your API keys and database credentials
```

### Running the Service
```bash
# Start FastAPI server (development)
uv run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Start with Docker (recommended for production)
./build.sh   # One-time setup: builds images, creates .env
./start.sh   # Starts all services (PostgreSQL, Redis, Neo4j, RAG-ARC)
./stop.sh    # Stops containers but keeps data
./cleanup.sh # Removes containers and volumes but keeps local data
```

### CLI Commands (No HTTP Server Required)
```bash
# Ingest documents
uv run rag-arc ingest-file ./doc.pdf --owner-id YOUR_UUID
uv run rag-arc ingest-folder ./docs --owner-id YOUR_UUID

# Query and retrieve
uv run rag-arc chat "What is RAG-ARC?" --owner-id YOUR_UUID
uv run rag-arc pipeline "query here" --skip-llm --subgraph --json

# Manage knowledge
uv run rag-arc list-files --owner-id YOUR_UUID --json
uv run rag-arc trigger-index FILE_ID --owner-id YOUR_UUID
uv run rag-arc export-graph --output graph.json
```

### Testing
```bash
# Run tests (pytest is available)
uv run pytest

# API integration tests (shell scripts)
./test/api/knowledge_comprehensive_test.sh
./test/api/session_comprehensive_test.sh
./test/api/stream_chat_comprehensive_test.sh
./test/api/mcp_comprehensive_test.sh
```

## Architecture & Design Patterns

### Layered Architecture
The codebase follows a strict layered architecture with clear separation of concerns:

1. **Framework Layer** (`framework/`): Base abstractions and component registry
   - `module.py`: `AbstractModule` base class for all components
   - `register.py`: Singleton `Register` class for dependency injection
   - `config.py`: `AbstractConfig` base for configuration objects
   - `singleton_decorator.py`: Singleton pattern implementation

2. **Encapsulation Layer** (`encapsulation/`): Database and model abstractions
   - `database/`: Abstraction over PostgreSQL, Redis, Neo4j, FAISS, Tantivy
   - `llm/`: LLM provider abstractions (OpenAI, HuggingFace)
   - `data_model/`: Pydantic schemas (Chunk, Document, Entity, Relation)

3. **Core Layer** (`core/`): Pure algorithms and business logic
   - `file_management/`: Parsing (native, OCR), chunking, indexing
   - `retrieval/`: Dense (FAISS), sparse (BM25/Tantivy), graph (HippoRAG)
   - `rerank/`: LLM-based and listwise rerankers
   - `query_rewrite/`: LLM-based query rewriting

4. **Application Layer** (`application/`): High-level workflows
   - `rag_inference/`: End-to-end RAG pipeline (rewrite → retrieve → rerank → LLM)
   - `knowledge/`: Document upload, indexing orchestration
   - `account/`: User, session, and chat message management

5. **API Layer** (`api/`): HTTP endpoints and MCP server
   - `routers/`: FastAPI routes (knowledge, rag_inference, auth, session, user)
   - `mcp/`: FastMCP server for Claude Desktop integration

6. **Config Layer** (`config/`): Configuration schemas mirroring the architecture
   - `application/`, `core/`, `encapsulation/`: Pydantic config classes
   - `json_configs/`: Runtime JSON config files with environment variable substitution

### Dependency Injection Pattern
- All components are registered at startup via `app_registration.py`
- `Register.register(config_path, app_name, config_type)` parses JSON, builds the module tree
- Components are retrieved with `Register.get_object(app_name)`
- Environment variables in JSON configs use `${VAR_NAME}` syntax (replaced by `Register._substitute_env_vars`)

### Configuration System
- **Dual profile system**: `MODEL_PROFILE=api` (default, uses OpenAI APIs) or `MODEL_PROFILE=local` (uses local HuggingFace models)
- Config paths resolved by `app_registration._resolve_config_path()` with env override support
- Default configs: `rag_inference.json` (API), `rag_inference_local.json` (local), same for `knowledge*.json`
- Override with env vars: `RAG_INFERENCE_CONFIG_PATH`, `KNOWLEDGE_CONFIG_PATH`, etc.

### Shared Resource Pattern
- **Singleton instances**: Tokenizer (`TokenizerManager`), database connections (via `@singleton` decorator)
- **Shared embedding models**: Embedding instances are reused across retrieval, indexing, and graph components to avoid redundant VRAM/RAM usage
- **Component reuse**: Retriever and reranker instances are built once at startup and cached in the `Register`

### Multi-Path Retrieval Architecture
Located in `core/retrieval/multipath.py`:
- Orchestrates 3 retrieval paths in parallel: Dense (FAISS), Sparse (Tantivy BM25), Graph (Neo4j HippoRAG)
- Fusion strategies in `core/utils/fusion.py`: RRF (Reciprocal Rank Fusion), weighted sum, rank fusion
- Graph retriever extracts subgraph metadata (`_subgraph_info`) which flows through the pipeline for visualization

### GraphRAG Implementation (Pruned HippoRAG)
Core files: `core/retrieval/graph_retrieveal/pruned_hipporag_neo4j.py`, `encapsulation/database/graph_db/pruned_hipporag_neo4j.py`

**Key innovations over baseline HippoRAG**:
- **Subgraph PPR**: Computes Personalized PageRank on query-relevant subgraphs instead of full graph
- **Query-aware pruning**: Dynamically adjusts neighbor retention during graph expansion based on entity relevance
- **Incremental updates**: Add entities/relations without full graph reconstruction
- **PPR backends**: "push" (fast, approximate) or "pull" (exact but slower) algorithms in `encapsulation/database/utils/ppr_push.py`

**Graph indexing flow**:
1. Extract entities/relations from chunks via `core/file_management/extractor/hipporag2_extractor.py`
2. Build graph index in Neo4j via `core/file_management/indexing/graph_indexing/pruned_hipporag_indexing.py`
3. Create synonym edges (optional) to connect semantically similar entities

**Graph retrieval flow**:
1. Extract entities from query using NER
2. Find matching entities in graph (exact + fuzzy + embedding similarity)
3. Expand subgraph around seed entities (1-2 hop neighbors)
4. Compute PPR scores on subgraph to rank facts
5. Optionally rerank facts with LLM
6. Map facts back to original chunks

### Document Processing Pipeline
`application/knowledge/module.py` orchestrates the full ingestion workflow:

1. **Upload**: Store file in PostgreSQL (or MinIO if configured)
2. **Parse**: `core/file_management/parser_combinator.py` routes to appropriate parser:
   - `native.py`: Direct PDF/DOCX/PPTX/XLSX parsing
   - `dots_ocr.py`: Layout-aware OCR with local models (DOTS framework)
   - `vlm_ocr.py`: Vision-Language Model based OCR (GPT-4o, Qwen-VL)
3. **Chunk**: `core/file_management/chunker/` splits text (token-based, semantic, recursive, markdown-header)
4. **Index**: `core/file_management/index_manager.py` dispatches to indexers:
   - `faiss_indexing.py`: Build FAISS vector index
   - `bm25_indexing.py`: Build Tantivy full-text index
   - `graph_indexing/pruned_hipporag_indexing.py`: Extract graph and build Neo4j index
5. **Store**: Persist chunks in PostgreSQL with owner isolation

### User Isolation & Multi-tenancy
- All chunks/files have `owner_id` (UUID) field
- Retrieval filters by `owner_id` at the database level (FAISS metadata filter, Neo4j WHERE clause, BM25 filter)
- Authentication via JWT tokens (`api/routers/auth.py`)
- Permission system supports VIEW/EDIT roles (`core/user_management/`)

## Configuration Knobs

### Model Provider Configuration
Each capability can independently use API or local models via `.env`:
- **Chat**: `CHAT_MODEL_PROVIDER=openai|huggingface`, `CHAT_MODEL_NAME`, `CHAT_API_KEY`
- **Embedding**: `EMBEDDING_MODEL_PROVIDER=openai|huggingface`, `EMBEDDING_MODEL_NAME`
- **OCR**: `OCR_MODEL_PROVIDER=openai|vllm|dots_ocr_parser`, `OCR_MODEL_NAME`
- **Reranker**: Auto-selected based on `MODEL_PROFILE` (API → listwise reranker using chat model, local → Qwen3-Reranker)

### Database Configuration
- **PostgreSQL**: Metadata, chunks, users, sessions (env: `POSTGRES_HOST`, `POSTGRES_PORT`, etc.)
- **Redis**: Caching layer (env: `REDIS_HOST`, `REDIS_PORT`)
- **Neo4j**: Knowledge graph (env: `NEO4J_URL`, `NEO4J_USERNAME`, `NEO4J_PASSWORD`)
- **FAISS**: Vector index stored at `./data/unified_faiss_index` (configurable in JSON)
- **Tantivy**: BM25 index at `./data/unified_bm25_index`

### Development Mode
- Set `DEVELOP_MODE=true` to expose all Docker service ports to `localhost` for debugging
- CLI uses `DEVELOP_OWNER_ID` (default UUID) and auto-creates a dev user
- Individual exposure flags: `EXPOSE_POSTGRES`, `EXPOSE_REDIS`, `EXPOSE_NEO4J`

### Rebuild Requirements
**⚠️ When using Docker**: If you change `MODEL_PROFILE` or any `*_MODEL_PROVIDER` in `.env`, you MUST rebuild:
```bash
./build.sh  # Rebuilds image with new dependencies
./start.sh  # Restart services
```
This is necessary because local model dependencies (torch, transformers, sentence-transformers) are bundled at build time.

## Important Implementation Notes

### Adding New Retrieval Methods
1. Create retriever class in `core/retrieval/` inheriting from `BaseRetriever`
2. Create config class in `config/core/retrieval/` inheriting from `AbstractConfig`
3. Register in `config/core/retrieval/multipath_config.py` retriever factory
4. Add to JSON config `retrievers` array with appropriate `type` field

### Adding New Parsers
1. Implement parser in `core/file_management/parser/` inheriting from `BaseParser`
2. Add config in `config/core/file_management/parser/`
3. Register in `core/file_management/parser_combinator.py`
4. Parser output goes to `{PARSER_OUTPUT_DIR}/{parser_name}/` (configurable via env)

### Adding New Chunking Strategies
1. Create chunker in `core/file_management/chunker/` inheriting from `BaseChunker`
2. Add config in `config/core/file_management/chunker/chunker_config.py`
3. Update `knowledge.json` to use new chunker type

### Graph Exporter for Visualization
- `encapsulation/database/utils/graph_export_utils.py`: Exports igraph/networkx subgraphs
- `encapsulation/database/utils/graph_export_utils_neo4j.py`: Exports Neo4j subgraphs
- Both produce JSON with nodes (id, name, type, embedding, ppr_score) and edges (source, target, relation, metadata)
- Used by `/rag_inference/chat?return_subgraph=true` API endpoint

### Subgraph Metadata Flow
1. Graph retriever attaches `_subgraph_info` to first chunk's metadata
2. `MultiPathRetriever` extracts and preserves it during fusion
3. `RAGInference.chat()` extracts before reranking (to avoid loss during reordering)
4. Exports subgraph using appropriate `GraphExporter` based on graph_store type
5. Returns to API layer for client visualization

### Environment Variable Substitution
- JSON configs support `${VAR_NAME}` placeholders
- `Register._substitute_env_vars()` recursively replaces at registration time
- Unset variables log warnings and keep placeholder (fail-safe behavior)

### CLI Owner ID Management
- CLI caches default owner UUID in `~/.rag_arc_owner_id`
- Override with `--owner-id UUID` flag or `CLI_OWNER_ID`/`RAG_ARC_OWNER_ID` env vars
- In `DEVELOP_MODE=true`, defaults to `DEVELOP_OWNER_ID` for reproducibility

## Common Patterns

### Reading Configuration
```python
from framework.register import Register
registrator = Register()
rag_module = registrator.get_object("rag_inference")
knowledge_module = registrator.get_object("knowledge")
```

### Adding a New API Endpoint
```python
# In api/routers/your_router.py
from fastapi import APIRouter, Depends
from api.routers.auth import get_current_user

router = APIRouter(prefix="/your_path", tags=["your_tag"])

@router.post("/action")
async def your_action(user=Depends(get_current_user)):
    module = registrator.get_object("your_module")
    return module.do_something(owner_id=user.id)

# In main.py
from api.routers import your_router
app.include_router(your_router.router)
```

### Invoking Retrieval Pipeline Programmatically
```python
rag = registrator.get_object("rag_inference")
response, chunks, subgraph_data = rag.chat(
    query="What is RAG?",
    owner_id=user_uuid,
    return_subgraph=True  # Optional: get graph visualization data
)
```

### Manual Graph Export
```python
from cli.rag import export_graph_cmd
export_graph_cmd(output="graph.json", owner_id=user_uuid, json_output=True)
```

## Debugging Tips

- **Enable detailed logging**: Set `LOG_LEVEL=DEBUG` in `.env`
- **Check component registration**: Failures appear in startup logs from `app_registration.py`
- **Inspect graph in Neo4j Browser**: Set `EXPOSE_NEO4J=true`, visit http://localhost:7474
- **Test retrieval without LLM**: Use `uv run rag-arc pipeline "query" --skip-llm --json`
- **Verify indexing status**: `uv run rag-arc list-files --json` shows chunk counts and indexing state
- **PostgreSQL inspection**: Set `EXPOSE_POSTGRES=true`, connect with `psql -h localhost -p 5555 -U postgres -d rag_arc`
