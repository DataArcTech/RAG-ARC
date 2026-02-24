# Configuration Guide

RAG-ARC uses a single-source-of-truth configuration flow:

- **Runtime secrets / deployment-specific values** live in environment variables (`.env`, see `.env.example`).
- **Tunable knobs** (thresholds, budgets, tool selection, paths, feature gates) live in JSON under `config/json_configs/`.
- Global JSON-output retry defaults for LLM calls live in `config/core/llm_json_retry_defaults.py` (env overrides available; see `config/env-*.md`).

## Entry Points

- DeepSearch service: `config/json_configs/deepsearch_service.json`
- RAG inference (HippoRAG Q&A): `config/json_configs/rag_inference.json`
- Knowledge pipelines: `config/json_configs/knowledge.json`

`MODEL_PROFILE` (see `.env.example`) controls which profile the app loads by default (e.g. `api` vs `local`).

## Environment Variables

Full env reference:
- English: `config/env-en.md`
- 中文: `config/env-zh.md`

Notes:
- `OPENAI_EMBEDDING_MODEL` takes precedence over `EMBEDDING_MODEL_NAME` when `EMBEDDING_MODEL_PROVIDER=openai`.
- DeepSearch web search requires `TAVILY_API_KEY` and enabling the external channel in `config/json_configs/deepsearch_service.json`.
- For RAG inference, `retrieval_config.search_kwargs.k` (pre-rerank candidates) should generally be > `candidate_selection.rerank_keep_k` (Sources passed to LLM) to keep multi-aspect questions from collapsing to a single file.

## Virtual Paths (`io://...`) vs Local Paths

Many path-like settings accept either:
- an `io://...` virtual path (preferred for portability; routed through IOManager and mapped to LocalDB/MinIO), or
- a local filesystem path (useful for hermetic unit tests and one-off scripts).

Examples (see `config/env-*.md` for the canonical list):
- Parser output dirs: `PARSER_OUTPUT_DIR`, `NATIVE_PARSER_OUTPUT_DIR`, `MINERU_SHARED_CACHE_DIR`
- DeepSearch artifacts: `DEEPSEARCH_TOOL_ARTIFACT_DIR`
- MQ external result store: `MQ_RESULT_STORE`, `MQ_RESULT_LOCAL_DIR` (set `MQ_RESULT_STORE=local` to avoid IOManager dependencies)
