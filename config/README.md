# Configuration Guide

RAG-ARC uses a single-source-of-truth configuration flow:

- **Runtime secrets / deployment-specific values** live in environment variables (`.env`, see `.env.example`).
- **Tunable knobs** (thresholds, budgets, tool selection, paths, feature gates) live in JSON under `config/json_configs/`.

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
