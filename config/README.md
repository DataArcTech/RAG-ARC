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

## DeepSearch Web Search Policy

In `config/json_configs/deepsearch_service.json`:
- `planner.web_step_policy="realtime_required"` injects/forces at least one `channel="web"` step when the question asks for realtime/latest/current info (e.g. FX rates/news).
- `external_channel.execute_forced_tasks_without_gap=true` ensures those forced tasks execute even when gap detection thinks coverage is sufficient.

## DeepSearch Tool Budget

In `config/json_configs/deepsearch_service.json`:
- `tool_budget.max_calls_total` caps total tool invocations per DeepSearch run (tool_manager + optional external calls; does not count graph adapter traversals).
- The remaining budget is attached to `graph_context.metadata.tool_budget` for LLM visibility and is also surfaced in tool diagnostics.
