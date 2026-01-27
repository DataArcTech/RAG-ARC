## Intent Routing (Semantic Router)

This package provides session-scoped *intent routing* for RAG-ARC, implemented with the third-party
[`semantic-router`](https://github.com/aurelio-labs/semantic-router) library.

Intent routing sits at the beginning of the chat pipeline and decides:
- the **intent** label (stable string), and
- the **action**: `rag` | `web_only` | `no_retrieval`

Key design goals:
- **Config-driven**: intents/utterances/thresholds live in `config/core/intent_routing/intent_router.toml`.
- **Decoupled embeddings**: intent embeddings can use a different model/provider than RAG embeddings.
- **Multi-turn aware**: a lightweight topic stack (derived state) tracks topic switch / return-to-topic.
- **No prompt scattering**: routing-intent-specific prompts are centralized in `core/prompts/intent_routing_prompts.py`.

### Where It Runs In The Pipeline

RAG inference calls `application.intent_routing.IntentRoutingService.route(...)`:
- if `action == no_retrieval`: skip retrieval and do not invoke the query rewrite model (use the original query); respond quickly using conversation-only messages.
- if `action == rag`: run internal retrieval/rerank (and rewrite).
- if `action == web_only`: run web search path (and rewrite); web is gated by the request flag.

### Intent Set (Current)

Configured in `config/core/intent_routing/intent_router.toml`:

1) `RAG_REQUIRED` (`rag`)
- “Please answer based on my uploaded document and cite sources.”
- “Compare A and B and explain the differences with evidence.”

2) `FOLLOWUP_RAG_REQUIRED` (`rag`, requires history)
- “Back to the previous topic, please find more evidence from the document.”
- “Continue searching the materials and provide more exact clauses/numbers.”

3) `WEB_ONLY` (`web_only`)
- “Search online for the latest announcement and provide links.”
- “What’s the weather today?”

4) `FOLLOWUP_NO_RAG` (`no_retrieval`, requires history)
- “Please elaborate on what you just said.”
- “Continue the previous topic and give more explanation.”

5) `NO_RETRIEVAL` (`no_retrieval`)
- “Thanks.”
- “Just answer directly, no need to look up materials.”

6) `TASK_EXECUTION` (`no_retrieval`)
- “Write a SQL CREATE TABLE statement for users(...).”
- “Rewrite the following paragraph to be clearer: ...”

7) `CLARIFY_REQUIRED` (`no_retrieval`)
- “What does this mean?” (without enough context)
- “Help me with this.” (missing key details)

8) `ANSWER_DISSATISFIED` (`no_retrieval`, requires history)
- “That’s not correct. You didn’t answer my question.”
- “Too vague. Please be more specific.”

Notes:
- `requires_history=true` intents are excluded on the first user turn via semantic-router’s official `route_filter`.
- Web search is frontend-controlled (`enable_web_search`). When it is `false`, `WEB_ONLY` is excluded via `route_filter`
  (capability gating); if it somehow still appears, the action is treated as normal `rag`.
- `ANSWER_DISSATISFIED` forces `same_topic` in topic selection to avoid spurious topic switches.

### Multi-turn Topic Handling (Session Only)

Topic tracking is implemented as derived state:
- Stored in Redis (TTL) under the session id.
- Rebuilt from the latest `max_user_messages` user turns on cache miss.
- Uses cosine similarity between topic centroids to decide:
  - `same_topic`
  - `topic_switch`
  - `return_to_topic`

Config knobs live in `config/core/intent_routing/intent_router.toml` under `[intent_router.topic_stack]`.

### How To Add A New Intent (Recommended Workflow)

1) Add a new intent route in `config/core/intent_routing/intent_router.toml`:
- pick a stable `name` (ALL_CAPS),
- set `action` (`rag` / `web_only` / `no_retrieval`),
- add 10–30 diverse `utterances` (domain-agnostic where possible),
- set a starting `threshold` (e.g. `0.50`), and `requires_history=true` if it is a follow-up-only intent.

2) Add labeled examples to `config/core/intent_routing/intent_router_eval.toml`.

3) Fit/evaluate thresholds using semantic-router’s official APIs:
- `uv run python scripts/intent_routing/fit_intent_router_thresholds.py`

4) Add/adjust routing-intent behavior prompts (if needed):
- `core/prompts/intent_routing_prompts.py`

5) Add tests:
- single-session: `bash test/e2e/intent_routing/e2e_intent_router_qwen_local.sh`
- concurrency (multi-session): `INTENT_CONCURRENCY_USERS=16 INTENT_CONCURRENCY_WORKERS=8 bash test/e2e/intent_routing/e2e_intent_router_concurrency_qwen_local.sh`
- full pipeline smoke: `bash test/e2e/rag/e2e_rag_pipeline_test_pdf.sh`

### Local Intent Embedding (Qwen) Env Vars

Intent embedding is configured via TOML placeholders and env vars (examples in `.env.example`):
- `INTENT_QWEN_EMBEDDING_MODEL_NAME` (e.g. `Qwen/Qwen3-Embedding-0.6B`)
- `INTENT_EMBEDDING_DEVICE` (`cpu` / `cuda`)
- `INTENT_EMBEDDING_CACHE_FOLDER` (e.g. `./models/Qwen`)
