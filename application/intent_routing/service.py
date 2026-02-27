import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from config.encapsulation.database.cache_db.redis_config import RedisConfig
from config.core.intent_routing.intent_router import IntentRouteConfig, IntentRouterConfig, load_intent_router_config
from encapsulation.database.cache_db.redis_db import RedisDB

from application.intent_routing.embedding import IntentEmbeddingProviderConfig, build_intent_embedding_provider
from core.intent_routing.semantic_router import IntentDecision, IntentRoute, SemanticIntentRouter
from application.intent_routing.topic_stack import TopicSelection, TopicStackManager, TopicStackSettings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntentRoutingResult:
    intent: str
    action: str  # "rag" | "web_only" | "no_retrieval"
    score: float
    scoped_history_text: str | None
    topic: TopicSelection | None
    debug: dict[str, Any]


def _parse_history_text(history_text: str | None) -> list[tuple[str, str]]:
    """Parse `role: content` lines into (role, content) messages, chronological order."""
    if not history_text or not str(history_text).strip():
        return []
    out: list[tuple[str, str]] = []
    for raw in str(history_text).splitlines():
        if ":" not in raw:
            continue
        role, content = raw.split(":", 1)
        role = role.strip()
        content = content.strip()
        if role in {"user", "assistant"} and content:
            out.append((role, content))
    return out


def _extract_last_user_messages(messages: list[tuple[str, str]], *, max_n: int, exclude_text: str | None) -> list[str]:
    """Return last N user messages (chronological)."""
    users = [content for role, content in messages if role == "user" and content]
    if exclude_text:
        last = users[-1] if users else None
        if last is not None and last.strip() == str(exclude_text).strip():
            users = users[:-1]
    if max_n <= 0:
        return []
    return users[-max_n:]


def _extract_last_qa_pair(messages: list[tuple[str, str]], *, exclude_text: str | None) -> str:
    """Return the last user+assistant exchange as bridge context for topic switches.

    When the user switches topics, the new topic's scoped history is empty.
    Including the last Q&A pair from the previous topic helps the rewriter
    disambiguate referential queries (e.g., "那B的呢？" after discussing A's features).
    """
    # Walk backward to find the last user message (excluding current query).
    last_user_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        role, content = messages[i]
        if role == "user" and content:
            if exclude_text and content.strip() == str(exclude_text).strip():
                continue
            last_user_idx = i
            break
    if last_user_idx < 0:
        return ""
    # Collect the user message and any assistant response that follows.
    lines: list[str] = [f"user: {messages[last_user_idx][1]}"]
    for j in range(last_user_idx + 1, len(messages)):
        role, content = messages[j]
        if role == "user":
            break
        if role == "assistant" and content:
            lines.append(f"assistant: {content}")
    return "\n".join(lines)


def _has_prior_user_history(messages: list[tuple[str, str]], *, current_user_text: str | None) -> bool:
    """True if there exists a user message before the current one.

    This is used to gate follow-up intents: without prior history we should not
    consider follow-up routes (semantic-router's `route_filter`).
    """
    users = [content for role, content in messages if role == "user" and content]
    if not users:
        return False
    if current_user_text and users[-1].strip() == str(current_user_text).strip():
        users = users[:-1]
    return bool(users)


def _topic_scoped_history(
    messages: list[tuple[str, str]],
    *,
    last_user_indices: list[int],
    assigned_topic_ids: list[str],
    target_topic_id: str,
) -> str:
    """Keep only turns whose user message belongs to target_topic_id."""
    keep_user_idx = {idx for idx, tid in zip(last_user_indices, assigned_topic_ids) if tid == target_topic_id}
    if not keep_user_idx:
        return ""

    out_lines: list[str] = []
    i = 0
    while i < len(messages):
        role, content = messages[i]
        if role != "user":
            i += 1
            continue
        if i in keep_user_idx:
            # keep this user and subsequent assistant messages until next user
            out_lines.append(f"user: {content}")
            j = i + 1
            while j < len(messages) and messages[j][0] != "user":
                out_lines.append(f"{messages[j][0]}: {messages[j][1]}")
                j += 1
            i = j
        else:
            i += 1
    return "\n".join(out_lines)


class IntentRoutingService:
    def __init__(self, *, config_path: str | None = None) -> None:
        cfg = load_intent_router_config(config_path)
        self._cfg = cfg
        self._router: SemanticIntentRouter | None = None
        self._embedder = None
        self._topic_mgr: TopicStackManager | None = None

        if not cfg.enabled:
            # Keep the service lightweight when disabled.
            return

        # Build embedding provider (decoupled from RAG embedding config/env).
        emb_cfg = self._build_embedding_cfg(cfg.embedding)
        self._embedder = build_intent_embedding_provider(emb_cfg)

        routes = [self._to_route(r) for r in cfg.intents]
        self._router = SemanticIntentRouter(
            embedding_provider=self._embedder,
            routes=routes,
            default_intent=cfg.default_intent,
            top_k=cfg.top_k,
            aggregation=cfg.aggregation,
        )

        try:
            redis = RedisDB(RedisConfig())  # singleton; config from env
            self._topic_mgr = TopicStackManager(
                redis=redis,
                settings=TopicStackSettings(
                    enabled=cfg.topic_stack.enabled,
                    redis_ttl_seconds=cfg.topic_stack.redis_ttl_seconds,
                    max_user_messages=cfg.topic_stack.max_user_messages,
                    max_topics=cfg.topic_stack.max_topics,
                    same_topic_threshold=cfg.topic_stack.same_topic_threshold,
                    return_to_topic_threshold=cfg.topic_stack.return_to_topic_threshold,
                    history_mode=cfg.topic_stack.history_mode,
                    sample_keep=cfg.topic_stack.sample_keep,
                ),
            )
        except Exception as exc:  # noqa: BLE001
            # Redis is an optimization for session-scoped derived state; intent routing still works without it.
            logger.warning("IntentRouting: failed to init Redis topic stack (disabled): %s", exc)

    @staticmethod
    def _to_route(r: IntentRouteConfig) -> IntentRoute:
        return IntentRoute(name=r.name, action=r.action, threshold=float(r.threshold), utterances=list(r.utterances))

    @staticmethod
    def _build_embedding_cfg(emb) -> IntentEmbeddingProviderConfig:
        provider = emb.provider
        # NOTE: validation for required fields happens here (not in TOML substitution).
        if provider == "qwen_local":
            if not (emb.qwen_model_name and emb.qwen_model_name.strip()):
                raise ValueError("Intent embedding provider qwen_local requires a non-empty model_name in TOML.")
            model_name = emb.qwen_model_name
            return IntentEmbeddingProviderConfig(
                provider="qwen_local",
                qwen_model_name=model_name,
                qwen_device=emb.qwen_device,
                qwen_cache_folder=emb.qwen_cache_folder,
                qwen_use_china_mirror=bool(emb.qwen_use_china_mirror),
                qwen_model_kwargs=dict(getattr(emb, "qwen_model_kwargs", None) or {}),
                qwen_sentence_transformer_model_kwargs=dict(
                    getattr(emb, "qwen_sentence_transformer_model_kwargs", None) or {}
                ),
                qwen_encode_kwargs=dict(getattr(emb, "qwen_encode_kwargs", None) or {}),
                openai_model_name="",
                openai_api_key="",
                openai_base_url="",
                openai_timeout_seconds=20.0,
                openai_max_retries=0,
                openai_request_batch_size=64,
                openai_supports_batch_input=True,
            )
        if provider == "openai_compat":
            if not (emb.openai_api_key and emb.openai_api_key.strip()):
                raise ValueError("Intent embedding provider openai_compat requires INTENT_EMBEDDING_API_KEY (or TOML api_key).")
            if not (emb.openai_base_url and emb.openai_base_url.strip()):
                raise ValueError("Intent embedding provider openai_compat requires INTENT_EMBEDDING_API_BASE_URL (or TOML base_url).")
            if not (emb.openai_model_name and emb.openai_model_name.strip()):
                raise ValueError("Intent embedding provider openai_compat requires a non-empty model_name (INTENT_OPENAI_EMBEDDING_MODEL or TOML model_name).")
            model_name = emb.openai_model_name
            return IntentEmbeddingProviderConfig(
                provider="openai_compat",
                qwen_model_name="",
                qwen_device=None,
                qwen_cache_folder=None,
                qwen_use_china_mirror=False,
                qwen_model_kwargs={},
                qwen_sentence_transformer_model_kwargs={},
                qwen_encode_kwargs={},
                openai_model_name=model_name,
                openai_api_key=emb.openai_api_key,
                openai_base_url=emb.openai_base_url,
                openai_timeout_seconds=float(emb.openai_timeout_seconds),
                openai_max_retries=int(emb.openai_max_retries),
                openai_request_batch_size=int(emb.openai_request_batch_size),
                openai_supports_batch_input=bool(emb.openai_supports_batch_input),
            )
        raise ValueError(f"Unsupported provider: {provider}")

    def route(
        self,
        *,
        session_id: str | None,
        user_query: str,
        history_text: str | None,
        enable_web_search: bool,
    ) -> IntentRoutingResult:
        if not self._cfg.enabled:
            return IntentRoutingResult(
                intent=self._cfg.default_intent,
                action="rag",
                score=0.0,
                scoped_history_text=history_text,
                topic=None,
                debug={"disabled": True},
            )

        start = time.perf_counter()
        if self._router is None or self._embedder is None:
            raise RuntimeError("IntentRoutingService is enabled but not initialized")

        parsed = _parse_history_text(history_text)
        scoped_history = history_text
        topic_sel: TopicSelection | None = None

        # Topic stack + topic-scoped history is session-only.
        #
        # Pre-pass before intent classification:
        # - ensure cached topic state exists (rebuild if needed)
        # - optionally scope history by the *current* topic from cached state
        #
        # We then apply intent-specific update rules after classification (e.g. ANSWER_DISSATISFIED forces same_topic).
        last_users: list[str] | None = None
        last_embeds: list[list[float]] | None = None
        cur_embed: list[float] | None = None
        topic_state: dict[str, Any] | None = None
        if session_id and self._topic_mgr is not None and self._cfg.topic_stack.enabled and parsed:
            max_n = int(self._cfg.topic_stack.max_user_messages)
            last_users = _extract_last_user_messages(parsed, max_n=max_n, exclude_text=user_query)
            if last_users:
                embeds = self._embedder.embed(last_users + [user_query])
                last_embeds = embeds[:-1]
                cur_embed = embeds[-1]

                topic_state = self._topic_mgr.load(str(session_id))
                if topic_state is None:
                    topic_state = self._topic_mgr.rebuild_from_user_messages(
                        session_id=str(session_id),
                        user_messages=last_users,
                        embeddings=last_embeds,
                    )
                    self._topic_mgr.save(str(session_id), topic_state)

                if self._cfg.topic_stack.history_mode == "topic_scoped":
                    current_topic = str(topic_state.get("current_topic_id") or "topic_0")
                    _topics, assigned = self._topic_mgr.assign_topics(user_messages=last_users, embeddings=last_embeds)
                    user_indices = [i for i, (role, _) in enumerate(parsed) if role == "user"]
                    last_user_indices = (
                        user_indices[-len(last_users) :] if len(user_indices) >= len(last_users) else user_indices
                    )
                    scoped = _topic_scoped_history(
                        parsed,
                        last_user_indices=last_user_indices,
                        assigned_topic_ids=assigned[-len(last_user_indices) :],
                        target_topic_id=current_topic,
                    )
                    scoped_history = scoped if scoped.strip() else (history_text or "")
        # Without prior user history, do NOT consider follow-up routes.
        # This is semantic-router's official `route_filter` best practice.
        route_filter: list[str] | None = None
        allow_followups = _has_prior_user_history(parsed, current_user_text=user_query)
        if not allow_followups:
            route_filter = [r.name for r in self._cfg.intents if not bool(getattr(r, "requires_history", False))]
            if not route_filter:
                route_filter = None

        # Business-capability gating: if web search is disabled for this request, do not consider WEB_ONLY routes.
        # This uses semantic-router's official `route_filter` feature (not a custom heuristic), and aligns with:
        # - "enable_web_search is controlled by the frontend"
        # - if WEB_ONLY appears when disabled, we should follow normal RAG instead.
        if not enable_web_search:
            if route_filter is None:
                route_filter = [r.name for r in self._cfg.intents if str(r.action) != "web_only"]
            else:
                route_filter = [name for name in route_filter if name != "WEB_ONLY"]
            if not route_filter:
                route_filter = None

        decision: IntentDecision = self._router.classify(user_query, route_filter=route_filter)

        # Apply intent-specific topic update rules after classification.
        if (
            session_id
            and self._topic_mgr is not None
            and self._cfg.topic_stack.enabled
            and parsed
            and last_users
            and last_embeds is not None
            and cur_embed is not None
            and topic_state is not None
        ):
            if str(decision.intent).upper() == "ANSWER_DISSATISFIED":
                # Refresh TTL without mutating topics using the "feedback" message.
                self._topic_mgr.save(str(session_id), topic_state)
                current_topic = str(topic_state.get("current_topic_id") or "topic_0")
                topic_sel = TopicSelection(topic_id=current_topic, action="same_topic", similarity=1.0)
            else:
                topic_state, topic_sel = self._topic_mgr.update_with_current_user_message(
                    session_id=str(session_id),
                    state=topic_state,
                    user_message=user_query,
                    embedding=cur_embed,
                )

            if self._cfg.topic_stack.history_mode == "topic_scoped" and topic_sel is not None:
                _topics, assigned = self._topic_mgr.assign_topics(user_messages=last_users, embeddings=last_embeds)
                user_indices = [i for i, (role, _) in enumerate(parsed) if role == "user"]
                last_user_indices = (
                    user_indices[-len(last_users) :] if len(user_indices) >= len(last_users) else user_indices
                )
                scoped_history = _topic_scoped_history(
                    parsed,
                    last_user_indices=last_user_indices,
                    assigned_topic_ids=assigned[-len(last_user_indices) :],
                    target_topic_id=topic_sel.topic_id,
                )
                if not scoped_history.strip():
                    if topic_sel.action == "topic_switch":
                        # Bridge context: include the last Q&A pair from the previous topic
                        # so the rewriter can disambiguate referential queries (e.g., "那B的呢？").
                        bridge = _extract_last_qa_pair(parsed, exclude_text=user_query)
                        scoped_history = bridge if bridge else (history_text or "")
                    else:
                        scoped_history = history_text or ""
        # If WEB_ONLY is chosen but the caller did not enable web, treat as normal RAG (per decision).
        final_action = decision.action
        if final_action == "web_only" and not enable_web_search:
            final_action = "rag"

        debug = {
            "intent": decision.intent,
            "raw_action": decision.action,
            "final_action": final_action,
            "score": decision.score,
            "top_matches": [
                {"intent": m.intent, "action": m.action, "score": m.score, "utterance": m.utterance}
                for m in decision.top_matches
            ],
            "duration_ms": int((time.perf_counter() - start) * 1000),
            **({"topic": {"topic_id": topic_sel.topic_id, "action": topic_sel.action, "similarity": topic_sel.similarity}} if topic_sel else {}),
        }
        logger.info("IntentRouting: %s", debug)
        return IntentRoutingResult(
            intent=decision.intent,
            action=final_action,
            score=float(decision.score),
            scoped_history_text=scoped_history if scoped_history else None,
            topic=topic_sel,
            debug=debug,
        )
