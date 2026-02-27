"""TOML-driven semantic intent routing configuration.

This module centralizes loading and validation of the intent router config. The router
is used to decide whether to run RAG retrieval, web-only retrieval, or no retrieval.
"""
import os
import re
import tomllib
from dataclasses import dataclass
from typing import Any, Literal, Optional

from config.env_placeholder_policy import ENV_DEFAULTS


_PLACEHOLDER_RE = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _default_config_path() -> str:
    # Keep the default in one place; can be overridden by INTENT_ROUTER_CONFIG_PATH.
    return os.getenv("INTENT_ROUTER_CONFIG_PATH", "config/core/intent_routing/intent_router.toml")


def _substitute_placeholders(text: str) -> str:
    """Replace ${ENV_VAR} placeholders using env vars or ENV_DEFAULTS.

    Missing values are substituted with an empty string; validation happens later.
    """

    def _replace(match: re.Match[str]) -> str:
        name = match.group(1)
        raw = os.getenv(name)
        if raw is not None and str(raw).strip():
            return str(raw)
        if name in ENV_DEFAULTS:
            return str(ENV_DEFAULTS[name])
        return ""

    return _PLACEHOLDER_RE.sub(_replace, text)


def _coerce_optional_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip()
    return s or None


@dataclass(frozen=True)
class IntentRouteConfig:
    name: str
    action: Literal["rag", "web_only", "no_retrieval"]
    threshold: float
    utterances: list[str]
    # If true, this intent should only be considered when there is prior user history.
    # Used to drive semantic-router's `route_filter` (best practice).
    requires_history: bool = False


@dataclass(frozen=True)
class TopicStackConfig:
    enabled: bool
    redis_ttl_seconds: int
    max_user_messages: int
    max_topics: int
    same_topic_threshold: float
    return_to_topic_threshold: float
    sample_keep: int
    history_mode: Literal["topic_scoped", "full"]


@dataclass(frozen=True)
class IntentEmbeddingConfig:
    provider: Literal["qwen_local", "openai_compat"]
    qwen_model_name: Optional[str]
    qwen_device: Optional[str]
    qwen_cache_folder: Optional[str]
    qwen_use_china_mirror: bool
    # Passed to SentenceTransformer(...) as **kwargs (advanced tuning).
    qwen_model_kwargs: dict[str, Any]
    # Passed to SentenceTransformer(..., model_kwargs=...) (Transformers model kwargs).
    qwen_sentence_transformer_model_kwargs: dict[str, Any]
    # Passed to SentenceTransformer.encode(...) as **kwargs (batch size, normalization, etc).
    qwen_encode_kwargs: dict[str, Any]
    openai_model_name: Optional[str]
    openai_api_key: Optional[str]
    openai_base_url: Optional[str]
    openai_timeout_seconds: float
    openai_max_retries: int
    openai_request_batch_size: int
    openai_supports_batch_input: bool


@dataclass(frozen=True)
class IntentRouterConfig:
    enabled: bool
    default_intent: str
    top_k: int
    aggregation: Literal["mean", "max", "sum"]
    topic_stack: TopicStackConfig
    embedding: IntentEmbeddingConfig
    intents: list[IntentRouteConfig]


def load_intent_router_config(path: str | None = None) -> IntentRouterConfig:
    """Load intent router TOML and validate the minimal shape."""
    path = path or _default_config_path()
    with open(path, "rb") as f:
        raw_text = f.read().decode("utf-8", errors="strict")
    substituted = _substitute_placeholders(raw_text)
    payload = tomllib.loads(substituted)

    router = payload.get("intent_router") or {}
    enabled = bool(router.get("enabled", True))
    default_intent = str(router.get("default_intent") or "RAG_REQUIRED").strip() or "RAG_REQUIRED"
    top_k = int(router.get("top_k") or 5)
    aggregation = str(router.get("aggregation") or "mean").strip().lower() or "mean"
    if aggregation not in {"mean", "max", "sum"}:
        raise ValueError(f"Invalid intent_router.aggregation: {aggregation}")

    # topic stack
    ts = router.get("topic_stack") or {}
    topic_stack = TopicStackConfig(
        enabled=bool(ts.get("enabled", True)),
        redis_ttl_seconds=int(ts.get("redis_ttl_seconds") or 259200),
        max_user_messages=int(ts.get("max_user_messages") or 10),
        max_topics=int(ts.get("max_topics") or 6),
        same_topic_threshold=float(ts.get("same_topic_threshold") or 0.82),
        return_to_topic_threshold=float(ts.get("return_to_topic_threshold") or 0.85),
        sample_keep=int(ts.get("sample_keep") or 3),
        history_mode=str(ts.get("history_mode") or "topic_scoped"),
    )
    if topic_stack.history_mode not in {"topic_scoped", "full"}:
        raise ValueError(f"Invalid intent_router.topic_stack.history_mode: {topic_stack.history_mode}")
    if topic_stack.redis_ttl_seconds <= 0:
        raise ValueError("intent_router.topic_stack.redis_ttl_seconds must be > 0")
    if topic_stack.max_user_messages <= 0:
        raise ValueError("intent_router.topic_stack.max_user_messages must be > 0")
    if topic_stack.sample_keep < 0:
        raise ValueError("intent_router.topic_stack.sample_keep must be >= 0")

    # embedding config
    emb = router.get("embedding") or {}
    provider = str(emb.get("provider") or "qwen_local").strip()

    # Env override: INTENT_EMBEDDING_PROVIDER (api|local) takes precedence over TOML.
    # Default is "api" — reuses RAG embedding credentials. Set "local" to use downloaded Qwen model.
    _env_provider = os.getenv("INTENT_EMBEDDING_PROVIDER", "api").strip().lower()
    if _env_provider in {"api", "openai_compat"}:
        provider = "openai_compat"
    elif _env_provider in {"local", "qwen_local"}:
        provider = "qwen_local"
    elif _env_provider:
        raise ValueError(f"INTENT_EMBEDDING_PROVIDER must be api|local, got: {_env_provider}")

    if provider not in {"qwen_local", "openai_compat"}:
        raise ValueError(f"Invalid intent_router.embedding.provider: {provider}")

    qwen = emb.get("qwen_local") or {}
    oai = emb.get("openai_compat") or {}
    qwen_model_kwargs = qwen.get("model_kwargs") or {}
    if not isinstance(qwen_model_kwargs, dict):
        qwen_model_kwargs = {}
    qwen_st_model_kwargs = qwen.get("sentence_transformer_model_kwargs") or {}
    if not isinstance(qwen_st_model_kwargs, dict):
        qwen_st_model_kwargs = {}
    qwen_encode_kwargs = qwen.get("encode_kwargs") or {}
    if not isinstance(qwen_encode_kwargs, dict):
        qwen_encode_kwargs = {}

    # When provider is openai_compat, apply credential fallback chain so CPU servers
    # only need INTENT_EMBEDDING_PROVIDER=api (credentials inherited from RAG embedding / main API).
    oai_api_key = _coerce_optional_str(oai.get("api_key"))
    oai_base_url = _coerce_optional_str(oai.get("base_url"))
    oai_model_name = _coerce_optional_str(oai.get("model_name"))
    if provider == "openai_compat":
        if not oai_api_key:
            oai_api_key = _coerce_optional_str(os.getenv("EMBEDDING_API_KEY")) or _coerce_optional_str(os.getenv("OPENAI_API_KEY"))
        if not oai_base_url:
            oai_base_url = _coerce_optional_str(os.getenv("EMBEDDING_API_BASE_URL")) or _coerce_optional_str(os.getenv("OPENAI_BASE_URL"))
        if not oai_model_name:
            oai_model_name = _coerce_optional_str(os.getenv("OPENAI_EMBEDDING_MODEL"))

    embedding = IntentEmbeddingConfig(
        provider=provider,  # type: ignore[arg-type]
        qwen_model_name=_coerce_optional_str(qwen.get("model_name")),
        qwen_device=_coerce_optional_str(qwen.get("device")),
        qwen_cache_folder=_coerce_optional_str(qwen.get("cache_folder")),
        qwen_use_china_mirror=bool(qwen.get("use_china_mirror", False)),
        qwen_model_kwargs=qwen_model_kwargs,
        qwen_sentence_transformer_model_kwargs=qwen_st_model_kwargs,
        qwen_encode_kwargs=qwen_encode_kwargs,
        openai_model_name=oai_model_name,
        openai_api_key=oai_api_key,
        openai_base_url=oai_base_url,
        openai_timeout_seconds=float(oai.get("timeout_seconds") or 20.0),
        openai_max_retries=int(oai.get("max_retries") or 0),
        openai_request_batch_size=int(oai.get("request_batch_size") or 64),
        openai_supports_batch_input=bool(oai.get("supports_batch_input", True)),
    )

    routes_raw = router.get("intents") or []
    if not isinstance(routes_raw, list) or not routes_raw:
        raise ValueError("intent_router.intents must be a non-empty list")
    intents: list[IntentRouteConfig] = []
    for item in routes_raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").strip()
        action = str(item.get("action") or "").strip()
        threshold = float(item.get("threshold") or 0.0)
        utterances = item.get("utterances") or []
        requires_history = bool(item.get("requires_history", False))
        if not name:
            raise ValueError("intent_router.intents[].name is required")
        if action not in {"rag", "web_only", "no_retrieval"}:
            raise ValueError(f"intent_router.intents[{name}].action invalid: {action}")
        if threshold < 0:
            raise ValueError(f"intent_router.intents[{name}].threshold must be >= 0")
        if not isinstance(utterances, list) or not [u for u in utterances if str(u).strip()]:
            raise ValueError(f"intent_router.intents[{name}].utterances must be non-empty")
        intents.append(
            IntentRouteConfig(
                name=name,
                action=action,  # type: ignore[arg-type]
                threshold=threshold,
                utterances=[str(u).strip() for u in utterances if str(u).strip()],
                requires_history=requires_history,
            )
        )

    # Default intent must exist.
    if default_intent not in {r.name for r in intents}:
        raise ValueError(f"intent_router.default_intent={default_intent} not found in intents")

    return IntentRouterConfig(
        enabled=enabled,
        default_intent=default_intent,
        top_k=top_k,
        aggregation=aggregation,  # type: ignore[arg-type]
        topic_stack=topic_stack,
        embedding=embedding,
        intents=intents,
    )


__all__ = ["IntentRouterConfig", "IntentRouteConfig", "TopicStackConfig", "IntentEmbeddingConfig", "load_intent_router_config"]
