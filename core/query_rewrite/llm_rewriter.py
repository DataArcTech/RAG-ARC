from typing import Dict, Any, TYPE_CHECKING
import re
from .base import AbstractQueryRewriter
from core.prompts.query_rewrite_prompt import (
    QUERY_REWRITE_USER_PROMPT,
    QUERY_REWRITE_USER_PROMPT_WITH_HISTORY,
    QUERY_REWRITE_AND_ROUTING_USER_PROMPT,
    QUERY_REWRITE_AND_ROUTING_USER_PROMPT_WITH_HISTORY,
    QUERY_REWRITE_ROUTING_SYSTEM_SUFFIX,
    QUERY_REWRITE_INTENT_SYSTEM_SUFFIX,
    QUERY_REWRITE_AND_INTENT_USER_PROMPT,
    QUERY_REWRITE_AND_INTENT_USER_PROMPT_WITH_HISTORY,
)
from config.benchmark_mode import benchmark_mode_enabled
from config.retrieval_routing import rag_retrieval_dynamic_quota_enabled
from config.rag_intent_routing import (
    rag_intent_routing_enabled,
    rag_rewrite_history_user_only,
    rag_rewrite_history_most_recent_first,
)

import logging

if TYPE_CHECKING:
    from config.core.query_rewrite_config import LLMQueryRewriterConfig

logger = logging.getLogger(__name__)

_ALLOWED_INTENTS = {"RAG_REQUIRED", "CLARIFICATION", "CORRECTION", "CHITCHAT_ACK", "TOPIC_SWITCH"}


def _history_for_rewrite(history_text: str | None, *, max_chars: int) -> str | None:
    if not history_text:
        return None
    try:
        from core.conversation.history import parse_role_prefixed_history, build_history_text

        msgs = parse_role_prefixed_history(history_text)
        return build_history_text(
            msgs,
            user_only=rag_rewrite_history_user_only(),
            most_recent_first=rag_rewrite_history_most_recent_first(),
            max_chars=int(max_chars),
        )
    except Exception:
        # Keep it non-fatal: call sites already tolerate missing history.
        text = str(history_text or "").strip()
        if not text:
            return None
        if max_chars > 0 and len(text) > max_chars:
            text = text[-max_chars:]
        return text or None


def _coerce_intent(value: object) -> str:
    token = str(value or "").strip().upper()
    if token in _ALLOWED_INTENTS:
        return token
    return "RAG_REQUIRED"


class LLMQueryRewriter(AbstractQueryRewriter):
    """
    LLM-based query rewriter for RAG systems.

    Uses any chat LLM to rewrite user queries for improved retrieval effectiveness.
    The rewriter can expand ambiguous queries, add context, rephrase for better
    semantic matching, and generate multiple query variations.

    This implementation is LLM-agnostic and works with any ChatLLMBase implementation
    from the encapsulation layer (OpenAI, Qwen, HuggingFace, etc.).

    RAG Pipeline Position:
        User Query → Query Rewrite → Retrieval → Rerank → LLM Generate Answer
                     ↑ This component

    Key features:
    - LLM-agnostic: Works with any ChatLLMBase implementation
    - Configurable system prompts for different rewriting strategies
    - Temperature control for creativity vs consistency
    - Token limit management for efficient processing
    - Integration with RAG-ARC LLM infrastructure via dependency injection
    """

    def __init__(self, config):
        super().__init__(config)
        # In benchmark/experiment mode, query rewrite is disabled; avoid building any LLM clients.
        if benchmark_mode_enabled():
            self.chat_llm = None
            return
        # Build LLM from sub-config following framework pattern
        # Accepts any ChatLLMBase implementation (OpenAI, Qwen, HuggingFace, etc.)
        self.chat_llm = config.chat_llm_config.build()

    def rewrite_query(
        self,
        query: str,
        *,
        history_text: str | None = None,
        **kwargs: Any
    ) -> str:
        """
        Rewrite a query using any chat LLM.

        Primary configuration (instruction, temperature, max_tokens) comes from config.

        Args:
            query: Original user query to rewrite
            **kwargs: Additional parameters (can override max_tokens, temperature)

        Returns:
            Rewritten query string optimized for retrieval

        Raises:
            ValueError: If query is empty or invalid
            Exception: If LLM call fails
        """
        if benchmark_mode_enabled():
            return str(query or "")
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        query = str(query)
        if getattr(self.config, "skip_rewrite_if_contains", None):
            for token in list(getattr(self.config, "skip_rewrite_if_contains") or []):
                if token and token in query:
                    logger.info("Skipping query rewrite because query contains %r", token)
                    return query
        if getattr(self.config, "skip_rewrite_regexes", None):
            for pattern in list(getattr(self.config, "skip_rewrite_regexes") or []):
                if not pattern:
                    continue
                try:
                    if re.search(pattern, query):
                        logger.info("Skipping query rewrite because query matches pattern %r", pattern)
                        return query
                except re.error:
                    logger.warning("Invalid skip rewrite regex ignored: %r", pattern)

        # Get instruction from config (with default value set in config)
        instruction = self.config.instruction
        logger.info("Using instruction from config")

        use_history = bool(getattr(self.config, "use_history_for_rewrite", False))
        max_history_chars = int(getattr(self.config, "rewrite_history_max_chars", 0) or 0)
        history_snippet = _history_for_rewrite(history_text, max_chars=max_history_chars) if use_history else None

        # Prepare messages for LLM
        messages = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": (
                    QUERY_REWRITE_USER_PROMPT_WITH_HISTORY.format(query=query, history=history_snippet)
                    if history_snippet
                    else QUERY_REWRITE_USER_PROMPT.format(query=query)
                ),
            },
        ]

        try:
            # Use LLM to rewrite query - pass all parameters to encapsulation layer
            rewritten = self.chat_llm.chat(
                messages=messages,
                **kwargs  # Pass through all parameters to encapsulation layer
            )

            # Clean up response (remove quotes, extra whitespace)
            rewritten = rewritten.strip().strip('"').strip("'")

            # Fallback to original if rewrite is empty
            if not rewritten:
                logger.warning("LLM returned empty rewrite, using original query")
                return query

            logger.info(f"Query rewritten: '{query}' → '{rewritten}'")
            return rewritten

        except Exception as e:
            logger.error(f"Query rewriting failed: {e}")
            # Return original query as fallback
            logger.warning("Using original query as fallback")
            return query

    def rewrite_query_with_intent(
        self,
        query: str,
        *,
        history_text: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Intent-aware rewrite for multi-turn conversations (opt-in via env).

        Returns a dict:
          - intent: one of _ALLOWED_INTENTS
          - rewritten_query: str
          - anchors: list[str]
          - reason: str (optional)

        This method is intentionally conservative:
        - If intent routing is disabled, it falls back to rewrite_query() and returns intent=RAG_REQUIRED.
        - On parsing errors, it returns the original query with intent=RAG_REQUIRED (observable via logs).
        """
        if benchmark_mode_enabled():
            return {"intent": "RAG_REQUIRED", "rewritten_query": str(query or ""), "anchors": []}
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        if not rag_intent_routing_enabled():
            return {"intent": "RAG_REQUIRED", "rewritten_query": self.rewrite_query(query, history_text=history_text, **kwargs), "anchors": []}

        instruction = str(self.config.instruction or "").rstrip() + "\n\n" + QUERY_REWRITE_INTENT_SYSTEM_SUFFIX
        use_history = bool(getattr(self.config, "use_history_for_rewrite", False))
        max_history_chars = int(getattr(self.config, "rewrite_history_max_chars", 0) or 0)
        history_snippet = _history_for_rewrite(history_text, max_chars=max_history_chars) if use_history else None

        messages = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": (
                    QUERY_REWRITE_AND_INTENT_USER_PROMPT_WITH_HISTORY.format(query=str(query), history=history_snippet)
                    if history_snippet
                    else QUERY_REWRITE_AND_INTENT_USER_PROMPT.format(query=str(query))
                ),
            },
        ]

        try:
            raw = self.chat_llm.chat(messages=messages, **kwargs)
            text = str(raw or "").strip()
            from core.utils.json_extract import safe_json_loads

            payload = safe_json_loads(text, expected="dict")
            if not isinstance(payload, dict):
                logger.warning("rewrite_query_with_intent: non-JSON response; fallback to original query")
                return {"intent": "RAG_REQUIRED", "rewritten_query": str(query), "anchors": []}

            intent = _coerce_intent(payload.get("intent"))
            rewritten = str(payload.get("rewritten_query") or "").strip().strip('"').strip("'") or str(query)
            anchors_raw = payload.get("anchors")
            anchors: list[str] = []
            if isinstance(anchors_raw, list):
                for a in anchors_raw:
                    s = str(a or "").strip()
                    if s and s not in anchors:
                        anchors.append(s)
            # Keep anchors aligned with rewritten query to avoid including both "intended" and "mistaken" subjects
            # on CORRECTION-like turns (domain-agnostic; no hardcoded keyword lists).
            if anchors and rewritten:
                try:
                    from core.utils.anchor_consistency import prune_anchors_by_query_text

                    anchors = prune_anchors_by_query_text(anchors=anchors, rewritten_query=rewritten)
                except Exception:  # noqa: BLE001
                    pass
            reason = str(payload.get("reason") or "").strip()
            out: dict[str, Any] = {"intent": intent, "rewritten_query": rewritten, "anchors": anchors}
            if reason:
                out["reason"] = reason
            return out
        except Exception as exc:  # noqa: BLE001
            logger.warning("rewrite_query_with_intent failed; using original query: %s", exc)
            return {"intent": "RAG_REQUIRED", "rewritten_query": str(query), "anchors": []}

    def rewrite_query_with_routing(
        self,
        query: str,
        *,
        history_text: str | None = None,
        **kwargs: Any,
    ) -> tuple[str, dict[str, float] | None, str | None]:
        """
        Rewrite query and (optionally) return per-query retrieval ratios for MultiPath quotas.

        Ratios are small non-negative numbers, e.g. {dense:1, bm25:1, graph:1.5}.
        """
        if benchmark_mode_enabled():
            return str(query or ""), None, None
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        # Gate: allow disabling dynamic routing without changing JSON config.
        if not rag_retrieval_dynamic_quota_enabled():
            return self.rewrite_query(query, history_text=history_text, **kwargs), None, None

        query = str(query)
        # Keep existing skip rules for rewrite; when triggered, fall back to static routing ratios.
        if getattr(self.config, "skip_rewrite_if_contains", None):
            for token in list(getattr(self.config, "skip_rewrite_if_contains") or []):
                if token and token in query:
                    logger.info("Skipping query rewrite (routing fallback) because query contains %r", token)
                    return query, None, None
        if getattr(self.config, "skip_rewrite_regexes", None):
            for pattern in list(getattr(self.config, "skip_rewrite_regexes") or []):
                if not pattern:
                    continue
                try:
                    if re.search(pattern, query):
                        logger.info("Skipping query rewrite (routing fallback) because query matches pattern %r", pattern)
                        return query, None, None
                except re.error:
                    logger.warning("Invalid skip rewrite regex ignored: %r", pattern)

        instruction = str(self.config.instruction or "").rstrip() + "\n\n" + QUERY_REWRITE_ROUTING_SYSTEM_SUFFIX

        use_history = bool(getattr(self.config, "use_history_for_rewrite", False))
        max_history_chars = int(getattr(self.config, "rewrite_history_max_chars", 0) or 0)
        history_snippet = _history_for_rewrite(history_text, max_chars=max_history_chars) if use_history else None

        messages = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": (
                    QUERY_REWRITE_AND_ROUTING_USER_PROMPT_WITH_HISTORY.format(query=query, history=history_snippet)
                    if history_snippet
                    else QUERY_REWRITE_AND_ROUTING_USER_PROMPT.format(query=query)
                ),
            },
        ]

        try:
            raw = self.chat_llm.chat(messages=messages, **kwargs)
            text = str(raw or "").strip()
            from core.utils.json_extract import safe_json_loads

            payload = safe_json_loads(text, expected="dict")
            if not isinstance(payload, dict):
                # Strict contract: routing mode must return JSON.
                # Avoid a second LLM call on failure; keep behavior deterministic.
                logger.warning("Query rewrite routing: non-JSON response; using original query and static routing")
                return query, None, None

            rewritten = str(payload.get("rewritten_query") or "").strip().strip('"').strip("'")
            if not rewritten:
                rewritten = query

            bm25_query = str(payload.get("bm25_query") or "").strip().strip('"').strip("'") or None

            ratios_raw = payload.get("retrieval_ratios")
            ratios: dict[str, float] = {}
            if isinstance(ratios_raw, dict):
                for key in ("dense", "bm25", "graph"):
                    val = ratios_raw.get(key)
                    try:
                        f = float(val)
                    except Exception:  # noqa: BLE001
                        continue
                    if f < 0:
                        continue
                    ratios[key] = f

            if not ratios:
                return rewritten, None, bm25_query

            return rewritten, ratios, bm25_query
        except Exception as exc:  # noqa: BLE001
            # Avoid a second LLM call on errors; keep retrieval functional by falling back to the original query.
            logger.warning("Query rewrite with routing failed; using original query and static routing: %s", exc)
            return query, None, None

    def get_rewriter_info(self) -> Dict[str, Any]:
        """
        Get information about this query rewriter's configuration.

        Returns:
            Dictionary containing rewriter information
        """
        return {
            "type": "llm_query_rewriter",
            "llm_info": self.chat_llm.get_model_info(),
            "instruction": self.config.instruction,
            "fallback_strategy": "original_query"
        }
