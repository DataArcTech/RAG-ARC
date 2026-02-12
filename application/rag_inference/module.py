from encapsulation.data_model.schema import Chunk
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Iterator, Callable, Mapping
import inspect
import logging
import time
import uuid
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from framework.module import AbstractModule
from core.utils.owner_guard import normalize_owner_id, is_admin_owner, get_admin_owner_id
from application.rabc.visibility import build_owner_visibility
from core.prompts import (
    MINDMAP_GENERATION_SYSTEM_PROMPT_ZH,
    build_mindmap_generation_user_prompt,
    get_rag_inference_system_prompt,
    get_rag_chat_system_prompt,
    build_intent_context_system_prompt,
    build_routing_intent_system_prompt,
)
from framework.register import Register
from framework.thread_pool import get_thread_pool
from config.output_limits import CHAT_MAX_IMAGE_INPUTS, RAG_RETRIEVAL_OBSERVABILITY, RAG_RETRIEVAL_LOG_TOP_FILES, RAG_RETRIEVAL_LOG_TOP_CHUNKS
from core.utils.multimodal_images import collect_image_paths_from_chunk_payloads
from core.utils.multimodal_llm import build_multimodal_user_message
from core.utils.chunk_dedupe import dedupe_chunks
from encapsulation.web_search import TavilySearchClient, aggregate_tavily_results
from config.retrieval_routing import rag_retrieval_dynamic_quota_enabled
from config.rag_intent_routing import (
    rag_evidence_consistency_enabled,
    rag_evidence_min_keep,
)
from config import pageindex as pageindex_cfg
from core.retrieval.pageindex_retriever import PageIndexRetriever
from application.intent_routing import IntentRoutingService
from config.rag_intent_routing import rag_intent_routing_enabled

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from config.application.rag_inference_config import RAGInferenceConfig
    from application.knowledge.module import Knowledge
 
class RAGInference(AbstractModule):
    def __init__(self, config: 'RAGInferenceConfig'):
        super().__init__(config=config)
        logger.info("Building query_rewriter...")
        self.query_rewriter = self.config.query_rewrite_config.build()
        logger.info("Query rewriter built successfully")
        
        logger.info("Building retriever...")
        self.retriever = self.config.retrieval_config.build()
        logger.info("Retriever built successfully")

        # Startup visibility: if a MultiPath branch weight is 0, that branch is skipped in normal RAG.
        # This affects online retrieval only; offline indexing/ingestion still builds indices as configured.
        try:
            mp_cfg = getattr(self.retriever, "config", None)
            weights = getattr(mp_cfg, "weights", None)
            retriever_cfgs = list(getattr(mp_cfg, "retrievers", None) or [])
            if isinstance(weights, list) and retriever_cfgs:
                enabled: list[str] = []
                disabled: list[str] = []
                for idx, rcfg in enumerate(retriever_cfgs):
                    rtype = str(getattr(rcfg, "type", "") or type(rcfg).__name__)
                    w = None
                    if idx < len(weights):
                        try:
                            w = float(weights[idx])
                        except Exception:  # noqa: BLE001
                            w = None
                    if w is not None and w <= 0:
                        disabled.append(f"{idx}:{rtype}")
                    else:
                        enabled.append(f"{idx}:{rtype}")
                if disabled:
                    logger.info(
                        "Normal RAG MultiPath: enabled=%s disabled=%s weights=%s",
                        enabled,
                        disabled,
                        weights,
                    )
        except Exception:  # noqa: BLE001
            pass
        self.graph_retriever = None
        graph_cls_name = "PrunedHippoRAGNeo4jRetriever"
        if hasattr(self.retriever, "retrievers"):
            for candidate in self.retriever.retrievers:
                if candidate.__class__.__name__ == graph_cls_name:
                    self.graph_retriever = candidate
                    logger.info("Detected graph retriever for admin/global access")
                    break
        
        logger.info("Building reranker...")
        self.reranker = self.config.reranker_config.build()
        logger.info("Reranker built successfully")
        
        logger.info("Building llm...")
        self.llm = self.config.llm_config.build()
        logger.info("LLM built successfully")
        self._knowledge_module: Optional["Knowledge"] = None
        self._tavily_client: TavilySearchClient | None = None
        self.pageindex_retriever: PageIndexRetriever | None = None
        if pageindex_cfg.pageindex_enabled():
            try:
                self.pageindex_retriever = PageIndexRetriever()
                logger.info("PageIndex retriever initialized")
            except Exception as exc:  # noqa: BLE001
                logger.warning("PageIndex retriever init failed: %s", exc)
                self.pageindex_retriever = None
        try:
            web_cfg = getattr(self.config, "web_search", None)
            if web_cfg is not None and bool(getattr(web_cfg, "enabled", False)):
                api_key = getattr(web_cfg, "api_key", None)
                if not api_key or not str(api_key).strip():
                    logger.warning("Tavily API key is empty or not configured (web search disabled)")
                    self._tavily_client = None
                else:
                    self._tavily_client = TavilySearchClient(
                        api_key=api_key,
                        endpoint_url=str(getattr(web_cfg, "endpoint_url", "") or "").strip(),
                        timeout_seconds=float(getattr(web_cfg, "timeout_seconds")),
                        search_depth=str(getattr(web_cfg, "search_depth") or "advanced"),
                        max_results=int(getattr(web_cfg, "max_results")),
                    )
                    logger.info("Tavily client initialized successfully (web search enabled)")
            else:
                logger.info("Web search disabled in config (web_search.enabled=False or web_search config missing)")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to initialize Tavily client (web search disabled): %s", exc)
            self._tavily_client = None

        self._intent_routing: IntentRoutingService | None = None
        if rag_intent_routing_enabled():
            try:
                self._intent_routing = IntentRoutingService()
                logger.info("Intent routing initialized (TOML-driven semantic router)")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Intent routing init failed; continuing without it: %s", exc)
                self._intent_routing = None
        else:
            logger.info("Intent routing disabled (RAG_INTENT_ROUTING_ENABLED=0)")

    async def _run_blocking(self, func, *args, **kwargs):
        """Run a blocking function in a separate thread to avoid blocking the event loop."""
        return await get_thread_pool().run_blocking(func, *args, **kwargs)

    def chat(
        self,
        query: str,
        owner_id: uuid.UUID,
        return_subgraph: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        *,
        include_share: bool = False,
        share_owner_id: uuid.UUID | None = None,
    ) -> tuple[str, list[Chunk], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Chat with RAG system (synchronous version, kept for backward compatibility).
        For async non-blocking version, use chat_async().
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            result = asyncio.run(
                self.chat_async(
                    query,
                    owner_id,
                    return_subgraph,
                    progress_callback=progress_callback,
                    include_share=include_share,
                    share_owner_id=share_owner_id,
                )
            )
            # chat_async may return additional experimental fields; keep chat() stable.
            if isinstance(result, tuple) and len(result) >= 4:
                return result[0], result[1], result[2], result[3]
            return result  # type: ignore[return-value]

        def _run_in_thread():
            result = asyncio.run(
                self.chat_async(
                    query,
                    owner_id,
                    return_subgraph,
                    progress_callback=progress_callback,
                    include_share=include_share,
                    share_owner_id=share_owner_id,
                )
            )
            # chat_async may return additional experimental fields; keep chat() stable.
            if isinstance(result, tuple) and len(result) >= 4:
                return result[0], result[1], result[2], result[3]
            return result

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_run_in_thread).result()

    async def chat_async(
        self,
        query: str,
        owner_id: uuid.UUID,
        return_subgraph: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        history_text: Optional[str] = None,
        current_user_query: Optional[str] = None,
        enable_web_search: bool = False,
        session_id: uuid.UUID | None = None,
        *,
        include_share: bool = False,
        share_owner_id: uuid.UUID | None = None,
    ) -> tuple[str, list[Chunk], Optional[Dict[str, Any]], Optional[Dict[str, Any]], Optional[Dict[str, Any]], str | None]:
        """
        Chat with RAG system

        Args:
            query: User query
            owner_id: User ID for user-isolated retrieval
        return_subgraph: If True, include serialized subgraph data in the response

        Returns:
            Tuple of (LLM response, chunks, subgraph_data, subgraph_info, raw_llm_response)
            - subgraph_data is None if return_subgraph=False or retriever doesn't support it
            - subgraph_info mirrors the retriever diagnostics used for graph export when available
            - raw_llm_response is the raw LLM API response (None for non-OpenAI LLMs)
        """
        messages, chunks, subgraph_data, subgraph_info = await self._run_blocking(
            self._build_messages_and_context,
            query=query,
            owner_id=owner_id,
            return_subgraph=return_subgraph,
            progress_callback=progress_callback,
            history_text=history_text,
            enable_web_search=enable_web_search,
            session_id=session_id,
            include_share=include_share,
            share_owner_id=share_owner_id,
        )
        
        # Capture raw LLM response (for debugging).
        def _chat_with_raw():
            # Debug log: show the messages we send to LLM (truncated)
            try:
                preview_messages = []
                for m in messages:
                    content = m.get("content", "")
                    if isinstance(content, str) and len(content) > 500:
                        content = content[:500] + "...[truncated]"
                    preview_messages.append({"role": m.get("role"), "content": content})
                logger.info(
                    "RAGInference.chat_async LLM request: owner_id=%s return_subgraph=%s messages=%s",
                    str(owner_id),
                    return_subgraph,
                    json.dumps(preview_messages, ensure_ascii=False, default=str),
                )
            except Exception:  # noqa: BLE001
                pass
            image_paths = collect_image_paths_from_chunk_payloads(chunks, max_images=CHAT_MAX_IMAGE_INPUTS)
            if hasattr(self.llm, 'client') and hasattr(self.llm.client, 'chat'):
                # OpenAI client.
                call_messages = messages
                if image_paths:
                    try:
                        call_messages = [dict(m) for m in messages]
                        user_idx = len(call_messages) - 1
                        if user_idx >= 0 and isinstance(call_messages[user_idx].get("content"), str):
                            call_messages[user_idx] = build_multimodal_user_message(
                                text=call_messages[user_idx]["content"],
                                image_paths=image_paths,
                            )
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Failed to build multimodal messages; using text-only: %s", exc)
                        call_messages = messages

                try:
                    raw_response = self.llm.client.chat.completions.create(
                        model=self.llm.model_name,
                        messages=call_messages,
                        max_tokens=self.llm.max_tokens,
                        temperature=self.llm.temperature,
                    )
                except Exception as exc:  # noqa: BLE001
                    if image_paths:
                        logger.warning(
                            "Model/API does not support multimodal chat inputs; continuing with text-only (captions as fallback): %s",
                            exc,
                        )
                    raw_response = self.llm.client.chat.completions.create(
                        model=self.llm.model_name,
                        messages=messages,
                        max_tokens=self.llm.max_tokens,
                        temperature=self.llm.temperature,
                    )
                response_text = raw_response.choices[0].message.content.strip()
                # Convert into a JSON-serializable dict.
                raw_dict = {
                    'id': raw_response.id,
                    'model': raw_response.model,
                    'choices': [{
                        'index': c.index,
                        'message': {'role': c.message.role, 'content': c.message.content},
                        'finish_reason': c.finish_reason
                    } for c in raw_response.choices],
                    'usage': {
                        'prompt_tokens': raw_response.usage.prompt_tokens if raw_response.usage else None,
                        'completion_tokens': raw_response.usage.completion_tokens if raw_response.usage else None,
                        'total_tokens': raw_response.usage.total_tokens if raw_response.usage else None,
                    } if raw_response.usage else None,
                }
                try:
                    # Debug log: full raw LLM response (truncated for safety)
                    logger.info(
                        "RAGInference.chat_async LLM raw_response: id=%s model=%s text_preview=%s",
                        raw_dict.get("id"),
                        raw_dict.get("model"),
                        (response_text[:500] + "...[truncated]") if len(response_text) > 500 else response_text,
                    )
                except Exception:  # noqa: BLE001
                    pass
                return response_text, raw_dict
            else:
                # Other LLMs (e.g., HuggingFace): return text only.
                if image_paths:
                    logger.warning(
                        "Configured LLM does not support multimodal inputs; continuing with text-only (captions as fallback)."
                    )
                response_text = self.llm.chat(messages)
                try:
                    logger.info(
                        "RAGInference.chat_async LLM text_response (no raw object): owner_id=%s text_preview=%s",
                        str(owner_id),
                        (response_text[:500] + "...[truncated]") if isinstance(response_text, str) and len(response_text) > 500 else response_text,
                    )
                except Exception:  # noqa: BLE001
                    pass
                return response_text, None
        
        response_text, raw_response = await self._run_blocking(_chat_with_raw)

        # Compact citations to the actually-used Sources, and keep returned chunks aligned with the new numbering.
        #
        # Motivation:
        # - The model may cite only a subset of sources (e.g. <sup>1</sup>, <sup>3</sup>, <sup>6</sup>).
        # - Downstream/UI expects citations to be dense (1..N) and aligned with the returned chunk list.
        try:
            from core.utils.citations import compact_sup_citations

            rewritten, used_keys = compact_sup_citations(str(response_text or ""), max_key=len(chunks))
            if used_keys:
                response_text = rewritten
                # Return only cited chunks, reindexed to match compacted citations.
                chunks = [chunks[k - 1] for k in used_keys if 1 <= k <= len(chunks)]
                for i, ch in enumerate(chunks, start=1):
                    meta = getattr(ch, "metadata", None) or {}
                    meta["_source_key"] = i
                    ch.metadata = meta
        except Exception:  # noqa: BLE001
            pass

        raw_mindmap_response = None
        # Always generate the mindmap using the mindmap prompt to keep it aligned with the chat answer.
        if return_subgraph:
            # Use the current user question instead of the full conversation history.
            mindmap_query = current_user_query if current_user_query else query
            subgraph_data, raw_mindmap_response = await self._run_blocking(
                self._generate_mindmap,
                mindmap_query,
                response_text,
                chunks,
            )
        return (response_text, chunks, subgraph_data, subgraph_info, raw_response, raw_mindmap_response)

    def stream_chat(
        self,
        query: str,
        owner_id: uuid.UUID,
        return_subgraph: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        history_text: Optional[str] = None,
        enable_web_search: bool = False,
        session_id: uuid.UUID | None = None,
        user_type: Optional[int] = None,
    ) -> tuple[Iterator[str], list[Chunk], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Stream chat completion from the configured LLM."""

        messages, chunks, subgraph_data, subgraph_info = self._build_messages_and_context(
            query=query,
            owner_id=owner_id,
            return_subgraph=return_subgraph,
            progress_callback=progress_callback,
            history_text=history_text,
            enable_web_search=enable_web_search,
            session_id=session_id,
            user_type=user_type,
        )
        return (self.llm.stream_chat(messages), chunks, subgraph_data, subgraph_info)

    @staticmethod
    def _emit_progress(
        callback: Optional[Callable[[Dict[str, Any]], None]],
        payload: Dict[str, Any],
    ) -> None:
        if callback is None:
            return
        try:
            enriched = dict(payload or {})
            enriched.setdefault("v", 1)
            enriched.setdefault("type", "progress")
            enriched.setdefault("ts_ms", int(time.time() * 1000))
            callback(enriched)
        except Exception:  # noqa: BLE001
            return

    def _build_messages_and_context(
        self,
        *,
        query: str,
        owner_id: uuid.UUID,
        return_subgraph: bool,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        history_text: Optional[str] = None,
        enable_web_search: bool = False,
        session_id: uuid.UUID | None = None,
        user_type: Optional[int] = None,
        include_share: bool = False,
        share_owner_id: uuid.UUID | None = None,
        debug_state: dict[str, Any] | None = None,
        disable_intent_routing: bool = False,
    ) -> tuple[List[Dict[str, str]], list[Chunk], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        def _chunk_preview(chunk: Chunk, *, max_chars: int = 160) -> str:
            metadata = getattr(chunk, "metadata", None) or {}
            text = metadata.get("prompt_text") or metadata.get("index_text") or getattr(chunk, "content", None) or ""
            text = str(text)
            text = " ".join(text.split())
            if max_chars <= 0:
                return ""
            if len(text) <= max_chars:
                return text
            return text[:max_chars].rstrip() + "…"

        def _chunk_brief(chunk: Chunk) -> Dict[str, Any]:
            metadata = getattr(chunk, "metadata", None) or {}
            def _num(value: Any) -> float | None:
                if value is None:
                    return None
                if isinstance(value, bool):
                    return float(value)
                if isinstance(value, (int, float)):
                    return float(value)
                try:
                    return float(value)
                except Exception:  # noqa: BLE001
                    return None

            return {
                "id": str(getattr(chunk, "id", "") or ""),
                "score": _num(metadata.get("score")),
                "rerank_score": _num(metadata.get("rerank_score")),
                "chunk_role": metadata.get("chunk_role"),
                "semantic_unit_type": metadata.get("semantic_unit_type"),
                "filename": metadata.get("filename") or metadata.get("source_file_id"),
                "source_file_id": metadata.get("source_file_id") or metadata.get("sourceFileId") or metadata.get("file_id"),
                "_hipporag_ppr_score": _num(metadata.get("_hipporag_ppr_score")),
                "_hipporag_dense_score": _num(metadata.get("_hipporag_dense_score")),
                "_hipporag_selection_strategy": metadata.get("_hipporag_selection_strategy"),
                "preview": _chunk_preview(chunk),
            }

        def _coerce_file_id(meta: Any) -> str | None:
            if not isinstance(meta, dict):
                return None
            for key in ("source_file_id", "sourceFileId", "file_id", "fileId", "document_id", "documentId"):
                token = str(meta.get(key) or "").strip()
                if token:
                    return token
            return None

        def _file_distribution(chunks: list[Chunk], *, limit: int) -> list[dict[str, Any]]:
            from collections import Counter

            ctr: Counter[str] = Counter()
            name_by_id: dict[str, str] = {}
            for ch in chunks:
                meta = getattr(ch, "metadata", None) or {}
                fid = _coerce_file_id(meta)
                if not fid:
                    continue
                ctr[fid] += 1
                if fid not in name_by_id:
                    name = str(meta.get("filename") or "").strip()
                    if name:
                        name_by_id[fid] = name
            top = []
            for fid, count in ctr.most_common(max(int(limit), 0)):
                top.append({"source_file_id": fid, "count": int(count), "filename": name_by_id.get(fid)})
            return top

        debug_enabled = isinstance(debug_state, dict)

        def _debug_section(name: str) -> dict[str, Any] | None:
            if not debug_enabled:
                return None
            section = debug_state.setdefault(name, {})
            if not isinstance(section, dict):
                section = {}
                debug_state[name] = section
            return section

        def _debug_stage(name: str) -> dict[str, Any] | None:
            if not debug_enabled:
                return None
            stages = debug_state.setdefault("stages", {})
            if not isinstance(stages, dict):
                stages = {}
                debug_state["stages"] = stages
            section = stages.setdefault(name, {})
            if not isinstance(section, dict):
                section = {}
                stages[name] = section
            return section

        def _chunk_debug(chunk: Chunk) -> dict[str, Any]:
            metadata = dict(getattr(chunk, "metadata", None) or {})
            content = getattr(chunk, "content", None)
            if isinstance(content, dict):
                content = content.get("text") or content.get("content") or content
            graph = getattr(chunk, "graph", None)
            graph_payload = None
            if graph is not None:
                try:
                    if hasattr(graph, "is_empty") and graph.is_empty():
                        graph_payload = None
                    elif hasattr(graph, "to_dict"):
                        graph_payload = graph.to_dict()
                    else:
                        graph_payload = graph
                except Exception:  # noqa: BLE001
                    graph_payload = None
            payload: dict[str, Any] = {
                "id": str(getattr(chunk, "id", "") or ""),
                "content": content,
                "metadata": metadata,
            }
            if graph_payload:
                payload["graph"] = graph_payload
            return payload

        def _chunks_debug_list(items: list[Chunk]) -> list[dict[str, Any]]:
            return [_chunk_debug(chunk) for chunk in items]

        if debug_enabled:
            debug_state.setdefault(
                "request",
                {
                    "query": str(query or ""),
                    "owner_id": str(owner_id),
                    "return_subgraph": bool(return_subgraph),
                    "enable_web_search": bool(enable_web_search),
                    "session_id": str(session_id) if session_id is not None else None,
                    "include_share": bool(include_share),
                    "share_owner_id": str(share_owner_id) if share_owner_id is not None else None,
                    "disable_intent_routing": bool(disable_intent_routing),
                },
            )

        routing_intent: str | None = None
        routing_action: str = "rag"
        routed_history_text: str | None = None
        # Some tests construct RAGInference via object.__new__ (skipping __init__),
        # so guard against missing attributes.
        intent_router = getattr(self, "_intent_routing", None)
        if disable_intent_routing:
            intent_debug = _debug_section("intent_routing")
            if intent_debug is not None:
                intent_debug.update(
                    {
                        "enabled": False,
                        "disabled_reason": "force_disabled",
                    }
                )
            intent_router = None
        elif intent_router is None:
            intent_debug = _debug_section("intent_routing")
            if intent_debug is not None:
                intent_debug.update({"enabled": False, "disabled_reason": "not_configured"})

        if intent_router is not None:
            self._emit_progress(progress_callback, {"stage": "intent_routing", "status": "start"})
            routing_start = time.perf_counter()
            try:
                result = intent_router.route(
                    session_id=str(session_id) if session_id is not None else None,
                    user_query=str(query or ""),
                    history_text=history_text,
                    enable_web_search=bool(enable_web_search),
                )
                routing_intent = str(result.intent or "").strip() or None
                routing_action = str(result.action or "").strip() or "rag"
                routed_history_text = result.scoped_history_text
                # Prefer scoped history when available; keep None for "no history".
                if routed_history_text is not None:
                    history_text = routed_history_text
                self._emit_progress(
                    progress_callback,
                    {
                        "stage": "intent_routing",
                        "status": "end",
                        "duration_ms": int((time.perf_counter() - routing_start) * 1000),
                        "intent": routing_intent,
                        "action": routing_action,
                        "score": float(getattr(result, "score", 0.0) or 0.0),
                        **({"topic": result.topic.__dict__} if getattr(result, "topic", None) else {}),
                    },
                )
                intent_debug = _debug_section("intent_routing")
                if intent_debug is not None:
                    intent_debug.update(
                        {
                            "enabled": True,
                            "status": "ok",
                            "duration_ms": int((time.perf_counter() - routing_start) * 1000),
                            "intent": routing_intent,
                            "action": routing_action,
                            "score": float(getattr(result, "score", 0.0) or 0.0),
                            "topic": (result.topic.__dict__ if getattr(result, "topic", None) else None),
                            "debug": getattr(result, "debug", None),
                            "scoped_history_text": routed_history_text,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Intent routing failed; defaulting to RAG: %s", exc)
                self._emit_progress(
                    progress_callback,
                    {
                        "stage": "intent_routing",
                        "status": "error",
                        "duration_ms": int((time.perf_counter() - routing_start) * 1000),
                        "error": str(exc) or exc.__class__.__name__,
                    },
                )
                intent_debug = _debug_section("intent_routing")
                if intent_debug is not None:
                    intent_debug.update(
                        {
                            "enabled": True,
                            "status": "error",
                            "duration_ms": int((time.perf_counter() - routing_start) * 1000),
                            "error": str(exc) or exc.__class__.__name__,
                        }
                    )

        self._emit_progress(progress_callback, {"stage": "rewrite", "status": "start"})
        rewrite_start = time.perf_counter()
        rewrite_kwargs: dict[str, Any] = {}
        rewrite_debug = _debug_section("rewrite")
        rewritten_query: str
        retrieval_ratios: dict[str, float] | None = None
        bm25_query: str | None = None
        # Legacy fields kept for backward compatibility with message building, now unused.
        rewrite_intent: str | None = None
        rewrite_anchors: list[str] = []
        rewrite_reason: str | None = None
        skip_retrieval = routing_action == "no_retrieval"
        try:
            sig = inspect.signature(self.query_rewriter.rewrite_query)
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if accepts_var_kw or "history_text" in sig.parameters:
                # Only pass history_text when supported to keep compatibility with simple stubs/lambdas.
                rewrite_kwargs["history_text"] = history_text
            if rewrite_debug is not None and (accepts_var_kw or "debug_info" in sig.parameters):
                rewrite_kwargs["debug_info"] = rewrite_debug
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to inspect query_rewriter.rewrite_query signature: %s", exc, exc_info=True)

        # Only rewrite when we plan to retrieve (rag/web_only). For no_retrieval, answer quickly.
        if skip_retrieval:
            rewritten_query = str(query or "").strip()
            retrieval_ratios = None
            bm25_query = None
        elif rag_retrieval_dynamic_quota_enabled() and hasattr(self.query_rewriter, "rewrite_query_with_routing"):
            try:
                fn = getattr(self.query_rewriter, "rewrite_query_with_routing")
                sig = inspect.signature(fn)
                accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
                call_kwargs = dict(rewrite_kwargs)
                if not accepts_var_kw and "history_text" not in sig.parameters:
                    call_kwargs.pop("history_text", None)
                if not accepts_var_kw and "debug_info" not in sig.parameters:
                    call_kwargs.pop("debug_info", None)
                rewritten_query, retrieval_ratios, bm25_query = fn(query, **call_kwargs)  # type: ignore[misc]
            except Exception as exc:  # noqa: BLE001
                logger.warning("rewrite_query_with_routing failed; falling back to rewrite_query: %s", exc)
                rewritten_query = self.query_rewriter.rewrite_query(query, **rewrite_kwargs)
                retrieval_ratios = None
                bm25_query = None
        else:
            rewritten_query = self.query_rewriter.rewrite_query(query, **rewrite_kwargs)
            retrieval_ratios = None
            bm25_query = None
        self._emit_progress(
            progress_callback,
            {
                "stage": "rewrite",
                "status": "end",
                "duration_ms": int((time.perf_counter() - rewrite_start) * 1000),
                "rewritten_query": rewritten_query,
                **({"intent": routing_intent} if routing_intent else {}),
                **({"anchors": rewrite_anchors} if rewrite_anchors else {}),
                **({"reason": rewrite_reason} if rewrite_reason else {}),
                **({"retrieval_ratios": retrieval_ratios} if retrieval_ratios else {}),
                **({"bm25_query": bm25_query} if bm25_query else {}),
            },
        )
        if rewrite_debug is not None:
            rewrite_debug.setdefault("duration_ms", int((time.perf_counter() - rewrite_start) * 1000))
            rewrite_debug.setdefault("rewritten_query", rewritten_query)
            rewrite_debug.setdefault("retrieval_ratios", retrieval_ratios)
            rewrite_debug.setdefault("bm25_query", bm25_query)
            rewrite_debug.setdefault("routing_intent", routing_intent)
            rewrite_debug.setdefault("routing_action", routing_action)
            rewrite_debug.setdefault("skip_retrieval", bool(skip_retrieval))

        if skip_retrieval:
            # Conversation-only mode: we still build messages (using history), but avoid retrieval/rerank.
            chunks = []
            subgraph_info = None
            messages: List[Dict[str, str]] = []
            system_prompt = get_rag_chat_system_prompt(profile="cli", user_type=user_type)
            messages.append({"role": "system", "content": system_prompt})
            # Routing-intent specific behavior (e.g. CLARIFY_REQUIRED) is expressed via centrally managed prompts.
            if routing_intent:
                routing_prompt = build_routing_intent_system_prompt(routing_intent)
                if routing_prompt:
                    messages.append({"role": "system", "content": routing_prompt})
            # Add a small, centralized guardrail so answers respect CORRECTION/TOPIC_SWITCH constraints even without retrieval.
            if rewrite_intent or rewrite_anchors or rewrite_reason:
                messages.append(
                    {
                        "role": "system",
                        "content": build_intent_context_system_prompt(
                            intent=rewrite_intent,
                            anchors=rewrite_anchors,
                            reason=rewrite_reason,
                        ),
                    }
                )
            if history_text:
                history_lines = history_text.strip().split("\n")
                for line in history_lines:
                    if ":" not in line:
                        continue
                    role, content = line.split(":", 1)
                    role = role.strip()
                    content = content.strip()
                    if role in ("user", "assistant") and content:
                        messages.append({"role": role, "content": content})
            messages.append({"role": "user", "content": str(query or "").strip()})
            debug_stage = _debug_stage("no_retrieval")
            if debug_stage is not None:
                debug_stage.update(
                    {
                        "reason": "routing_action_no_retrieval",
                        "messages": list(messages),
                        "chunks": [],
                        "subgraph_info": subgraph_info,
                    }
                )
            debug_response = _debug_section("response")
            if debug_response is not None:
                debug_response.update({"messages": list(messages), "chunks": []})
            return (messages, chunks, None, subgraph_info)

        do_internal_retrieval = routing_action == "rag"

        self._emit_progress(progress_callback, {"stage": "retrieve", "status": "start"})
        retrieve_start = time.perf_counter()
        web_future = None
        web_step_id = uuid.uuid4().hex[:8]
        candidate_cfg = getattr(self.config, "candidate_selection")
        graph_candidates_k = int(getattr(candidate_cfg, "graph_candidates_k"))
        web_candidates_k = int(getattr(candidate_cfg, "web_candidates_k"))
        rerank_keep_k = int(getattr(candidate_cfg, "rerank_keep_k"))

        web_cfg = getattr(self.config, "web_search")
        web_enabled = bool(getattr(web_cfg, "enabled"))
        # Debug logging for web search conditions
        if enable_web_search:
            logger.info(f"Web search check: enable_web_search={enable_web_search}, web_enabled={web_enabled}, "
                       f"_tavily_client={'not None' if self._tavily_client is not None else 'None'}, "
                       f"web_candidates_k={web_candidates_k}")
        if enable_web_search and web_enabled and self._tavily_client is not None and web_candidates_k > 0:
            logger.info(f"Starting web search for query: {rewritten_query}")
            self._emit_progress(
                progress_callback,
                {"stage": "web_search", "status": "start", "provider": "tavily", "max_results": web_candidates_k},
            )

            def _run_web_search() -> dict[str, Any]:
                try:
                    logger.info(f"Executing Tavily search for query: {rewritten_query}")
                    results = self._tavily_client.search(query=rewritten_query, max_results=web_candidates_k)
                    logger.info(f"Tavily search returned {len(results)} results")
                    agg_stats = None
                    web_agg_enabled = bool(getattr(web_cfg, 'aggregate_enabled', True))
                    if web_agg_enabled:
                        group_by = str(getattr(web_cfg, 'aggregate_group_by', 'domain') or 'domain')
                        max_groups = int(getattr(web_cfg, 'aggregate_max_groups', 3))
                        max_per_group = int(getattr(web_cfg, 'aggregate_max_results_per_group', 2))
                        agg_results, agg_stats = aggregate_tavily_results(
                            results,
                            group_by=group_by,
                            max_groups=max_groups,
                            max_items_per_group=max_per_group,
                        )
                        logger.info(
                            'Web search aggregation applied: group_by=%s in=%d out=%d groups=%d',
                            agg_stats.group_by,
                            agg_stats.total_in,
                            agg_stats.total_out,
                            len(agg_stats.groups),
                        )
                        results = agg_results
                    evidence_chunks = self._tavily_client.to_evidence_chunks(results=results, step_id=web_step_id, query=rewritten_query)
                    logger.info(f"Converted to {len(evidence_chunks)} evidence chunks")
                    return {"evidences": evidence_chunks, "aggregation": (agg_stats.__dict__ if agg_stats is not None else None)}
                except Exception as e:
                    logger.error(f"Error in web search: {e}", exc_info=True)
                    raise

            web_future = get_thread_pool().executor.submit(_run_web_search)
            logger.info("Web search future submitted")

        pageindex_section_scores: Dict[str, float] = {}
        chunks: list[Chunk] = []
        subgraph_infos: list[dict[str, Any]] = []
        retriever_info = None

        if do_internal_retrieval:
            visibility = build_owner_visibility(
                primary_owner_id=owner_id,
                include_share=bool(include_share),
                share_owner_id=share_owner_id,
                label=("me+share" if include_share else "me"),
            )
            retrieve_stage = _debug_stage("retrieve")
            if retrieve_stage is not None:
                retrieve_stage["owner_visibility"] = {
                    "label": visibility.label,
                    "owner_ids": list(visibility.owner_ids),
                }
            self._emit_progress(
                progress_callback,
                {
                    "stage": "retrieve",
                    "status": "scope",
                    "owner_visibility": {"label": visibility.label, "owner_ids": list(visibility.owner_ids)},
                },
            )

            # Some tests bypass __init__ so optional components may be missing.
            pageindex_retriever = getattr(self, "pageindex_retriever", None)
            if pageindex_retriever is not None and pageindex_cfg.pageindex_enabled():
                try:
                    for owner_token in visibility.owner_ids:
                        section_hits = pageindex_retriever.retrieve_sections(
                            rewritten_query,
                            owner_id=str(owner_token),
                            file_ids=None,
                        )
                        for hit in section_hits:
                            meta = getattr(hit, "metadata", None) or {}
                            section_id = str(meta.get("section_id") or getattr(hit, "id", "") or "").strip()
                            if not section_id:
                                continue
                            score = meta.get("score")
                            if not isinstance(score, (int, float)):
                                score = 0.0
                            existing = pageindex_section_scores.get(section_id, None)
                            if existing is None or score > existing:
                                pageindex_section_scores[section_id] = float(score)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("PageIndex retrieval failed: %s", exc)

            for owner_token in visibility.owner_ids:
                retrieved: list[Chunk] = self.retriever.invoke(
                    rewritten_query,
                    owner_id=owner_token,
                    return_subgraph_info=return_subgraph,
                    k=graph_candidates_k,
                    **({"bm25_query": bm25_query} if bm25_query else {}),
                    **({"retrieval_ratios": retrieval_ratios} if retrieval_ratios else {}),
                )
                per_info = self._consume_subgraph_info(retrieved)
                if per_info:
                    try:
                        per_info.setdefault("owner_scope", str(owner_token))
                    except Exception:  # noqa: BLE001
                        pass
                    subgraph_infos.append(per_info)
                chunks.extend(retrieved)

            try:
                if hasattr(self.retriever, "get_multipath_info"):
                    retriever_info = self.retriever.get_multipath_info()
            except Exception:  # noqa: BLE001
                retriever_info = None
            self._emit_progress(
                progress_callback,
                {
                    "stage": "retrieve",
                    "status": "end",
                    "duration_ms": int((time.perf_counter() - retrieve_start) * 1000),
                    "chunks": len(chunks),
                    "retriever": retriever_info,
                    "top_chunks": [_chunk_brief(chunk) for chunk in chunks[: min(len(chunks), 10)]],
                    **(
                        {
                            "file_distribution": _file_distribution(chunks, limit=RAG_RETRIEVAL_LOG_TOP_FILES),
                        }
                        if RAG_RETRIEVAL_OBSERVABILITY
                        else {}
                    ),
                },
            )
            retrieve_stage = _debug_stage("retrieve")
            if retrieve_stage is not None:
                retrieve_stage.update(
                    {
                        "duration_ms": int((time.perf_counter() - retrieve_start) * 1000),
                        "chunks": _chunks_debug_list(chunks),
                        "chunks_count": len(chunks),
                        "retriever": retriever_info,
                        "pageindex_section_scores": dict(pageindex_section_scores),
                    }
                )
        else:
            # Web-only mode: skip internal retrieval and wait for (optional) web search.
            self._emit_progress(
                progress_callback,
                {
                    "stage": "retrieve",
                    "status": "end",
                    "duration_ms": int((time.perf_counter() - retrieve_start) * 1000),
                    "chunks": 0,
                    "skipped_internal_retrieval": True,
                },
            )
            retrieve_stage = _debug_stage("retrieve")
            if retrieve_stage is not None:
                retrieve_stage.update(
                    {
                        "duration_ms": int((time.perf_counter() - retrieve_start) * 1000),
                        "chunks": [],
                        "chunks_count": 0,
                        "skipped_internal_retrieval": True,
                    }
                )

        if do_internal_retrieval and owner_id is not None and is_admin_owner(owner_id) and not chunks and self.graph_retriever is not None:
            logger.info("Admin/global mode: multipath returned 0 results, falling back to graph retriever")
            graph_chunks = self.graph_retriever.invoke(
                rewritten_query,
                owner_id=owner_id,
                return_subgraph_info=return_subgraph,
            )
            if graph_chunks:
                chunks = graph_chunks

        file_status_before = list(chunks) if debug_enabled else None
        chunks = self._filter_chunks_by_file_status(chunks)
        file_status_stage = _debug_stage("file_status_filter")
        if file_status_stage is not None:
            file_status_stage.update(
                {
                    "chunks_in": _chunks_debug_list(file_status_before or []),
                    "chunks_out": _chunks_debug_list(chunks),
                    "chunks_in_count": len(file_status_before or []),
                    "chunks_out_count": len(chunks),
                }
            )
        before = len(chunks)
        chunks = dedupe_chunks(chunks)
        if len(chunks) != before:
            logger.info("Deduped retrieved chunks: %d -> %d", before, len(chunks))
        dedupe_stage = _debug_stage("dedupe_after_file_status")
        if dedupe_stage is not None:
            dedupe_stage.update(
                {
                    "chunks_in_count": before,
                    "chunks_out_count": len(chunks),
                    "chunks": _chunks_debug_list(chunks),
                }
            )

        if pageindex_section_scores:
            chunks, info = self._filter_chunks_by_section_scores(
                chunks,
                section_scores=pageindex_section_scores,
                min_keep=pageindex_cfg.section_min_keep(),
                score_weight=pageindex_cfg.section_score_weight(),
            )
            logger.info("PageIndex section filter: %s", info)
            section_stage = _debug_stage("pageindex_section_filter")
            if section_stage is not None:
                section_stage.update(
                    {
                        "info": info,
                        "chunks": _chunks_debug_list(chunks),
                        "chunks_count": len(chunks),
                    }
                )

        # Evidence consistency filtering (opt-in): keep chunks aligned to anchors to reduce drift.
        if rag_evidence_consistency_enabled() and rewrite_anchors:
            try:
                from core.utils.evidence_consistency import filter_chunks_by_anchors

                self._emit_progress(progress_callback, {"stage": "evidence_filter", "status": "start"})
                before = len(chunks)
                evidence_stage = _debug_stage("evidence_filter")
                if evidence_stage is not None:
                    evidence_stage.update(
                        {
                            "chunks_in": _chunks_debug_list(chunks),
                            "chunks_in_count": len(chunks),
                            "anchors": list(rewrite_anchors),
                        }
                    )
                chunks, info = filter_chunks_by_anchors(
                    chunks=chunks,
                    anchors=rewrite_anchors,
                    min_keep=rag_evidence_min_keep(),
                )
                after = len(chunks)
                self._emit_progress(
                    progress_callback,
                    {
                        "stage": "evidence_filter",
                        "status": "end",
                        "chunks_in": before,
                        "chunks_out": after,
                        "passed": bool(getattr(info, "passed", True)),
                        "matched_by_filename": getattr(info, "matched_by_filename", None),
                        "matched_by_content": getattr(info, "matched_by_content", None),
                    },
                )
                if evidence_stage is not None:
                    evidence_stage.update(
                        {
                            "chunks_out": _chunks_debug_list(chunks),
                            "chunks_out_count": len(chunks),
                            "info": info,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("Evidence consistency filter failed (ignored): %s", exc)

        web_chunks: list[Chunk] = []
        web_aggregation: dict[str, Any] | None = None
        if web_future is not None:
            logger.info("Waiting for web search results...")
            web_start = time.perf_counter()
            try:
                timeout_seconds = float(getattr(web_cfg, "timeout_seconds"))
                timeout_grace_seconds = float(getattr(web_cfg, "timeout_grace_seconds"))
                total_timeout = timeout_seconds + timeout_grace_seconds
                logger.info(f"Web search timeout: {total_timeout}s")
                # Use done() to check if completed, avoid blocking indefinitely
                if web_future.done():
                    web_payload = web_future.result()
                    if isinstance(web_payload, dict):
                        evidences = web_payload.get("evidences") if isinstance(web_payload.get("evidences"), list) else []
                        web_aggregation = web_payload.get("aggregation") if isinstance(web_payload.get("aggregation"), dict) else None
                    else:
                        evidences = web_payload if isinstance(web_payload, list) else []
                    logger.info(f"Web search already completed, got {len(evidences) if evidences else 0} evidences")
                else:
                    # Wait with timeout, but don't block forever
                    try:
                        web_payload = web_future.result(timeout=total_timeout)
                        if isinstance(web_payload, dict):
                            evidences = web_payload.get("evidences") if isinstance(web_payload.get("evidences"), list) else []
                            web_aggregation = web_payload.get("aggregation") if isinstance(web_payload.get("aggregation"), dict) else None
                        else:
                            evidences = web_payload if isinstance(web_payload, list) else []
                        logger.info(f"Web search completed within timeout, got {len(evidences) if evidences else 0} evidences")
                    except TimeoutError:
                        logger.warning(f"Web search timed out after {total_timeout}s, continuing without web results")
                        evidences = []
                    except Exception as timeout_exc:
                        logger.warning(f"Web search future error: {timeout_exc}, continuing without web results")
                        evidences = []
                for evidence in evidences or []:
                    if not isinstance(evidence, dict):
                        continue
                    provenance = evidence.get("provenance") if isinstance(evidence.get("provenance"), dict) else {}
                    url = provenance.get("url")
                    title = (str(evidence.get("content") or "").split("\n", 1)[0]).strip() or "web"
                    filename = title
                    external_chunk_id = str(evidence.get("chunk_id") or "").strip() or None
                    if external_chunk_id:
                        provenance = dict(provenance)
                        provenance.setdefault("external_chunk_id", external_chunk_id)
                    web_chunks.append(
                        Chunk(
                            id=str(uuid.uuid4()),
                            content=str(evidence.get("content") or ""),
                            metadata={
                                "source": evidence.get("source") or "web.tavily",
                                "score": evidence.get("score"),
                                "filename": filename,
                                "provenance": provenance,
                                "prompt_text": str(evidence.get("content") or ""),
                            },
                        )
                    )
                self._emit_progress(
                    progress_callback,
                    {
                        "stage": "web_search",
                        "status": "end",
                        "duration_ms": int((time.perf_counter() - web_start) * 1000),
                        "results": len(web_chunks),
                        "aggregation": web_aggregation,
                    },
                )
                web_stage = _debug_stage("web_search")
                if web_stage is not None:
                    web_stage.update(
                        {
                            "status": "end",
                            "duration_ms": int((time.perf_counter() - web_start) * 1000),
                            "results": len(web_chunks),
                            "chunks": _chunks_debug_list(web_chunks),
                            "aggregation": web_aggregation,
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Web search failed: {exc}", exc_info=True)
                self._emit_progress(
                    progress_callback,
                    {
                        "stage": "web_search",
                        "status": "error",
                        "duration_ms": int((time.perf_counter() - web_start) * 1000),
                        "error": str(exc) or exc.__class__.__name__,
                    },
                )
                web_stage = _debug_stage("web_search")
                if web_stage is not None:
                    web_stage.update(
                        {
                            "status": "error",
                            "duration_ms": int((time.perf_counter() - web_start) * 1000),
                            "error": str(exc) or exc.__class__.__name__,
                        }
                    )
            if web_chunks:
                logger.info(f"Adding {len(web_chunks)} web chunks to final chunks list")
                chunks.extend(web_chunks)
            else:
                logger.info("No web chunks to add")

        before = len(chunks)
        chunks = dedupe_chunks(chunks)
        if len(chunks) != before:
            logger.info("Deduped chunks after web merge: %d -> %d", before, len(chunks))
        web_merge_stage = _debug_stage("web_merge")
        if web_merge_stage is not None:
            web_merge_stage.update(
                {
                    "web_chunks": _chunks_debug_list(web_chunks),
                    "web_chunks_count": len(web_chunks),
                    "chunks_in_count": before,
                    "chunks_out_count": len(chunks),
                    "chunks": _chunks_debug_list(chunks),
                }
            )

        # File-level candidate pruning (file-first): keep top files then top chunks per file.
        # Keep it close to rerank so it sees the final merged candidate pool.
        try:
            from core.retrieval.file_pruning import prune_chunks_by_file

            enabled = bool(getattr(candidate_cfg, "file_prune_enabled", True))
            max_files = int(getattr(candidate_cfg, "file_prune_max_files", 4))
            max_chunks_per_file = int(getattr(candidate_cfg, "file_prune_max_chunks_per_file", 6))
            self._emit_progress(
                progress_callback,
                {
                    "stage": "candidate_prune",
                    "status": "start",
                    "enabled": enabled,
                    "max_files": max_files,
                    "max_chunks_per_file": max_chunks_per_file,
                    "chunks_in": len(chunks),
                },
            )
            prune_before = list(chunks)
            candidate_stage = _debug_stage("candidate_prune")
            if candidate_stage is not None:
                candidate_stage.update(
                    {
                        "enabled": enabled,
                        "max_files": max_files,
                        "max_chunks_per_file": max_chunks_per_file,
                        "chunks_in": _chunks_debug_list(prune_before),
                        "chunks_in_count": len(prune_before),
                    }
                )
            pruned, info = prune_chunks_by_file(
                chunks,
                enabled=enabled,
                max_files=max_files,
                max_chunks_per_file=max_chunks_per_file,
            )
            chunks = pruned
            self._emit_progress(
                progress_callback,
                {
                    "stage": "candidate_prune",
                    "status": "end",
                    "enabled": bool(getattr(info, "enabled", False)),
                    "chunks_in": int(getattr(info, "chunks_in", len(chunks))),
                    "chunks_out": int(getattr(info, "chunks_out", len(chunks))),
                    "files_in": int(getattr(info, "files_in", 0)),
                    "files_kept": int(getattr(info, "files_kept", 0)),
                    "top_files": getattr(info, "top_files", None),
                },
            )
            if candidate_stage is not None:
                candidate_stage.update(
                    {
                        "chunks_out": _chunks_debug_list(chunks),
                        "chunks_out_count": len(chunks),
                        "info": info,
                    }
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Candidate prune failed (ignored): %s", exc)

        subgraph_info = None
        if subgraph_infos:
            if len(subgraph_infos) == 1:
                subgraph_info = subgraph_infos[0]
            else:
                subgraph_info = {"_multi_owner": True, "scopes": subgraph_infos}
        # Track web chunks before reranking to ensure they're included
        web_chunk_ids = set()
        if web_future is not None:
            for chunk in chunks:
                metadata = getattr(chunk, "metadata", None) or {}
                source = metadata.get("source", "")
                if source == "web.tavily" or "tavily" in str(source).lower():
                    web_chunk_ids.add(getattr(chunk, "id", None))
        
        self._emit_progress(
            progress_callback,
            {"stage": "rerank", "status": "start", "chunks_in": len(chunks)},
        )
        rerank_before = list(chunks)
        rerank_stage = _debug_stage("rerank")
        if rerank_stage is not None:
            rerank_stage.update(
                {
                    "chunks_in": _chunks_debug_list(rerank_before),
                    "chunks_in_count": len(rerank_before),
                    "top_k": int(rerank_keep_k),
                }
            )
        rerank_start = time.perf_counter()
        reranked_chunks = self.reranker.rerank(rewritten_query, chunks, top_k=rerank_keep_k)
        
        # Ensure web chunks are included in final results
        if web_chunk_ids:
            reranked_chunk_ids = {getattr(chunk, "id", None) for chunk in reranked_chunks}
            missing_web_chunks = []
            for chunk in chunks:
                chunk_id = getattr(chunk, "id", None)
                if chunk_id in web_chunk_ids and chunk_id not in reranked_chunk_ids:
                    missing_web_chunks.append(chunk)
            
            if missing_web_chunks:
                logger.info(f"Adding {len(missing_web_chunks)} web chunks that were filtered out by reranker")
                # Add web chunks at the beginning to give them priority
                reranked_chunks = missing_web_chunks + reranked_chunks
                # Trim to keep_k if needed
                if len(reranked_chunks) > rerank_keep_k:
                    reranked_chunks = reranked_chunks[:rerank_keep_k]
        
        chunks = reranked_chunks
        before = len(chunks)
        chunks = dedupe_chunks(chunks)
        if len(chunks) != before:
            logger.info("Deduped chunks after rerank: %d -> %d", before, len(chunks))
        reranker_info = None
        try:
            reranker_info = self.reranker.get_reranker_info()
        except Exception:  # noqa: BLE001
            reranker_info = None
        self._emit_progress(
            progress_callback,
            {
                "stage": "rerank",
                "status": "end",
                "duration_ms": int((time.perf_counter() - rerank_start) * 1000),
                "chunks_out": len(chunks),
                "reranker": reranker_info,
                "top_chunks": [_chunk_brief(chunk) for chunk in chunks[: min(len(chunks), 10)]],
                **(
                    {
                        "file_distribution": _file_distribution(chunks, limit=RAG_RETRIEVAL_LOG_TOP_FILES),
                        "chunks_preview": [_chunk_brief(chunk) for chunk in chunks[: min(len(chunks), RAG_RETRIEVAL_LOG_TOP_CHUNKS)]],
                    }
                    if RAG_RETRIEVAL_OBSERVABILITY
                    else {}
                ),
            },
        )
        if rerank_stage is not None:
            rerank_stage.update(
                {
                    "duration_ms": int((time.perf_counter() - rerank_start) * 1000),
                    "chunks_out": _chunks_debug_list(chunks),
                    "chunks_out_count": len(chunks),
                    "reranker": reranker_info,
                }
            )

        # Note: subgraph_data is now generated in _generate_mindmap based on final answer chunks
        # We don't export subgraph here to avoid including all retrieval-stage chunks
        # The subgraph will be built from the reranked chunks that are actually used in the final answer
        subgraph_data = None

        messages: List[Dict[str, str]] = []
        
        # Add system prompt (requires <sup> citations).
        system_prompt = get_rag_inference_system_prompt(user_type=user_type)
        messages.append({"role": "system", "content": system_prompt})
        # Provide intent/anchors to generation as constraints (kept domain-agnostic; centrally managed prompt).
        if rewrite_intent or rewrite_anchors or rewrite_reason:
            messages.append(
                {
                    "role": "system",
                    "content": build_intent_context_system_prompt(
                        intent=rewrite_intent,
                        anchors=rewrite_anchors,
                        reason=rewrite_reason,
                    ),
                }
            )
        
        # If history exists, prepend it (aligned with WebSocket behavior).
        if history_text:
            # Parse history text into message list.
            history_lines = history_text.strip().split("\n")
            for line in history_lines:
                if ":" in line:
                    role, content = line.split(":", 1)
                    role = role.strip()
                    content = content.strip()
                    if role in ("user", "assistant") and content:
                        messages.append({"role": role, "content": content})
        
        # Use "Source key=N" format (aligned with chatbot.py).
        if chunks:
            # Build a contiguous, non-empty source list for the LLM.
            # This avoids gaps in Source key numbering (which can confuse citation parsing and downstream tools).
            source_items: list[tuple[Chunk, str, str]] = []
            llm_source_stage = _debug_stage("llm_sources")
            skipped_sources: list[dict[str, Any]] = []
            if llm_source_stage is not None:
                llm_source_stage["chunks_in"] = _chunks_debug_list(chunks)
                llm_source_stage["chunks_in_count"] = len(chunks)
            for chunk in chunks:
                metadata = getattr(chunk, "metadata", None) or {}
                filename = str(metadata.get("filename") or "").strip() or "source"
                chunk_id = getattr(chunk, "id", None)

                # Extract chunk text from various sources.
                prompt_text = metadata.get("prompt_text")
                index_text = metadata.get("index_text")
                chunk_content = getattr(chunk, "content", None)

                chunk_text: str | None = None
                if prompt_text and isinstance(prompt_text, str) and prompt_text.strip():
                    chunk_text = prompt_text
                elif index_text and isinstance(index_text, str) and index_text.strip():
                    chunk_text = index_text
                elif chunk_content:
                    # Handle dict content (e.g. {'text': '', 'metadata': ...}).
                    if isinstance(chunk_content, dict):
                        chunk_text = str(chunk_content.get("text") or chunk_content.get("content") or "")
                        if not chunk_text and isinstance(chunk_content.get("metadata"), dict):
                            chunk_text = str(
                                chunk_content["metadata"].get("text") or chunk_content["metadata"].get("content") or ""
                            )
                    elif isinstance(chunk_content, str):
                        chunk_text = chunk_content
                    else:
                        chunk_text = str(chunk_content) if chunk_content else ""

                if not chunk_text or not chunk_text.strip():
                    logger.warning(
                        "Skipping empty chunk for LLM sources: filename=%s chunk_id=%s "
                        "prompt_text=%s index_text=%s chunk_content_type=%s",
                        filename,
                        chunk_id,
                        ("empty" if not prompt_text else f"type={type(prompt_text).__name__} len={len(str(prompt_text))}"),
                        ("empty" if not index_text else f"type={type(index_text).__name__} len={len(str(index_text))}"),
                        ("empty" if not chunk_content else type(chunk_content).__name__),
                    )
                    skipped_sources.append(
                        {
                            "id": str(chunk_id or ""),
                            "filename": filename,
                            "prompt_text_empty": not bool(prompt_text),
                            "index_text_empty": not bool(index_text),
                            "chunk_content_type": (None if not chunk_content else type(chunk_content).__name__),
                        }
                    )
                    continue

                source_items.append((chunk, filename, str(chunk_text).strip()))

            # Replace `chunks` with the exact list used as LLM sources.
            chunks = [item[0] for item in source_items]
            if llm_source_stage is not None:
                llm_source_stage.update(
                    {
                        "sources": [
                            {
                                "chunk_id": str(getattr(item[0], "id", "") or ""),
                                "filename": item[1],
                                "text": item[2],
                            }
                            for item in source_items
                        ],
                        "sources_count": len(source_items),
                        "chunks_out": _chunks_debug_list(chunks),
                        "chunks_out_count": len(chunks),
                        "skipped_sources": list(skipped_sources),
                    }
                )

            chunk_ids_for_llm = [getattr(chunk, "id", None) for chunk in chunks[:10]]
            logger.info(
                "RAGInference._build_messages_and_context: chunks passed to LLM (first 10 IDs): %s",
                chunk_ids_for_llm,
            )

            for i, (chunk, filename, chunk_text) in enumerate(source_items, start=1):
                # Persist the Source key on the chunk so downstream can align citations.
                meta = getattr(chunk, "metadata", None) or {}
                meta["_source_key"] = i
                chunk.metadata = meta

                messages.append({"role": "user", "content": f"Source key={i} title={filename}\n{chunk_text}"})
                if i <= 5:
                    logger.debug(
                        "RAGInference: LLM source key=%d chunk_id=%s filename=%s",
                        i,
                        getattr(chunk, "id", None),
                        filename,
                    )
            messages.append({"role": "user", "content": f"Based on the above Sources, please answer question: {rewritten_query}"})
        else:
            # If there are no Sources, send the question directly (do not mention Sources).
            messages.append({"role": "user", "content": rewritten_query})
        logger.info("Invoked chat with query: %s (owner_id=%s)", query, owner_id)
        logger.info("Query rewritten to: %s", rewritten_query)
        if history_text:
            history_count = len([m for m in messages if m.get("role") in ("user", "assistant") and "Chunk" not in m.get("content", "")])
            logger.info("Including history: %d history messages", history_count)
        logger.info("Prepared %d messages for LLM", len(messages))
        # Log full messages payload (debug only).
        logger.debug("Full messages sent to LLM: %s", json.dumps(messages, ensure_ascii=False, indent=2))
        debug_response = _debug_section("response")
        if debug_response is not None:
            debug_response.update(
                {
                    "messages": list(messages),
                    "chunks": _chunks_debug_list(chunks),
                    "chunks_count": len(chunks),
                    "subgraph_info": subgraph_info,
                }
            )
        return (messages, chunks, subgraph_data, subgraph_info)

    @staticmethod
    def _dedupe_chunks_by_id(chunks: List[Chunk]) -> List[Chunk]:
        # Backward-compatible wrapper; prefer `core.utils.chunk_dedupe.dedupe_chunks`.
        return dedupe_chunks(chunks)

    @staticmethod
    def _merge_graph_payloads(payloads: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not payloads:
            return None
        merged_chunks_by_id: dict[str, Dict[str, Any]] = {}
        merged_nodes_by_id: dict[str, Dict[str, Any]] = {}
        merged_edges_by_id: dict[str, Dict[str, Any]] = {}
        merged_metadata: dict[str, Any] = {"sources": len(payloads)}

        for payload in payloads:
            if not isinstance(payload, dict):
                continue
            for chunk in payload.get("chunks", []) or []:
                cid = str(chunk.get("id") or "").strip()
                if not cid:
                    continue
                merged_chunks_by_id.setdefault(cid, dict(chunk))
            for node in payload.get("nodes", []) or []:
                nid = str(node.get("id") or "").strip()
                if not nid:
                    continue
                merged_nodes_by_id.setdefault(nid, dict(node))
            for edge in payload.get("edges", []) or []:
                eid = str(edge.get("id") or "").strip()
                if not eid:
                    src = str(edge.get("source") or "").strip()
                    dst = str(edge.get("target") or "").strip()
                    rel = str(edge.get("relation") or "").strip()
                    if src and dst and rel:
                        eid = f"{src}::{rel}::{dst}"
                if not eid:
                    continue
                merged_edges_by_id.setdefault(eid, dict(edge))
            meta = payload.get("metadata")
            if isinstance(meta, dict):
                merged_metadata.setdefault("categories", meta.get("categories"))

        merged = {
            "chunks": list(merged_chunks_by_id.values()),
            "nodes": list(merged_nodes_by_id.values()),
            "edges": list(merged_edges_by_id.values()),
            "metadata": merged_metadata,
        }
        return merged

    def _locate_graph_store(self):
        """Locate the configured graph store if one exists."""
        candidate_retrievers: List[Any] = []
        if self.graph_retriever is not None:
            candidate_retrievers.append(self.graph_retriever)

        if hasattr(self.retriever, 'graph_store'):
            candidate_retrievers.append(self.retriever)

        if hasattr(self.retriever, 'retrievers'):
            candidate_retrievers.extend(self.retriever.retrievers or [])

        nested = getattr(getattr(self.retriever, 'config', None), 'built_retrievers', None)
        if nested:
            candidate_retrievers.extend(nested)

        for retriever in candidate_retrievers:
            graph_store = getattr(retriever, 'graph_store', None)
            if graph_store is not None:
                return graph_store
        return None

    def get_graph_store(self):
        """Expose the underlying graph store for CLI and admin APIs."""
        return self._locate_graph_store()

    def _get_knowledge_module(self) -> Optional["Knowledge"]:
        if self._knowledge_module is None:
            try:
                registrator = Register()
                knowledge_module = registrator.get_object("knowledge")
                self._knowledge_module = knowledge_module
            except Exception as e:
                logger.warning(f"Failed to locate knowledge module for file filtering: {e}")
                self._knowledge_module = None
        return self._knowledge_module

    def _filter_chunks_by_file_status(self, chunks: List[Chunk]) -> List[Chunk]:
        knowledge_module = self._get_knowledge_module()
        if knowledge_module is None:
            return chunks

        def _coerce_file_id(value: Any) -> str | None:
            token = str(value or "").strip()
            return token or None

        def _extract_file_id(metadata: Any) -> str | None:
            if not isinstance(metadata, Mapping):
                return None

            for key in (
                "source_file_id",
                "sourceFileId",
                "file_id",
                "fileId",
                "document_id",
                "documentId",
                "doc_id",
                "docId",
            ):
                token = _coerce_file_id(metadata.get(key))
                if token:
                    return token

            nested = metadata.get("chunk_metadata") or metadata.get("chunkMetadata")
            if isinstance(nested, Mapping):
                for key in (
                    "source_file_id",
                    "sourceFileId",
                    "file_id",
                    "fileId",
                    "document_id",
                    "documentId",
                    "doc_id",
                    "docId",
                ):
                    token = _coerce_file_id(nested.get(key))
                    if token:
                        return token

            return None

        file_status_cache: dict[str, bool] = {}
        filtered_chunks: List[Chunk] = []
        for chunk in chunks:
            file_id = _extract_file_id(getattr(chunk, "metadata", None))
            if file_id and file_id not in file_status_cache:
                file_status_cache[file_id] = knowledge_module.is_file_active(file_id)

            if not file_id or file_status_cache.get(file_id, True):
                filtered_chunks.append(chunk)
        if len(filtered_chunks) != len(chunks):
            logger.info(f"Filtered out {len(chunks) - len(filtered_chunks)} chunks from deleting files")
        return filtered_chunks

    @staticmethod
    def _extract_section_id(chunk: Chunk) -> str | None:
        metadata = getattr(chunk, "metadata", None) or {}
        token = str(metadata.get("section_id") or metadata.get("sectionId") or "").strip()
        if token:
            return token
        chunk_id = str(getattr(chunk, "id", "") or "").strip()
        return chunk_id or None

    def _filter_chunks_by_section_scores(
        self,
        chunks: List[Chunk],
        *,
        section_scores: Dict[str, float],
        min_keep: int,
        score_weight: float,
    ) -> tuple[List[Chunk], Dict[str, Any]]:
        info: Dict[str, Any] = {"sections": len(section_scores)}
        if not section_scores:
            info["filtered"] = False
            return chunks, info

        filtered: List[Chunk] = []
        for chunk in chunks:
            section_id = self._extract_section_id(chunk)
            if not section_id or section_id not in section_scores:
                continue
            filtered.append(chunk)
            if score_weight <= 0:
                continue
            meta = getattr(chunk, "metadata", None) or {}
            base_score = meta.get("score")
            if not isinstance(base_score, (int, float)):
                base_score = 0.0
            section_score = float(section_scores.get(section_id, 0.0))
            meta["section_score"] = section_score
            meta["score"] = float(base_score) + section_score * score_weight
            chunk.metadata = meta

        info["filtered"] = True
        info["chunks_in"] = len(chunks)
        info["chunks_out"] = len(filtered)
        if min_keep > 0 and len(filtered) < min_keep:
            info["fallback"] = True
            return chunks, info
        info["fallback"] = False
        return filtered, info

    @staticmethod
    def _filter_chunks_by_doc_ids(
        chunks: List[Chunk],
        *,
        doc_ids: List[str],
        min_keep: int,
    ) -> tuple[List[Chunk], Dict[str, Any]]:
        info: Dict[str, Any] = {"docs": len(doc_ids)}
        if not doc_ids:
            info["filtered"] = False
            return chunks, info

        allowed = {str(doc_id) for doc_id in doc_ids if str(doc_id)}
        if not allowed:
            info["filtered"] = False
            return chunks, info

        def _coerce_file_id(meta: Any) -> str | None:
            token = str(meta or "").strip()
            return token or None

        filtered: List[Chunk] = []
        for chunk in chunks:
            meta = getattr(chunk, "metadata", None) or {}
            file_id = None
            for key in ("source_file_id", "file_id", "document_id", "doc_id"):
                file_id = _coerce_file_id(meta.get(key))
                if file_id:
                    break
            if file_id and file_id in allowed:
                filtered.append(chunk)

        info["filtered"] = True
        info["chunks_in"] = len(chunks)
        info["chunks_out"] = len(filtered)
        if min_keep > 0 and len(filtered) < min_keep:
            info["fallback"] = True
            return chunks, info
        info["fallback"] = False
        return filtered, info

    @staticmethod
    def _consume_subgraph_info(chunks: List[Chunk]) -> Optional[Dict[str, Any]]:
        """Pop and return embedded subgraph diagnostics when available."""

        for chunk in chunks:
            metadata = getattr(chunk, "metadata", None) or {}
            if "_subgraph_info" in metadata:
                info = metadata.pop("_subgraph_info")
                logger.info("Extracted subgraph info before reranking")
                return info
        return None

    def export_graph_overview(
        self,
        owner_id: Optional[uuid.UUID],
        max_nodes: int = 1000,
        max_edges: int = 5000,
        include_node_types: Optional[List[str]] = None,
        *,
        include_share: bool = False,
        share_owner_id: uuid.UUID | None = None,
        owner_ids: Optional[List[uuid.UUID | str]] = None,
    ) -> Dict[str, Any]:
        """
        Export a graph overview for visual inspection.

        Args:
            owner_id: Optional owner scope. None returns all owners (admin use).
            max_nodes: Maximum number of nodes to sample.
            max_edges: Maximum number of edges to sample.
            include_node_types: Optional white-list of node types.

        Returns:
            Graph overview payload compatible with frontend graph viewers.
        """
        graph_store = self._locate_graph_store()
        if graph_store is None:
            raise RuntimeError("Graph store is not configured for the current retriever profile")

        normalized_owner = normalize_owner_id(owner_id) if owner_id is not None else None
        owner_scope_label = normalized_owner
        if owner_scope_label is None:
            owner_scope_label = get_admin_owner_id() or "GLOBAL_ADMIN"
        graph_store_class_name = graph_store.__class__.__name__

        if graph_store_class_name == 'PrunedHippoRAGNeo4jStore':
            from encapsulation.database.utils.graph_export_utils_neo4j import GraphExporterNeo4j as GraphExporter
        else:
            from encapsulation.database.utils.graph_export_utils import GraphExporter

        if owner_id is None:
            overview = GraphExporter.export_full_graph(
                graph_store=graph_store,
                max_nodes=max_nodes,
                max_edges=max_edges,
                include_node_types=include_node_types,
                owner_id=None,
                owner_scope_label=owner_scope_label,
            )
        else:
            visibility = build_owner_visibility(
                primary_owner_id=owner_id,
                owner_ids=owner_ids,
                include_share=bool(include_share),
                share_owner_id=share_owner_id,
                label=("graph_overview" if not include_share else "graph_overview+share"),
            )
            payloads: list[dict[str, Any]] = []
            for owner_token in visibility.owner_ids:
                payloads.append(
                    GraphExporter.export_full_graph(
                        graph_store=graph_store,
                        max_nodes=max_nodes,
                        max_edges=max_edges,
                        include_node_types=include_node_types,
                        owner_id=str(owner_token),
                        owner_scope_label=str(owner_token),
                    )
                )
            overview = payloads[0] if len(payloads) == 1 else (self._merge_graph_payloads(payloads) or payloads[0])
        logger.info(
            "Exported graph overview (owner_scope=%s) with %d nodes and %d edges",
            normalized_owner,
            len(overview.get('nodes', [])) + len(overview.get('chunks', [])),
            len(overview.get('edges', [])),
        )
        return overview

    def _generate_mindmap(self, query: str, response: str, chunks: list[Chunk]) -> tuple[Dict[str, Any], str | None]:
        """
        Generate mind map data based on query, response and chunks.
        Uses TSV format (same as mindmap_export) for better reliability.
        Only includes chunks that are actually referenced in the response (via <sup>key</sup> tags).

        Args:
            query: User query
            response: LLM response
            chunks: Retrieved chunks (reranked, final chunks used for answer)

        Returns:
            Tuple of (subgraph_data, raw_mindmap_response)
        """
        try:
            # Use TSV format like mindmap_export for better reliability
            from core.prompts import MINDMAP_MERGE_SYSTEM_PROMPT_EN, build_mindmap_merge_user_prompt
            from application.knowledge.mindmap_export import extract_tsv_from_response, convert_tsv_to_graph
            from core.utils.citations import extract_sup_keys
            
            # Extract only chunks that are actually referenced in the response
            # Chunks are indexed from 1 in the messages (Source key=1, key=2, etc.)
            used_keys = set(extract_sup_keys(response, max_key=len(chunks)))
            
            # Filter chunks to only those actually referenced in the response
            # If no citations found, use all chunks (fallback for cases without citations)
            if used_keys:
                referenced_chunks = [chunks[key - 1] for key in sorted(used_keys) if 1 <= key <= len(chunks)]
                logger.info(f"Filtered chunks: {len(chunks)} total, {len(referenced_chunks)} actually referenced in response")
            else:
                # Fallback: if no citations found, use all chunks (may happen if LLM doesn't use citations)
                referenced_chunks = chunks
                logger.info(f"No citations found in response, using all {len(chunks)} chunks")
            
            # Build prompt using TSV format (more reliable than JSON)
            chunks_text = "\n\n".join([f"Chunk {i+1}:\n{chunk.content}" for i, chunk in enumerate(referenced_chunks)])
            # Create a section-like format for the prompt
            sections_text = f"Segment 1:\nContent summary:\n{response}\n\nRetrieved chunks:\n{chunks_text}"
            
            mindmap_prompt = build_mindmap_merge_user_prompt(
                filename=f"Query: {query}",
                sections_text=sections_text
            )

            # Call LLM to generate TSV mind map
            mindmap_messages = [
                {"role": "system", "content": MINDMAP_MERGE_SYSTEM_PROMPT_EN},
                {"role": "user", "content": mindmap_prompt}
            ]

            logger.info("Generating mind map structure with LLM (TSV format)...")
            # Note: This is called from _run_blocking, so it's already in a thread
            mindmap_response = self.llm.chat(mindmap_messages)

            # Log raw mindmap response
            logger.info("Raw mindmap response (first 500 chars): %s", mindmap_response[:500] if mindmap_response else None)

            # Extract TSV from response (same logic as mindmap_export)
            merged_tsv = extract_tsv_from_response(mindmap_response)
            if not merged_tsv.strip():
                logger.warning("LLM did not return valid TSV content")
                return {"nodes": [], "edges": []}, mindmap_response

            # Convert TSV to graph structure (same logic as mindmap_export)
            nodes, edges = convert_tsv_to_graph(merged_tsv)
            
            logger.info(f"Converted TSV to graph: {len(nodes)} nodes, {len(edges)} edges")

            # Build final subgraph data with only referenced chunks
            subgraph_data = self._build_subgraph_data(referenced_chunks, {"nodes": nodes, "edges": edges})

            nodes_count = len(subgraph_data.get('nodes', []))
            edges_count = len(subgraph_data.get('edges', []))
            logger.info(f"Generated mind map: {nodes_count} nodes, {edges_count} edges")
            
            if nodes_count == 0:
                logger.warning("Generated mind map has no nodes - TSV may be empty or invalid")
            
            return subgraph_data, mindmap_response

        except Exception as e:
            logger.error(f"Failed to generate mind map: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Return empty structure consistent with mindmap_export format
            return {"nodes": [], "edges": []}, None


    def _build_subgraph_data(self, chunks: list[Chunk], mindmap_json: Dict[str, Any]) -> Dict[str, Any]:
        """Build final subgraph data combining chunks and mind map structure"""
        from core.mindmap.utils import add_chunks_to_nodes, ensure_mindmap_edge_relation
        
        # Convert Chunk objects to dictionaries for frontend formatting
        chunks_dict = []
        for chunk in chunks:
            # Convert Chunk to dict format expected by add_chunks_to_nodes
            chunk_dict = {
                "content": chunk.content,
                "id": chunk.id,
                "metadata": chunk.metadata or {}
            }
            chunks_dict.append(chunk_dict)
        
        # Get nodes and edges from mindmap_json
        nodes = mindmap_json.get("nodes", [])
        edges = mindmap_json.get("edges", [])
        
        # Ensure edges have "contains" relation (for mindmap consistency)
        edges = ensure_mindmap_edge_relation(edges, relation="contains")
        
        # Add chunks to all nodes (multi-file mode: chunks may come from different files)
        # When filename/file_id are None, the function will extract from each chunk's metadata
        nodes = add_chunks_to_nodes(nodes, chunks_dict, filename=None, file_id=None)

        # Return structure consistent with mindmap_export format
        return {
            "nodes": nodes,
            "edges": edges
        }
