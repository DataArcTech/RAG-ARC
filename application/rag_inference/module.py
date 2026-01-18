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
)
from framework.register import Register
from framework.thread_pool import get_thread_pool
from config.output_limits import CHAT_MAX_IMAGE_INPUTS, RAG_RETRIEVAL_OBSERVABILITY, RAG_RETRIEVAL_LOG_TOP_FILES, RAG_RETRIEVAL_LOG_TOP_CHUNKS
from core.utils.multimodal_images import collect_image_paths_from_chunk_payloads
from core.utils.multimodal_llm import build_multimodal_user_message
from encapsulation.web_search import TavilySearchClient

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
        current_user_query: Optional[str] = None,
        enable_web_search: bool = False,
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
            enable_web_search=enable_web_search,
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
        user_type: Optional[int] = None,
        include_share: bool = False,
        share_owner_id: uuid.UUID | None = None,
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

        self._emit_progress(progress_callback, {"stage": "rewrite", "status": "start"})
        rewrite_start = time.perf_counter()
        rewrite_kwargs: dict[str, Any] = {}
        try:
            sig = inspect.signature(self.query_rewriter.rewrite_query)
            accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
            if accepts_var_kw or "history_text" in sig.parameters:
                # Only pass history_text when supported to keep compatibility with simple stubs/lambdas.
                rewrite_kwargs["history_text"] = history_text
        except Exception as exc:  # noqa: BLE001
            logger.debug("Failed to inspect query_rewriter.rewrite_query signature: %s", exc, exc_info=True)
        rewritten_query = self.query_rewriter.rewrite_query(query, **rewrite_kwargs)
        self._emit_progress(
            progress_callback,
            {
                "stage": "rewrite",
                "status": "end",
                "duration_ms": int((time.perf_counter() - rewrite_start) * 1000),
                "rewritten_query": rewritten_query,
            },
        )

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

            def _run_web_search() -> list[dict]:
                try:
                    logger.info(f"Executing Tavily search for query: {rewritten_query}")
                    results = self._tavily_client.search(query=rewritten_query, max_results=web_candidates_k)
                    logger.info(f"Tavily search returned {len(results)} results")
                    evidence_chunks = self._tavily_client.to_evidence_chunks(results=results, step_id=web_step_id, query=rewritten_query)
                    logger.info(f"Converted to {len(evidence_chunks)} evidence chunks")
                    return evidence_chunks
                except Exception as e:
                    logger.error(f"Error in web search: {e}", exc_info=True)
                    raise

            web_future = get_thread_pool().executor.submit(_run_web_search)
            logger.info("Web search future submitted")

        visibility = build_owner_visibility(
            primary_owner_id=owner_id,
            include_share=bool(include_share),
            share_owner_id=share_owner_id,
            label=("me+share" if include_share else "me"),
        )
        self._emit_progress(
            progress_callback,
            {
                "stage": "retrieve",
                "status": "scope",
                "owner_visibility": {"label": visibility.label, "owner_ids": list(visibility.owner_ids)},
            },
        )

        chunks: list[Chunk] = []
        subgraph_infos: list[dict[str, Any]] = []
        for owner_token in visibility.owner_ids:
            retrieved: list[Chunk] = self.retriever.invoke(
                rewritten_query,
                owner_id=owner_token,
                return_subgraph_info=return_subgraph,
                k=graph_candidates_k,
            )
            per_info = self._consume_subgraph_info(retrieved)
            if per_info:
                try:
                    per_info.setdefault("owner_scope", str(owner_token))
                except Exception:  # noqa: BLE001
                    pass
                subgraph_infos.append(per_info)
            chunks.extend(retrieved)
        retriever_info = None
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

        if owner_id is not None and is_admin_owner(owner_id) and not chunks and self.graph_retriever is not None:
            logger.info("Admin/global mode: multipath returned 0 results, falling back to graph retriever")
            graph_chunks = self.graph_retriever.invoke(
                rewritten_query,
                owner_id=owner_id,
                return_subgraph_info=return_subgraph,
            )
            if graph_chunks:
                chunks = graph_chunks

        chunks = self._filter_chunks_by_file_status(chunks)
        chunks = self._dedupe_chunks_by_id(chunks)

        if web_future is not None:
            logger.info("Waiting for web search results...")
            web_start = time.perf_counter()
            web_chunks: list[Chunk] = []
            try:
                timeout_seconds = float(getattr(web_cfg, "timeout_seconds"))
                timeout_grace_seconds = float(getattr(web_cfg, "timeout_grace_seconds"))
                total_timeout = timeout_seconds + timeout_grace_seconds
                logger.info(f"Web search timeout: {total_timeout}s")
                # Use done() to check if completed, avoid blocking indefinitely
                if web_future.done():
                    evidences = web_future.result()
                    logger.info(f"Web search already completed, got {len(evidences) if evidences else 0} evidences")
                else:
                    # Wait with timeout, but don't block forever
                    try:
                        evidences = web_future.result(timeout=total_timeout)
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
                    },
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
            if web_chunks:
                logger.info(f"Adding {len(web_chunks)} web chunks to final chunks list")
                chunks.extend(web_chunks)
            else:
                logger.info("No web chunks to add")

        chunks = self._dedupe_chunks_by_id(chunks)
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

        # Note: subgraph_data is now generated in _generate_mindmap based on final answer chunks
        # We don't export subgraph here to avoid including all retrieval-stage chunks
        # The subgraph will be built from the reranked chunks that are actually used in the final answer
        subgraph_data = None

        messages: List[Dict[str, str]] = []
        
        # Add system prompt (requires <sup> citations).
        system_prompt = get_rag_inference_system_prompt(user_type=user_type)
        messages.append({"role": "system", "content": system_prompt})
        
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
            # Log chunks passed to LLM for debugging key mismatch
            chunk_ids_for_llm = [getattr(chunk, "id", None) for chunk in chunks[:10]]  # Log first 10
            logger.info("RAGInference._build_messages_and_context: chunks passed to LLM (first 10 IDs): %s", chunk_ids_for_llm)
            for i, chunk in enumerate(chunks):
                metadata = getattr(chunk, "metadata", None) or {}
                filename = str(metadata.get("filename") or "").strip() or "source"
                chunk_id = getattr(chunk, "id", None)
                
                # Extract chunk text from various sources
                prompt_text = metadata.get("prompt_text")
                index_text = metadata.get("index_text")
                chunk_content = getattr(chunk, "content", None)
                
                # Handle different content formats
                chunk_text = None
                if prompt_text and isinstance(prompt_text, str) and prompt_text.strip():
                    chunk_text = prompt_text
                elif index_text and isinstance(index_text, str) and index_text.strip():
                    chunk_text = index_text
                elif chunk_content:
                    # Handle dict content (e.g., {'text': '', 'metadata': ...})
                    if isinstance(chunk_content, dict):
                        chunk_text = chunk_content.get("text") or chunk_content.get("content") or ""
                        # If still empty, try to extract from metadata
                        if not chunk_text and isinstance(chunk_content.get("metadata"), dict):
                            chunk_text = chunk_content["metadata"].get("text") or chunk_content["metadata"].get("content") or ""
                    elif isinstance(chunk_content, str):
                        chunk_text = chunk_content
                    else:
                        chunk_text = str(chunk_content) if chunk_content else ""
                
                # Log and skip empty chunks
                if not chunk_text or not chunk_text.strip():
                    logger.warning(
                        f"Skipping empty chunk: Source key={i+1}, filename={filename}, "
                        f"chunk_id={chunk_id}, "
                        f"prompt_text={'empty' if not prompt_text else f'type={type(prompt_text).__name__}, len={len(str(prompt_text))}'}, "
                        f"index_text={'empty' if not index_text else f'type={type(index_text).__name__}, len={len(str(index_text))}'}, "
                        f"chunk.content={'empty' if not chunk_content else f'type={type(chunk_content).__name__}, value={str(chunk_content)[:100]}'}"
                    )
                    continue

                messages.append({"role": "user", "content": f"Source key={i+1} title={filename}\n{chunk_text}"})
                if i < 5:  # Log first 5 for debugging
                    logger.debug("RAGInference: LLM source key=%d chunk_id=%s filename=%s", i+1, chunk_id, filename)
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
        return (messages, chunks, subgraph_data, subgraph_info)

    @staticmethod
    def _dedupe_chunks_by_id(chunks: List[Chunk]) -> List[Chunk]:
        if not chunks:
            return chunks
        seen: set[str] = set()
        out: list[Chunk] = []
        for chunk in chunks:
            cid = str(getattr(chunk, "id", "") or "").strip()
            if not cid:
                out.append(chunk)
                continue
            if cid in seen:
                continue
            seen.add(cid)
            out.append(chunk)
        return out

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
            import re
            
            # Extract only chunks that are actually referenced in the response
            # Chunks are indexed from 1 in the messages (Source key=1, key=2, etc.)
            _SUP_KEY_RE = re.compile(r"<sup>\s*(?P<key>\d{1,4})\s*</sup>")
            used_keys = set()
            for match in _SUP_KEY_RE.finditer(response):
                try:
                    key = int(match.group("key"))
                    if key > 0 and key <= len(chunks):
                        used_keys.add(key)
                except Exception:  # noqa: BLE001
                    continue
            
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

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response with error recovery"""
        try:
            # Try to find JSON block in markdown code fence
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
            else:
                # Try to parse the entire response as JSON
                json_str = response.strip()

            # Try to parse JSON
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as parse_error:
                # Try to fix common JSON truncation issues
                logger.warning(f"Initial JSON parse failed: {parse_error}, attempting repair...")
                json_str = self._repair_truncated_json(json_str)
                try:
                    parsed = json.loads(json_str)
                    logger.info("Successfully repaired truncated JSON")
                except json.JSONDecodeError:
                    # If repair fails, try to extract partial data
                    logger.error(f"JSON repair failed: {parse_error}")
                    return self._extract_partial_json(json_str)

            # Validate that parsed JSON has expected structure
            if not isinstance(parsed, dict):
                logger.warning("LLM response is not a JSON object, got: %s", type(parsed))
                return {"nodes": [], "edges": []}
            
            # Ensure nodes and edges are lists
            nodes = parsed.get("nodes", [])
            edges = parsed.get("edges", [])
            if not isinstance(nodes, list) or not isinstance(edges, list):
                logger.warning("LLM response nodes/edges are not lists: nodes=%s, edges=%s", type(nodes), type(edges))
                return {"nodes": [], "edges": []}
            
            logger.info("Successfully extracted mindmap JSON: %d nodes, %d edges", len(nodes), len(edges))
            return {"nodes": nodes, "edges": edges}
        except Exception as e:
            logger.error(f"Unexpected error extracting JSON from response: {e}")
            logger.debug(f"Response (first 500 chars): {response[:500]}")
            return {"nodes": [], "edges": []}
    
    def _repair_truncated_json(self, json_str: str) -> str:
        """Attempt to repair truncated JSON by closing open structures"""
        json_str = json_str.strip()
        
        # Count open and close braces/brackets
        open_braces = json_str.count('{')
        close_braces = json_str.count('}')
        open_brackets = json_str.count('[')
        close_brackets = json_str.count(']')
        
        # Remove trailing comma if present (common in truncated JSON)
        json_str = json_str.rstrip().rstrip(',')
        
        # Close open arrays first (edges, nodes)
        while close_brackets < open_brackets:
            json_str += ']'
            close_brackets += 1
        
        # Close open objects
        while close_braces < open_braces:
            json_str += '}'
            close_braces += 1
        
        return json_str
    
    def _extract_partial_json(self, json_str: str) -> Dict[str, Any]:
        """Extract partial JSON data when full parse fails"""
        nodes = []
        edges = []
        
        # Try to extract nodes using regex or string matching
        import re
        # Look for node objects
        node_pattern = r'"id"\s*:\s*"([^"]+)"[^}]*"name"\s*:\s*"([^"]+)"'
        for match in re.finditer(node_pattern, json_str):
            node_id = match.group(1)
            node_name = match.group(2)
            nodes.append({
                "id": node_id,
                "name": node_name,
                "category": node_name,
                "weight": 1
            })
        
        # Look for edge objects
        edge_pattern = r'"id"\s*:\s*"([^"]+)"[^}]*"source"\s*:\s*"([^"]+)"[^}]*"target"\s*:\s*"([^"]+)"'
        for match in re.finditer(edge_pattern, json_str):
            edge_id = match.group(1)
            source = match.group(2)
            target = match.group(3)
            edges.append({
                "id": edge_id,
                "source": source,
                "target": target,
                "relation": "contains",
                "weight": 0.8
            })
        
        if nodes or edges:
            logger.info(f"Extracted partial JSON: {len(nodes)} nodes, {len(edges)} edges using regex fallback")
            return {"nodes": nodes, "edges": edges}
        
        return {"nodes": [], "edges": []}

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
