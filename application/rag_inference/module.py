from encapsulation.data_model.schema import Chunk
from typing import TYPE_CHECKING, Optional, Dict, Any, List, Iterator, Callable
import logging
import time
import uuid
import json
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from framework.module import AbstractModule
from core.utils.owner_guard import normalize_owner_id, is_admin_owner, get_admin_owner_id
from core.prompts import MINDMAP_GENERATION_SYSTEM_PROMPT_ZH, build_mindmap_generation_user_prompt
from framework.register import Register
from framework.thread_pool import get_thread_pool

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

    async def _run_blocking(self, func, *args, **kwargs):
        """Run a blocking function in a separate thread to avoid blocking the event loop."""
        return await get_thread_pool().run_blocking(func, *args, **kwargs)

    def chat(
        self,
        query: str,
        owner_id: uuid.UUID,
        return_subgraph: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> tuple[str, list[Chunk], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Chat with RAG system (synchronous version, kept for backward compatibility).
        For async non-blocking version, use chat_async().
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.chat_async(
                    query,
                    owner_id,
                    return_subgraph,
                    progress_callback=progress_callback,
                )
            )

        def _run_in_thread():
            return asyncio.run(
                self.chat_async(
                    query,
                    owner_id,
                    return_subgraph,
                    progress_callback=progress_callback,
                )
            )

        with ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_run_in_thread).result()

    async def chat_async(
        self,
        query: str,
        owner_id: uuid.UUID,
        return_subgraph: bool = False,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        current_user_query: Optional[str] = None,
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
        )
        
        # 获取原始 LLM response（用于调试）
        def _chat_with_raw():
            if hasattr(self.llm, 'client') and hasattr(self.llm.client, 'chat'):
                # OpenAI 客户端
                raw_response = self.llm.client.chat.completions.create(
                    model=self.llm.model_name,
                    messages=messages,
                    max_tokens=self.llm.max_tokens,
                    temperature=self.llm.temperature,
                )
                response_text = raw_response.choices[0].message.content.strip()
                # 转换为可序列化的 dict
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
                return response_text, raw_dict
            else:
                # 其他 LLM（HuggingFace 等），只返回文本
                response_text = self.llm.chat(messages)
                return response_text, None
        
        response_text, raw_response = await self._run_blocking(_chat_with_raw)

        raw_mindmap_response = None
        # 总是用 mindmap prompt 生成图，确保图和 chat 回答相关
        if return_subgraph:
            # 使用当前用户问题而不是整个对话历史
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
    ) -> tuple[Iterator[str], list[Chunk], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Stream chat completion from the configured LLM."""

        messages, chunks, subgraph_data, subgraph_info = self._build_messages_and_context(
            query=query,
            owner_id=owner_id,
            return_subgraph=return_subgraph,
            progress_callback=progress_callback,
            history_text=history_text,
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
                "preview": _chunk_preview(chunk),
            }

        self._emit_progress(progress_callback, {"stage": "rewrite", "status": "start"})
        rewrite_start = time.perf_counter()
        rewritten_query = self.query_rewriter.rewrite_query(query)
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
        chunks: list[Chunk] = self.retriever.invoke(
            rewritten_query,
            owner_id=owner_id,
            return_subgraph_info=return_subgraph,
        )
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
        subgraph_info = self._consume_subgraph_info(chunks)
        self._emit_progress(
            progress_callback,
            {"stage": "rerank", "status": "start", "chunks_in": len(chunks)},
        )
        rerank_start = time.perf_counter()
        chunks = self.reranker.rerank(rewritten_query, chunks)
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
            },
        )

        subgraph_data = None
        if subgraph_info and return_subgraph:
            self._emit_progress(progress_callback, {"stage": "subgraph_export", "status": "start"})
            export_start = time.perf_counter()
            try:
                graph_store = self._locate_graph_store()
                if graph_store:
                    graph_store_class_name = graph_store.__class__.__name__
                    if graph_store_class_name == "PrunedHippoRAGNeo4jStore":
                        from encapsulation.database.utils.graph_export_utils_neo4j import (
                            GraphExporterNeo4j as GraphExporter,
                        )

                        subgraph_data = GraphExporter.export_subgraph(
                            graph_store=graph_store,
                            subgraph_node_ids=set(subgraph_info["subgraph_nodes"]),
                            seed_entity_ids=set(subgraph_info["seed_entity_ids"]),
                            retrieved_chunk_ids=subgraph_info["retrieved_chunk_ids"],
                            node_ppr_scores=subgraph_info.get("node_ppr_scores", {}),
                        )
                    else:
                        from encapsulation.database.utils.graph_export_utils import GraphExporter

                        subgraph_data = GraphExporter.export_subgraph(
                            graph_store=graph_store,
                            subgraph_node_indices=set(subgraph_info["subgraph_nodes"]),
                            seed_entity_ids=set(subgraph_info["seed_entity_ids"]),
                            retrieved_chunk_ids=subgraph_info["retrieved_chunk_ids"],
                            node_ppr_scores=subgraph_info.get("node_ppr_scores", {}),
                        )
                    if subgraph_data is not None:
                        logger.info(
                            "Exported subgraph: %d nodes, %d edges",
                            len(subgraph_data.get("nodes", [])),
                            len(subgraph_data.get("edges", [])),
                        )
                else:
                    logger.warning("Graph store not found in retriever")
            except Exception as exc:  # noqa: BLE001
                logger.warning("Failed to export subgraph: %s", exc)
                import traceback
                logger.debug("Traceback: %s", traceback.format_exc())
            self._emit_progress(
                progress_callback,
                {
                    "stage": "subgraph_export",
                    "status": "end",
                    "duration_ms": int((time.perf_counter() - export_start) * 1000),
                    "nodes": len((subgraph_data or {}).get("nodes", []) or []),
                    "edges": len((subgraph_data or {}).get("edges", []) or []),
                },
            )

        messages: List[Dict[str, str]] = []
        
        # 如果有历史对话，先添加历史消息（参考 WebSocket 的实现）
        if history_text:
            # 解析历史文本为消息列表
            history_lines = history_text.strip().split("\n")
            for line in history_lines:
                if ":" in line:
                    role, content = line.split(":", 1)
                    role = role.strip()
                    content = content.strip()
                    if role in ("user", "assistant") and content:
                        messages.append({"role": role, "content": content})
        
        for i, chunk in enumerate(chunks):
            metadata = getattr(chunk, "metadata", None) or {}
            chunk_text = metadata.get("prompt_text") or metadata.get("index_text")
            if not isinstance(chunk_text, str) or not chunk_text.strip():
                chunk_text = chunk.content
            messages.append({"role": "user", "content": f"Chunk {i+1}:\n{chunk_text}"})
        messages.append({"role": "user", "content": f"Based on the above chunks, please answer question: {rewritten_query}"})
        logger.info("Invoked chat with query: %s (owner_id=%s)", query, owner_id)
        logger.info("Query rewritten to: %s", rewritten_query)
        if history_text:
            history_count = len([m for m in messages if m.get("role") in ("user", "assistant") and "Chunk" not in m.get("content", "")])
            logger.info("Including history: %d history messages", history_count)
        logger.info("Prepared %d messages for LLM", len(messages))
        return (messages, chunks, subgraph_data, subgraph_info)

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

        filtered_chunks: List[Chunk] = []
        for chunk in chunks:
            file_id = None
            if hasattr(chunk, "metadata") and chunk.metadata:
                file_id = chunk.metadata.get("source_file_id")

            if not file_id or knowledge_module.is_file_active(file_id):
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

        overview = GraphExporter.export_full_graph(
            graph_store=graph_store,
            max_nodes=max_nodes,
            max_edges=max_edges,
            include_node_types=include_node_types,
            owner_id=normalized_owner,
            owner_scope_label=owner_scope_label,
        )
        logger.info(
            "Exported graph overview (owner_scope=%s) with %d nodes and %d edges",
            normalized_owner,
            len(overview.get('nodes', [])) + len(overview.get('chunks', [])),
            len(overview.get('edges', [])),
        )
        return overview

    def _generate_mindmap(self, query: str, response: str, chunks: list[Chunk]) -> tuple[Dict[str, Any], str | None]:
        """
        Generate mind map data based on query, response and chunks

        Args:
            query: User query
            response: LLM response
            chunks: Retrieved chunks

        Returns:
            Tuple of (subgraph_data, raw_mindmap_response)
        """
        try:
            # Prepare prompt for LLM to generate mind map structure
            mindmap_prompt = self._build_mindmap_prompt(query, response, chunks)

            # Call LLM to generate nodes and edges
            mindmap_messages = [
                {"role": "system", "content": MINDMAP_GENERATION_SYSTEM_PROMPT_ZH},
                {"role": "user", "content": mindmap_prompt}
            ]

            logger.info("Generating mind map structure with LLM...")
            logger.info("Mindmap prompt: %s", mindmap_prompt)
            # Note: This is called from _run_blocking, so it's already in a thread
            mindmap_response = self.llm.chat(mindmap_messages)

            # Log raw mindmap response
            logger.info("Raw mindmap response: %s", mindmap_response)

            # Parse LLM response to extract JSON
            mindmap_json = self._extract_json_from_response(mindmap_response)

            # Build final subgraph data with chunks
            subgraph_data = self._build_subgraph_data(chunks, mindmap_json)

            logger.info(f"Generated mind map: {len(subgraph_data.get('nodes', []))} nodes, {len(subgraph_data.get('edges', []))} edges")
            return subgraph_data, mindmap_response

        except Exception as e:
            logger.error(f"Failed to generate mind map: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return {"chunks": [], "nodes": [], "edges": []}

    def _build_mindmap_prompt(self, query: str, response: str, chunks: list[Chunk]) -> str:
        """Build prompt for LLM to generate mind map structure"""
        chunks_text = "\n\n".join([f"Chunk {i+1}:\n{chunk.content}" for i, chunk in enumerate(chunks)])
        return build_mindmap_generation_user_prompt(query=query, response=response, chunks_text=chunks_text)

    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON from LLM response"""
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

            return json.loads(json_str)
        except Exception as e:
            logger.error(f"Failed to extract JSON from response: {e}")
            logger.debug(f"Response: {response}")
            return {"nodes": [], "edges": []}

    def _build_subgraph_data(self, chunks: list[Chunk], mindmap_json: Dict[str, Any]) -> Dict[str, Any]:
        """Build final subgraph data combining chunks and mind map structure"""
        # Build chunks data
        chunks_data = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"chunk-{800 + i + 1}"
            # Try to infer chunk type from content or use default
            chunk_type = "检索片段"
            chunks_data.append({
                "id": chunk_id,
                "type": chunk_type,
                "content": chunk.content
            })

        # Combine with mind map nodes and edges
        return {
            "chunks": chunks_data,
            "nodes": mindmap_json.get("nodes", []),
            "edges": mindmap_json.get("edges", [])
        }
