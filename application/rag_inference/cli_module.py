import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple

from encapsulation.data_model.schema import Chunk
from core.retrieval.graph_retrieveal.base import BaseGraphRetriever
from encapsulation.data_model.pipeline import PipelineArtifacts
from config.output_limits import CHAT_MAX_IMAGE_INPUTS
from core.utils.multimodal_images import collect_image_paths_from_chunk_payloads
from core.utils.multimodal_llm import build_multimodal_user_message
from core.prompts import get_rag_chat_system_prompt
from config.retrieval_routing import rag_retrieval_dynamic_quota_enabled
from .module import RAGInference

logger = logging.getLogger(__name__)


class RAGInferenceCLIModule:
    """Lightweight adapter that exposes pipeline internals for CLI usage."""

    def __init__(self, rag_inference: RAGInference):
        self._rag = rag_inference

    def run_pipeline(
        self,
        query: str,
        owner_id: Optional[uuid.UUID],
        return_subgraph: bool = False,
        skip_llm: bool = False,
    ) -> PipelineArtifacts:
        if owner_id is None:
            raise ValueError("owner_id is required for pipeline execution")
        return self._run_with_retriever(
            retriever=self._rag.retriever,
            query=query,
            owner_id=owner_id,
            return_subgraph=return_subgraph,
            skip_llm=skip_llm,
        )

    def run_graph_pipeline(
        self,
        query: str,
        owner_id: Optional[uuid.UUID],
        return_subgraph: bool = True,
        skip_llm: bool = False,
    ) -> PipelineArtifacts:
        if owner_id is None:
            raise ValueError("owner_id is required for graph pipeline execution")
        graph_retriever = self._get_graph_retriever()
        if graph_retriever is None:
            raise RuntimeError("Graph retriever is not configured in the current profile")
        return self._run_with_retriever(
            retriever=graph_retriever,
            query=query,
            owner_id=owner_id,
            return_subgraph=return_subgraph,
            skip_llm=skip_llm,
        )

    def _run_with_retriever(
        self,
        retriever: Any,
        query: str,
        owner_id: Optional[uuid.UUID],
        return_subgraph: bool,
        skip_llm: bool,
    ) -> PipelineArtifacts:
        original_query = query
        rewritten_query = query
        retrieval_ratios: Dict[str, float] | None = None
        bm25_query: str | None = None

        # Keep rewrite+routing behavior aligned with the user-facing rag_inference pipeline.
        if rag_retrieval_dynamic_quota_enabled() and hasattr(self._rag.query_rewriter, "rewrite_query_with_routing"):
            try:
                rewritten_query, retrieval_ratios, bm25_query = self._rag.query_rewriter.rewrite_query_with_routing(query)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                logger.warning("rewrite_query_with_routing failed; falling back to rewrite_query: %s", exc)
                rewritten_query = self._rag.query_rewriter.rewrite_query(query)
                retrieval_ratios = None
                bm25_query = None
        else:
            rewritten_query = self._rag.query_rewriter.rewrite_query(query)
            retrieval_ratios = None
            bm25_query = None

        logger.info("Query rewritten from '%s' to '%s'", original_query, rewritten_query)

        chunks = retriever.invoke(
            rewritten_query,
            owner_id=owner_id,
            return_subgraph_info=return_subgraph,
            **({"bm25_query": bm25_query} if bm25_query else {}),
            **({"retrieval_ratios": retrieval_ratios} if retrieval_ratios else {}),
        )
        logger.info("Retriever returned %d chunks", len(chunks))

        subgraph_info = self._extract_subgraph_info(chunks)
        # Respect the same rerank_keep_k config as the user-facing pipeline.
        rerank_keep_k = 10
        try:
            candidate_cfg = getattr(self._rag.config, "candidate_selection", None)
            if candidate_cfg is not None:
                rerank_keep_k = int(getattr(candidate_cfg, "rerank_keep_k", rerank_keep_k))
        except Exception:  # noqa: BLE001
            pass
        rerank_instruct = "The user's question is: {}. Please find the most relevant passages.".format(rewritten_query)
        try:
            if hasattr(self._rag.reranker, "rerank_llm") and hasattr(self._rag.reranker.rerank_llm, "instruct"):
                self._rag.reranker.rerank_llm.instruct = rerank_instruct
        except Exception:  # noqa: BLE001
            pass
        reranked_chunks = self._rag.reranker.rerank(rewritten_query, chunks, top_k=rerank_keep_k)
        logger.info("Reranker produced %d chunks", len(reranked_chunks))

        subgraph_data = self._export_subgraph(subgraph_info) if (subgraph_info and return_subgraph) else None
        messages = self._build_messages(reranked_chunks, rewritten_query)
        llm_response = None
        if not skip_llm:
            llm = self._rag.llm
            image_paths = collect_image_paths_from_chunk_payloads(reranked_chunks, max_images=CHAT_MAX_IMAGE_INPUTS)
            call_messages: List[Dict[str, Any]] = messages

            supports_multimodal = (
                bool(image_paths)
                and getattr(llm, "loading_method", None) == "openai"
                and hasattr(llm, "client")
                and hasattr(getattr(llm, "client", None), "chat")
            )

            if supports_multimodal:
                try:
                    call_messages = [dict(m) for m in messages]
                    user_idx = len(call_messages) - 1
                    if user_idx >= 0 and isinstance(call_messages[user_idx].get("content"), str):
                        call_messages[user_idx] = build_multimodal_user_message(
                            text=str(call_messages[user_idx]["content"]),
                            image_paths=image_paths,
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to build multimodal messages; continuing with text-only: %s", exc)
                    call_messages = messages

                try:
                    llm_response = llm.chat(call_messages)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Model/API does not support multimodal inputs; continuing with text-only (captions as fallback): %s",
                        exc,
                    )
                    llm_response = llm.chat(messages)
            else:
                if image_paths:
                    logger.warning(
                        "Configured LLM does not support multimodal inputs; continuing with text-only (captions as fallback)."
                    )
                llm_response = llm.chat(messages)
            logger.info("LLM response generated for query '%s'", rewritten_query)
        else:
            logger.info("Skipping LLM call for query '%s'", rewritten_query)

        return PipelineArtifacts(
            original_query=original_query,
            rewritten_query=rewritten_query,
            retrieved_chunks=chunks,
            reranked_chunks=reranked_chunks,
            messages=messages,
            subgraph_data=subgraph_data,
            subgraph_info=subgraph_info,
            llm_response=llm_response,
        )

    def chat(
        self,
        query: str,
        owner_id: uuid.UUID,
        return_subgraph: bool = False,
    ) -> tuple[str, List[Chunk], Optional[Dict[str, Any]]]:
        artifacts = self.run_pipeline(
            query=query,
            owner_id=owner_id,
            return_subgraph=return_subgraph,
            skip_llm=False,
        )
        if artifacts.llm_response is None:
            raise RuntimeError("LLM response missing after pipeline execution")
        return artifacts.llm_response, artifacts.reranked_chunks, artifacts.subgraph_data

    def get_graph_store(self):
        """Expose the underlying graph store for export/debugging."""
        return self._rag.get_graph_store()

    def _get_graph_retriever(self) -> Optional[BaseGraphRetriever]:
        retriever = self._rag.retriever
        if isinstance(retriever, BaseGraphRetriever):
            return retriever

        child_retrievers = []
        if hasattr(retriever, "retrievers"):
            child_retrievers = retriever.retrievers or []
        elif hasattr(getattr(retriever, "config", None), "built_retrievers"):
            child_retrievers = retriever.config.built_retrievers or []

        for candidate in child_retrievers:
            if isinstance(candidate, BaseGraphRetriever):
                return candidate
        return None

    def _build_messages(self, chunks: List[Chunk], query: str) -> List[Dict[str, str]]:
        # CLI is for debugging: use ONLY the base system prompt (no citation/style/domain addons).
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": get_rag_chat_system_prompt(profile="cli")}
        ]
        for idx, chunk in enumerate(chunks):
            metadata = getattr(chunk, "metadata", None) or {}
            chunk_text = metadata.get("prompt_text") or metadata.get("index_text")
            if not isinstance(chunk_text, str) or not chunk_text.strip():
                chunk_text = chunk.content
            chunk_content = f"Chunk {idx + 1}:\n{chunk_text}"
            messages.append({"role": "user", "content": chunk_content})
        messages.append(
            {
                "role": "user",
                "content": f"Based on the above chunks, please answer question: {query}",
            }
        )
        logger.debug("Prepared %d messages for LLM input", len(messages))
        return messages

    def _extract_subgraph_info(self, chunks: List[Chunk]) -> Optional[Dict[str, Any]]:
        for chunk in chunks:
            if hasattr(chunk, "metadata") and chunk.metadata and "_subgraph_info" in chunk.metadata:
                logger.info("Extracted subgraph info before reranking")
                return chunk.metadata.pop("_subgraph_info")
        return None

    def _export_subgraph(self, subgraph_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        try:
            graph_store = self._rag.get_graph_store()
            if not graph_store:
                logger.warning("Graph store not found in retriever")
                return None

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
            logger.info(
                "Exported subgraph: %d nodes, %d edges",
                len(subgraph_data.get("nodes", [])),
                len(subgraph_data.get("edges", [])),
            )
            return subgraph_data
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to export subgraph: %s", exc)
            logger.debug("Subgraph export traceback", exc_info=True)
            return None

    def _locate_graph_store(self):
        return self._rag.get_graph_store()
