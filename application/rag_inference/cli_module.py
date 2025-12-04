import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from encapsulation.data_model.schema import Chunk
from core.retrieval.graph_retrieveal.base import BaseGraphRetriever
from .module import RAGInference

logger = logging.getLogger(__name__)


@dataclass
class PipelineArtifacts:
    """Intermediate outputs captured from the RAG pipeline."""

    original_query: str
    rewritten_query: str
    retrieved_chunks: List[Chunk]
    reranked_chunks: List[Chunk]
    messages: List[Dict[str, str]]
    subgraph_data: Optional[Dict[str, Any]]
    llm_response: Optional[str]


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
        rewritten_query = self._rag.query_rewriter.rewrite_query(query)
        logger.info("Query rewritten from '%s' to '%s'", original_query, rewritten_query)

        chunks = retriever.invoke(
            rewritten_query,
            owner_id=owner_id,
            return_subgraph_info=return_subgraph,
        )
        logger.info("Retriever returned %d chunks", len(chunks))

        subgraph_info = self._extract_subgraph_info(chunks) if return_subgraph else None
        reranked_chunks = self._rag.reranker.rerank(rewritten_query, chunks)
        logger.info("Reranker produced %d chunks", len(reranked_chunks))

        subgraph_data = self._export_subgraph(subgraph_info) if subgraph_info else None
        messages = self._build_messages(reranked_chunks, rewritten_query)
        llm_response = None
        if not skip_llm:
            llm_response = self._rag.llm.chat(messages)
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
        return self._locate_graph_store()

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
        messages: List[Dict[str, str]] = []
        for idx, chunk in enumerate(chunks):
            chunk_content = f"Chunk {idx + 1}:\n{chunk.content}"
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
            graph_store = self._locate_graph_store()
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
        graph_retriever = self._get_graph_retriever()
        if graph_retriever and hasattr(graph_retriever, "graph_store"):
            return graph_retriever.graph_store
        retriever = self._rag.retriever
        if hasattr(retriever, "graph_store"):
            return retriever.graph_store
        if hasattr(retriever, "config") and hasattr(retriever.config, "built_retrievers"):
            for child_retriever in retriever.config.built_retrievers:
                if hasattr(child_retriever, "graph_store"):
                    return child_retriever.graph_store
        return None
