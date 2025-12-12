from encapsulation.data_model.schema import Chunk
from typing import TYPE_CHECKING, Optional, Dict, Any, List
import logging
import uuid
from framework.module import AbstractModule
from core.utils.owner_guard import normalize_owner_id, is_admin_owner, get_admin_owner_id
from framework.register import Register

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

    def chat(
        self,
        query: str,
        owner_id: uuid.UUID,
        return_subgraph: bool = False,
    ) -> tuple[str, list[Chunk], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """
        Chat with RAG system

        Args:
            query: User query
            owner_id: User ID for user-isolated retrieval
        return_subgraph: If True, include serialized subgraph data in the response

        Returns:
            Tuple of (LLM response, chunks, subgraph_data, subgraph_info)
            - subgraph_data is None if return_subgraph=False or retriever doesn't support it
            - subgraph_info mirrors the retriever diagnostics used for graph export when available
        """
        rewritten_query = self.query_rewriter.rewrite_query(query)

        # Pass owner_id and return_subgraph_info to the configured retriever.
        chunks: list[Chunk] = self.retriever.invoke(
            rewritten_query,
            owner_id=owner_id,
            return_subgraph_info=return_subgraph
        )

        # Admin/global mode: if multipath returns nothing, fall back to the graph retriever.
        if owner_id is not None and is_admin_owner(owner_id) and not chunks and self.graph_retriever is not None:
            logger.info("Admin/global mode: multipath returned 0 results, falling back to graph retriever")
            graph_chunks = self.graph_retriever.invoke(
                rewritten_query,
                owner_id=owner_id,
                return_subgraph_info=return_subgraph
            )
            if graph_chunks:
                chunks = graph_chunks

        chunks = self._filter_chunks_by_file_status(chunks)

        # Extract subgraph info BEFORE reranking (to avoid losing it after reordering)
        subgraph_info = self._consume_subgraph_info(chunks)

        chunks = self.reranker.rerank(rewritten_query, chunks)

        # Export subgraph data if subgraph_info is available
        subgraph_data = None
        if subgraph_info and return_subgraph:
            # Import GraphExporter here to avoid circular dependency
            try:
                graph_store = self._locate_graph_store()

                if graph_store:
                    # Import appropriate GraphExporter based on graph_store type
                    # Check by class name to avoid import issues
                    graph_store_class_name = graph_store.__class__.__name__

                    if graph_store_class_name == 'PrunedHippoRAGNeo4jStore':
                        from encapsulation.database.utils.graph_export_utils_neo4j import GraphExporterNeo4j as GraphExporter
                        # Neo4j version uses node IDs (strings)
                        subgraph_data = GraphExporter.export_subgraph(
                            graph_store=graph_store,
                            subgraph_node_ids=set(subgraph_info['subgraph_nodes']),
                            seed_entity_ids=set(subgraph_info['seed_entity_ids']),
                            retrieved_chunk_ids=subgraph_info['retrieved_chunk_ids'],
                            node_ppr_scores=subgraph_info.get('node_ppr_scores', {})
                        )
                    else:
                        from encapsulation.database.utils.graph_export_utils import GraphExporter
                        # igraph version uses node indices (integers)
                        subgraph_data = GraphExporter.export_subgraph(
                            graph_store=graph_store,
                            subgraph_node_indices=set(subgraph_info['subgraph_nodes']),
                            seed_entity_ids=set(subgraph_info['seed_entity_ids']),
                            retrieved_chunk_ids=subgraph_info['retrieved_chunk_ids'],
                            node_ppr_scores=subgraph_info.get('node_ppr_scores', {})
                        )
                    logger.info(f"Exported subgraph: {len(subgraph_data.get('nodes', []))} nodes, {len(subgraph_data.get('edges', []))} edges")
                else:
                    logger.warning("Graph store not found in retriever")
            except Exception as e:
                logger.warning(f"Failed to export subgraph: {e}")
                import traceback
                logger.debug(f"Traceback: {traceback.format_exc()}")

        # Format chunks and query as messages
        messages = []
        for i, chunk in enumerate(chunks):
            chunk_content = f"Chunk {i+1}:\n{chunk.content}"
            messages.append({"role": "user", "content": chunk_content})
        messages.append({"role": "user", "content": f"Based on the above chunks, please answer question: {rewritten_query}"})
        logger.info(f"Invoked chat with query: {query} (owner_id={owner_id})")
        logger.info(f"Query rewritten to: {rewritten_query}")
        logger.info(f"Retrieved chunks: {[getattr(chunk, 'content', str(chunk)) for chunk in chunks]}")
        logger.info(f"Reranked chunks: {[getattr(chunk, 'content', str(chunk)) for chunk in chunks]}")
        logger.info(f"Prepared messages for LLM: {messages}")
        response = self.llm.chat(messages)
        return (response, chunks, subgraph_data, subgraph_info)

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
