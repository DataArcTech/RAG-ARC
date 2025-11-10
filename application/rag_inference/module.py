from encapsulation.data_model.schema import Chunk
from typing import TYPE_CHECKING, Optional, Dict, Any
import logging
import uuid
from framework.module import AbstractModule

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from config.application.rag_inference_config import RAGInferenceConfig
 
class RAGInference(AbstractModule):
    def __init__(self, config: 'RAGInferenceConfig'):
        super().__init__(config=config)
        logger.info("Building query_rewriter...")
        self.query_rewriter = self.config.query_rewrite_config.build()
        logger.info("Query rewriter built successfully")
        
        logger.info("Building retriever...")
        self.retriever = self.config.retrieval_config.build()
        logger.info("Retriever built successfully")
        
        logger.info("Building reranker...")
        self.reranker = self.config.reranker_config.build()
        logger.info("Reranker built successfully")
        
        logger.info("Building llm...")
        self.llm = self.config.llm_config.build()
        logger.info("LLM built successfully")

    def chat(self, query: str, owner_id: uuid.UUID, return_subgraph: bool = False) -> tuple[str, list[Chunk], Optional[Dict[str, Any]]]:
        """
        Chat with RAG system

        Args:
            query: User query
            owner_id: User ID for user-isolated retrieval
            return_subgraph: If True, return subgraph visualization data

        Returns:
            Tuple of (LLM response, chunks, subgraph_data)
            - subgraph_data is None if return_subgraph=False or retriever doesn't support it
        """
        query = self.query_rewriter.rewrite_query(query)

        # Pass owner_id and return_subgraph_info to retriever
        # All retrievers support invoke() method which will handle these parameters
        chunks = self.retriever.invoke(
            query,
            owner_id=owner_id,
            return_subgraph_info=return_subgraph
        )

        # Extract subgraph info BEFORE reranking (to avoid losing it after reordering)
        subgraph_info = None
        if return_subgraph and chunks:
            for chunk in chunks:
                if hasattr(chunk, 'metadata') and chunk.metadata and '_subgraph_info' in chunk.metadata:
                    subgraph_info = chunk.metadata.pop('_subgraph_info')
                    logger.info("Extracted subgraph info before reranking")
                    break

        chunks = self.reranker.rerank(query, chunks)

        # Export subgraph data if subgraph_info is available
        subgraph_data = None
        if subgraph_info:
            # Import GraphExporter here to avoid circular dependency
            try:
                # Find graph_store from retriever or its children
                graph_store = None
                if hasattr(self.retriever, 'graph_store'):
                    # Direct graph retriever
                    graph_store = self.retriever.graph_store
                elif hasattr(self.retriever, 'config') and hasattr(self.retriever.config, 'built_retrievers'):
                    # Multipath retriever: find graph retriever
                    for child_retriever in self.retriever.config.built_retrievers:
                        if hasattr(child_retriever, 'graph_store'):
                            graph_store = child_retriever.graph_store
                            break

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
        messages.append({"role": "user", "content": f"Based on the above chunks, please answer question: {query}"})
        logger.info(f"Invoked chat with query: {query} (owner_id={owner_id})")
        logger.info(f"Query rewritten to: {self.query_rewriter.rewrite_query(query)}")
        logger.info(f"Retrieved chunks: {[getattr(chunk, 'content', str(chunk)) for chunk in chunks]}")
        logger.info(f"Reranked chunks: {[getattr(chunk, 'content', str(chunk)) for chunk in chunks]}")
        logger.info(f"Prepared messages for LLM: {messages}")
        response = self.llm.chat(messages)
        return (response, chunks, subgraph_data)
