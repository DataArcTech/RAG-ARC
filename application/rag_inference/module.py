from typing import TYPE_CHECKING
import logging
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

    def chat(self, query: str, owner_id: str = None) -> str:
        """
        Chat with RAG system

        Args:
            query: User query
            owner_id: User ID for user-isolated retrieval

        Returns:
            LLM response
        """
        query = self.query_rewriter.rewrite_query(query)

        # Pass owner_id to retriever for user isolation
        chunks = self.retriever.invoke(query, owner_id=owner_id)
        chunks = self.reranker.rerank(query, chunks)

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
        return response