import asyncio
import logging
from typing import Any, List, Optional
from pydantic import BaseModel, ConfigDict, Field
import warnings

from core.utils.data_model import Document
from core.utils.Fusion import FusionMethod, RRFusion, RetrievalResult

logger = logging.getLogger(__name__)

class MultiPathRetriever(BaseModel):
    """
    MultiPathRetriever is a multi-path document retriever that can use multiple retrievers simultaneously for document retrieval and merge and sort the results of multiple retrievers through a specified fusion method.

    This class implements multi-path retrieval functionality, supporting the combination of results from different retrieval algorithms (such as BM25, vector retrieval, etc.) to improve retrieval accuracy and robustness.
    
    Key features:
    - Supports multiple retrievers running in parallel
    - Supports configurable fusion methods (default is Reciprocal Rank Fusion)
    - Compatible with synchronous and asynchronous invocation
    - Provides dynamic addition and removal of retrievers
    - Validates parameters through Pydantic to ensure configuration safety

    Main parameters:
        retrievers (List[Any]): Retriever list, each retriever needs to implement the invoke method
        fusion_method (FusionMethod): Fusion method for merging results from multiple retrievers
        top_k_per_retriever (int): Number of results returned by each retriever

    Core methods:
        - invoke/ainvoke: Synchronous/asynchronous invocation entry point
        - add_retriever/remove_retriever: Dynamically manage retrievers
        - set_fusion_method: Set fusion method
        - _get_relevant_documents: Core retrieval implementation

    Performance considerations:
        - Each retriever runs independently, and the overall performance depends on the slowest retriever
        - The fusion process adds additional computation overhead
        - For scenarios with high real-time requirements, it is recommended to optimize the performance of individual retrievers

    Typical usage:
        >>> from core.retrieval.bm25 import BM25Retriever
        >>> from core.retrieval.dense import VectorStoreRetriever
        >>> bm25_retriever = BM25Retriever.from_texts(["text1", "text2"])
        >>> vector_retriever = VectorStoreRetriever(vectorstore=vectorstore)
        >>> multi_retriever = MultiPathRetriever(retrievers=[bm25_retriever, vector_retriever])
        >>> results = multi_retriever.invoke("query statement")

    Attributes:
        retrievers: Retriever list
        fusion_method: Fusion method
        top_k_per_retriever: Number of results returned by each retriever
    """
    
    retrievers: List[Any] = Field(
        default_factory=list, 
        description="Retriever list, each retriever needs to implement the invoke method"
    )
    fusion_method: FusionMethod = Field(
        default_factory=RRFusion,
        description="Fusion method for merging results from multiple retrievers"
    )
    top_k_per_retriever: int = Field(
        default=50,
        gt=0,
        description="Number of results returned by each retriever"
    )
    
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )
    
    def __init__(self, **kwargs):
        """Initialize MultiPath Retriever
        
        Args:
            retrievers: Retriever list
            fusion_method: Fusion method
            top_k_per_retriever: Number of results returned by each retriever
            **kwargs: Other parameters
        """
        super().__init__(**kwargs)
    
    def invoke(self, input: str, **kwargs: Any) -> List[Document]:
        """Invoke retriever to get relevant documents
        
        Main entry point for synchronous retriever invocation.
        
        Args:
            input: Query string
            **kwargs: Other parameters passed to the retriever
            
        Returns:
            List of relevant documents
            
        Examples:
            >>> retriever.invoke("query")
        """
        return self._get_relevant_documents(input, **kwargs)
    
    async def ainvoke(self, input: str, **kwargs: Any) -> List[Document]:
        """Asynchronously invoke retriever to get relevant documents
        
        Main entry point for asynchronous retriever invocation.
        
        Args:
            input: Query string
            **kwargs: Other parameters passed to the retriever
            
        Returns:
            List of relevant documents
            
        Examples:
            >>> await retriever.ainvoke("query")
        """
        try:
            return await asyncio.to_thread(self._get_relevant_documents, input, **kwargs)
        except AttributeError:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self._get_relevant_documents, input, **kwargs)
    
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """
        Get documents related to the query
        
        This method will call all configured retrievers, get the retrieval results of each retriever,
        and then use the specified fusion method to merge and sort all results.
        
        Args:
            query: Query string
            **kwargs: Other parameters, including top_k, etc.
            
        Returns:
            List of fused relevant documents, sorted by relevance
            
        Note:
            - Each retriever's results will be converted to RetrievalResult format
            - Input will only be Document objects
            - The fused results only return sorted Document objects
        """
        top_k = kwargs.get('k', 10)
        
        all_results = []
        for retriever in self.retrievers:
            try:
                documents = retriever.invoke(query, **{**kwargs, 'k': self.top_k_per_retriever})
                
                formatted_results = []
                for i, doc in enumerate(documents):
                    retrieval_result = RetrievalResult(
                        document=doc,
                        score=getattr(doc, 'score', 1.0),
                        rank=i + 1
                    )
                    formatted_results.append(retrieval_result)
                
                all_results.append(formatted_results)
                logger.debug(f"Retriever {type(retriever).__name__} returned {len(formatted_results)} results")
                
            except Exception as e:
                logger.error(f"Retriever {type(retriever).__name__} failed: {e}")
                warnings.warn(f"Retriever {type(retriever).__name__} execution failed: {e}", RuntimeWarning)
                all_results.append([])
        
        if not all_results or all(len(results) == 0 for results in all_results):
            logger.warning("No results from any retriever")
            return []
        
        fused_results = self.fusion_method.fuse(all_results, top_k)
        logger.info(f"Fused {len(fused_results)} results using {type(self.fusion_method).__name__}")
        
        documents = []
        for result in fused_results:
            documents.append(result.document)
        
        return documents

    def add_retriever(self, retriever: Any) -> None:
        """
        Add a new retriever to the multipath retriever
        
        Args:
            retriever: Retriever instance to be added
        """
        self.retrievers.append(retriever)
        logger.info(f"Added retriever {type(retriever).__name__}")
    
    def remove_retriever(self, name: str) -> bool:
        """
        Remove the specified retriever
        
        Args:
            name: Class name of the retriever to be removed
            
        Returns:
            Whether the removal was successful
            
        Note:
            This method identifies the retriever to be removed by comparing the retriever's class name
        """
        for i, retriever in enumerate(self.retrievers):
            if hasattr(retriever, '__class__') and retriever.__class__.__name__ == name:
                removed_retriever = self.retrievers.pop(i)
                logger.info(f"Removed retriever {type(removed_retriever).__name__}")
                return True
        logger.warning(f"Retriever {name} not found")
        return False
    
    def set_fusion_method(self, fusion_method: FusionMethod) -> None:
        """
        Set fusion method
        
        Args:
            fusion_method: New fusion method instance
        """
        self.fusion_method = fusion_method
        logger.info(f"Set fusion method to {type(fusion_method).__name__}")

    def get_multipath_info(self) -> dict:
        """Get multipath retriever information"""
        return {
            "retriever_count": len(self.retrievers),
            "retriever_types": [type(retriever).__name__ for retriever in self.retrievers],
            "fusion_method": type(self.fusion_method).__name__,
            "top_k_per_retriever": self.top_k_per_retriever
        }