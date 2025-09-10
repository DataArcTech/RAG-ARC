import asyncio
import logging
from typing import Any, List, Optional, Literal, Union, Annotated, Dict
from pydantic import ConfigDict, Field, field_validator, model_validator
import warnings

from core.retrieval.base import BaseRetriever, BaseRetrieverConfig
from core.utils.data_model import Document
from core.utils.Fusion import FusionMethod, RRFusion
from encapsulation.database.vector_db.faiss import FaissVectorDBConfig
from encapsulation.database.bm25_indexer import BM25IndexBuilderConfig

logger = logging.getLogger(__name__)


class MultiPathRetrieverConfig(BaseRetrieverConfig):
    """Configuration for MultiPath Retriever"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    type: Literal["multipath"] = "multipath"
    
    indexers: List[Annotated[Union[FaissVectorDBConfig, BM25IndexBuilderConfig], Field(discriminator="type")]] = Field(
        default_factory=list,
        description="List of indexer config objects. Each must provide build() to create indexer and support loading + as_retriever()"
    )
    
    # Runtime built retrievers (populated after build)
    built_retrievers: Optional[List[BaseRetriever]] = Field(
        default=None,
        exclude=True,
        description="Built retriever instances (internal use only)"
    )
    
    fusion_method: FusionMethod = Field(
        default_factory=RRFusion,
        exclude=True,
        description="Fusion method for merging results from multiple retrievers"
    )
    
    # Retrieval parameters
    top_k_per_retriever: int = Field(
        default=50,
        gt=0,
        description="Number of results returned by each retriever",
        exclude=True
    )
    
    # Explicitly redefine search_kwargs to ensure proper default behavior
    search_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional search parameters",
        exclude=True
    )
    
    @model_validator(mode='after')
    def validate_indexers_and_k(self) -> 'MultiPathRetrieverConfig':
        """Validate indexers presence and k/top_k_per_retriever relation"""
        if not isinstance(self.indexers, list) or len(self.indexers) == 0:
            raise ValueError("At least one indexer config is required")

        if self.k > self.top_k_per_retriever:
            raise ValueError(
                f"k ({self.k}) must be less than or equal to top_k_per_retriever ({self.top_k_per_retriever}). "
                f"Each retriever can only return at most {self.top_k_per_retriever} results, "
                f"so the final result cannot exceed this limit."
            )
        return self
    
    def _get_retrievers(self) -> List[BaseRetriever]:
        """
        Get retriever instances based on the configured indexer configurations.

        This method iterates over each indexer configuration in self.indexers, builds an indexer instance,
        loads local data if possible, and then converts the indexer instance into a retriever.

        Returns:
            List[BaseRetriever]: A list of built retriever instances.
        """
        retrieved_retrievers = []
        for idx, indexer_config in enumerate(self.indexers):
            try:
                # Check if the indexer configuration provides a build() method
                if not hasattr(indexer_config, 'build') or not callable(getattr(indexer_config, 'build')):
                    raise TypeError(f"Indexer config at position {idx} does not provide a build() method")

                # Build the indexer instance
                indexer_instance = indexer_config.build()

                # Try to load local data if the indexer instance supports it and the index path is configured
                if hasattr(indexer_instance, 'load_local'):
                    try:
                        if hasattr(indexer_config, 'index_path'):
                            indexer_instance.load_local()
                    except Exception:
                        pass

                # Check if the indexer instance provides an as_retriever() method
                if not hasattr(indexer_instance, 'as_retriever'):
                    raise TypeError(f"Indexer instance built from position {idx} does not provide as_retriever()")

                # Convert the indexer instance into a retriever
                retriever = indexer_instance.as_retriever(
                    k=self.k,
                    with_score=self.with_score,
                    search_kwargs=self.search_kwargs.copy()
                )
                retrieved_retrievers.append(retriever)

            except Exception as e:
                logger.error(f"Failed to build retriever from indexer config at position {idx}: {e}")
                raise

        return retrieved_retrievers
    
    def build(self) -> "MultiPathRetriever":
        """Build the MultiPathRetriever instance"""

        built_retrievers = self._get_retrievers()
        
        # Ensure there is at least one retriever
        if not built_retrievers:
            raise ValueError("No retrievers available. Please provide at least one retriever reference.")
        
        # Store built retrievers in the internal field
        self.built_retrievers = built_retrievers
        
        return MultiPathRetriever(config=self)


class MultiPathRetriever(BaseRetriever[MultiPathRetrieverConfig]):
    """
    MultiPathRetriever is a multi-path document retriever that can use multiple retrievers simultaneously for document retrieval. 
    It merges and sorts the results from multiple retrievers using a specified fusion method.

    This class implements multi-path retrieval functionality, supporting the combination of results from different retrieval algorithms 
    (e.g., BM25, vector retrieval, etc.) to improve retrieval accuracy and robustness.
    
    Key Features:
    - Supports parallel execution of multiple retrievers
    - Supports configurable fusion methods (Reciprocal Rank Fusion by default)
    - Compatible with both synchronous and asynchronous calls
    - Provides dynamic addition and removal of retrievers
    - Ensures configuration security through Pydantic parameter validation

    Configuration Parameters (from config):
        retrievers (List[Any]): List of retrievers, each of which needs to implement the invoke method
        fusion_method (FusionMethod): Fusion method for merging results from multiple retrievers
        top_k_per_retriever (int): Number of results returned by each retriever
        k (int): Default number of documents to return
        with_score (bool): Whether to include relevance scores by default
        search_kwargs (dict): Additional search parameters

    Core Methods:
        - invoke: Main entry point for synchronous retrieval
        - _get_relevant_documents: Core retrieval implementation
        - add_retriever/remove_retriever: Dynamically manage retrievers
        - set_fusion_method: Set the fusion method

    Performance Considerations:
        - Each retriever runs independently, and the overall performance depends on the slowest retriever
        - The fusion process adds extra computational overhead
        - For scenarios with high real-time requirements, it is recommended to optimize the performance of individual retrievers

    Typical Usage:
        >>> config = MultiPathRetrieverConfig(
        ...     retrievers=[bm25_config, vector_config],
        ...     fusion_method=RRFusion(),
        ...     top_k_per_retriever=50
        ... )
        >>> multi_retriever = config.build()
        >>> results = multi_retriever.invoke("Query statement")
    """
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """
        Retrieve documents relevant to the query.
        
        This method invokes all configured retrievers, collects the retrieval results from each retriever,
        and then merges and sorts all results using the specified fusion method.
        
        Args:
            query: Query string.
            **kwargs: Other parameters, including k, etc.
            
        Returns:
            A list of fused relevant documents sorted by relevance.
            
        Note:
            - Each retriever returns a list of Document objects.
            - The fused results return sorted Document objects with scores stored in metadata['score'].
        """
        # Use default configuration values
        top_k = kwargs.get('k', self.config.k)
        top_k_per_retriever = kwargs.get('top_k_per_retriever', self.config.top_k_per_retriever)
        
        # Validate parameters
        if top_k <= 0:
            raise ValueError(f"Parameter 'k' must be greater than 0, got {top_k}")
        
        if top_k > top_k_per_retriever:
            raise ValueError(
                f"k ({top_k}) must be less than or equal to top_k_per_retriever ({top_k_per_retriever}). "
                f"Each retriever can return at most {top_k_per_retriever} results, "
                f"so the final result cannot exceed this limit."
            )
        
        if not query.strip():
            logger.info("Empty query, returning empty results")
            return []
        
        all_results = []
        for retriever in self.config.built_retrievers:
            try:
                # Pass the correct parameters when invoking each retriever
                retriever_kwargs = {**kwargs, 'k': top_k_per_retriever}
                documents = retriever.invoke(query, **retriever_kwargs)
                
                # Ensure each document has a score in its metadata
                for doc in documents:
                    if doc.metadata is None:
                        doc.metadata = {}
                    # Use a default score of 1.0 if no score is provided
                    if 'score' not in doc.metadata:
                        doc.metadata['score'] = 1.0
                
                all_results.append(documents)
                logger.debug(f"Retriever {type(retriever).__name__} returned {len(documents)} results")
                
            except Exception as e:
                logger.error(f"Retriever {type(retriever).__name__} failed to execute: {e}")
                warnings.warn(f"Retriever {type(retriever).__name__} failed to execute: {e}", RuntimeWarning)
                all_results.append([])
        
        if not all_results or all(len(results) == 0 for results in all_results):
            logger.warning("All retrievers returned no results")
            return []
        
        fused_documents = self.config.fusion_method.fuse(all_results, top_k)
        logger.info(f"Fused {len(fused_documents)} results using {type(self.config.fusion_method).__name__}")
        
        return fused_documents

    def add_retriever(self, retriever: Any) -> None:
        """
        Add a new retriever to the multi-path retriever.
        
        Args:
            retriever: The retriever instance to add.
        """
        if not hasattr(retriever, 'invoke'):
            raise ValueError(f"Retriever {type(retriever).__name__} must implement the invoke method")
        
        if self.config.built_retrievers is None:
            self.config.built_retrievers = []
        self.config.built_retrievers.append(retriever)
        logger.info(f"Added retriever {type(retriever).__name__}")
    
    def remove_retriever(self, name: str) -> bool:
        """
        Remove the specified retriever.
        
        Args:
            name: The class name of the retriever to remove.
            
        Returns:
            Whether the removal was successful.
            
        Note:
            This method identifies the retriever to remove by comparing the class names of retrievers.
        """
        if self.config.built_retrievers is None:
            return False
            
        for i, retriever in enumerate(self.config.built_retrievers):
            if hasattr(retriever, '__class__') and retriever.__class__.__name__ == name:
                removed_retriever = self.config.built_retrievers.pop(i)
                logger.info(f"Removed retriever {type(removed_retriever).__name__}")
                return True
        logger.warning(f"Retriever {name} not found")
        return False
    
    def set_fusion_method(self, fusion_method: FusionMethod) -> None:
        """
        Set the fusion method.
        
        Args:
            fusion_method: The new fusion method instance.
        """
        self.config.fusion_method = fusion_method
        logger.info(f"Set fusion method to {type(fusion_method).__name__}")

    def get_multipath_info(self) -> dict:
        """Get information about the multi-path retriever."""
        retrievers = self.config.built_retrievers or []
        return {
            "retriever_count": len(retrievers),
            "retriever_types": [type(retriever).__name__ for retriever in retrievers],
            "fusion_method": type(self.config.fusion_method).__name__,
            "top_k_per_retriever": self.config.top_k_per_retriever,
            "k": self.config.k,
            "with_score": self.config.with_score,
            "search_kwargs": self.config.search_kwargs
        }
    
    def get_name(self) -> str:
        """Get the name of the retriever."""
        retrievers = self.config.built_retrievers or []
        retriever_names = [type(r).__name__ for r in retrievers]
        return f"MultiPath[{','.join(retriever_names)}]"