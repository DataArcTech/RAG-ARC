#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Retrieval API Interface
Provides a simple and consistent retrieval interface for upper-level services, supporting multiple retriever types.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Literal, TYPE_CHECKING
from pathlib import Path
import shutil
import time

from core.retrieval.base import BaseRetriever
from core.retrieval.dense import DenseRetrieverConfig
from core.retrieval.tantivy_bm25 import TantivyBM25RetrieverConfig
from core.retrieval.multipath import MultiPathRetrieverConfig
from core.utils.data_model import Document

if TYPE_CHECKING:
    from core.retrieval.dense import DenseRetriever
    from core.retrieval.tantivy_bm25 import TantivyBM25Retriever
    from core.retrieval.multipath import MultiPathRetriever
    from encapsulation.database.vector_db.faiss import FaissIndex
    from encapsulation.database.bm25_indexer import BM25IndexBuilder

logger = logging.getLogger(__name__)

RETRIEVER_TYPES = {"dense", "tantivy_bm25", "multipath"}

class RetrievalAPI:
    """
    Unified Retrieval API that provides a simple interface for upper-level services.
    
    Supported retriever types:
    - dense: Dense retrieval based on vector database
    - tantivy_bm25: BM25-based retrieval
    - multipath: Multi-path fusion retrieval
    """
    
    def __init__(self) -> None:
        """
        Initialize the RetrievalAPI instance.
        """
        self.retrievers: Dict[str, BaseRetriever] = {}
        self.configs: Dict[str, Any] = {}
        self.retriever_types: Dict[str, str] = {}
    
    def create_retriever(
        self, 
        name: str, 
        retriever_type: Literal["dense", "tantivy_bm25", "multipath"],
        config: Dict[str, Any]
    ) -> BaseRetriever:
        """
        Create a retriever instance.
        
        Args:
            name (str): The name of the retriever.
            retriever_type (Literal["dense", "tantivy_bm25", "multipath"]): The type of the retriever.
            config (Dict[str, Any]): Configuration dictionary for the retriever.
            
        Returns:
            BaseRetriever: The created retriever instance.
            
        Raises:
            ValueError: If the retriever type is not supported.
            Exception: If failed to create the retriever.
        """
        if retriever_type not in RETRIEVER_TYPES:
            raise ValueError(f"Unsupported retriever type: {retriever_type}")

        try:
            config_map = {
                "dense": DenseRetrieverConfig,
                "tantivy_bm25": TantivyBM25RetrieverConfig,
                "multipath": MultiPathRetrieverConfig
            }
            
            retriever_config = config_map[retriever_type](**config)
            retriever = retriever_config.build()
            
            self.retrievers[name] = retriever
            self.configs[name] = config
            self.retriever_types[name] = retriever_type

            if retriever_type == "tantivy_bm25":
                self._initialize_bm25_index(retriever, name)

            logger.info(f"Created {retriever_type} retriever: {name}")
            return retriever
            
        except Exception as e:
            logger.error(f"Failed to create retriever {name}: {e}")
            raise
    
    def _initialize_bm25_index(self, retriever: BaseRetriever, name: str) -> None:
        """
        Initialize BM25 index.
        
        Args:
            retriever (BaseRetriever): The retriever instance.
            name (str): The name of the retriever.
        """
        try:
            index_instance = getattr(retriever, '_index', None)
            if (index_instance is not None and 
                hasattr(index_instance, '_index') and 
                getattr(index_instance, '_index', None) is None):
                logger.info(f"BM25 index for {name} needs to be built with documents before use")
        except Exception as e:
            logger.debug(f"BM25 index initialization check failed: {e}")
        
    def create_from_config_file(self, name: str, config_path: str) -> BaseRetriever:
        """
        Create a retriever from a configuration file.
        
        Args:
            name (str): The name of the retriever.
            config_path (str): Path to the configuration file.
            
        Returns:
            BaseRetriever: The created retriever instance.
            
        Raises:
            FileNotFoundError: If the configuration file is not found.
            ValueError: If the configuration file does not specify the 'type' field.
            Exception: If failed to create the retriever from the configuration file.
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            retriever_type = config.get("type")
            if not retriever_type:
                raise ValueError("Config file must specify 'type' field")
            
            return self.create_retriever(name, retriever_type, config)
            
        except Exception as e:
            logger.error(f"Failed to create retriever from config file {config_path}: {e}")
            raise
    
    def _get_retriever(self, retriever_name: str) -> BaseRetriever:
        """
        Get a retriever instance by name.
        
        Args:
            retriever_name (str): The name of the retriever.
            
        Returns:
            BaseRetriever: The retriever instance.
            
        Raises:
            ValueError: If the retriever is not found.
        """
        if retriever_name not in self.retrievers:
            raise ValueError(f"Retriever '{retriever_name}' not found")
        return self.retrievers[retriever_name]

    def search(
        self, 
        retriever_name: str, 
        query: str, 
        k: int = 5,
        **kwargs
    ) -> List[Document]:
        """
        Execute a search.
        
        Args:
            retriever_name (str): The name of the retriever.
            query (str): The query text.
            k (int, optional): Number of results to return. Defaults to 5.
            **kwargs: Other search parameters.
            
        Returns:
            List[Document]: List of search result documents.
            
        Raises:
            Exception: If the search fails.
        """
        retriever = self._get_retriever(retriever_name)
        
        try:
            results = retriever.invoke(query, k=k, **kwargs)
            logger.debug(f"Search completed: {len(results)} results for query '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Search failed for retriever {retriever_name}: {e}")
            raise
    
    async def asearch(
        self, 
        retriever_name: str, 
        query: str, 
        k: int = 5,
        **kwargs
    ) -> List[Document]:
        """
        Execute an asynchronous search.
        
        Args:
            retriever_name (str): The name of the retriever.
            query (str): The query text.
            k (int, optional): Number of results to return. Defaults to 5.
            **kwargs: Other search parameters.
            
        Returns:
            List[Document]: List of search result documents.
            
        Raises:
            Exception: If the asynchronous search fails.
        """
        retriever = self._get_retriever(retriever_name)
        
        try:
            results = await retriever.ainvoke(query, k=k, **kwargs)
            logger.debug(f"Async search completed: {len(results)} results for query '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Async search failed for retriever {retriever_name}: {e}")
            raise
    
    def add_documents(self, retriever_name: str, documents: List[Document]) -> List[str]:
        """
        Add documents to the retriever.
        
        Args:
            retriever_name (str): The name of the retriever.
            documents (List[Document]): List of documents to add.
            
        Returns:
            List[str]: List of added document IDs.
            
        Raises:
            AttributeError: If the retriever does not support the add_documents operation.
            Exception: If failed to add documents.
        """
        retriever = self._get_retriever(retriever_name)

        if not hasattr(retriever, 'add_documents'):
            raise AttributeError(f"Retriever '{retriever_name}' does not support add_documents operation")

        try:
            add_documents_method = getattr(retriever, 'add_documents')
            doc_ids = add_documents_method(documents)
            logger.info(f"Added {len(documents)} documents to retriever {retriever_name}")
            return doc_ids
            
        except Exception as e:
            logger.error(f"Failed to add documents to retriever {retriever_name}: {e}")
            raise
    
    def delete_documents(self, retriever_name: str, ids: List[str]) -> bool:
        """
        Delete documents.
        
        Args:
            retriever_name (str): The name of the retriever.
            ids (List[str]): List of document IDs to delete.
            
        Returns:
            bool: True if deletion is successful, False otherwise.
            
        Raises:
            AttributeError: If the retriever does not support the delete_documents operation.
            Exception: If failed to delete documents.
        """
        retriever = self._get_retriever(retriever_name)
        
        if not hasattr(retriever, 'delete_documents'):
            raise AttributeError(f"Retriever '{retriever_name}' does not support delete_documents operation")

        try:
            delete_documents_method = getattr(retriever, 'delete_documents')
            result = delete_documents_method(ids)
            logger.info(f"Deleted {len(ids)} documents from retriever {retriever_name}")
            return result is not False
            
        except Exception as e:
            logger.error(f"Failed to delete documents from retriever {retriever_name}: {e}")
            raise
    
    def build_index(self, retriever_name: str, documents: List[Document]) -> None:
        """
        Build index (only used when index does not exist, will raise exception if index already exists).
        
        Args:
            retriever_name (str): The name of the retriever.
            documents (List[Document]): List of documents.
            
        Raises:
            RuntimeError: If the index already exists.
            Exception: If failed to build the index.
        """
        retriever = self._get_retriever(retriever_name)

        try:
            logger.info(f"Building index for {retriever_name}")

            build_index_method = getattr(retriever, 'build_index')
            build_index_method(documents)

            logger.info(f"Successfully built index for retriever {retriever_name} with {len(documents)} documents")

        except RuntimeError as e:
            logger.warning(f"Cannot build index for retriever {retriever_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to build index for retriever {retriever_name}: {e}")
            raise

    def initialize_index(self, retriever_name: str, documents: List[Document]) -> None:
        """
        Initialize index (force rebuild even if index already exists).
        
        Args:
            retriever_name (str): The name of the retriever.
            documents (List[Document]): List of documents.
            
        Raises:
            Exception: If failed to initialize the index.
        """
        retriever = self._get_retriever(retriever_name)

        try:
            logger.info(f"Force initializing index for {retriever_name}")

            self._reset_index(retriever, retriever_name)
            self._rebuild_index(retriever, retriever_name, documents)

            logger.info(f"Successfully initialized index for {retriever_name} with {len(documents)} documents")

        except Exception as e:
            logger.error(f"Failed to initialize index for retriever {retriever_name}: {e}")
            raise
    
    def _reset_index(self, retriever: BaseRetriever, retriever_name: str) -> None:
        """
        Reset index - optimized version.
        
        Args:
            retriever (BaseRetriever): The retriever instance.
            retriever_name (str): The name of the retriever.
        """
        index = getattr(retriever, '_index', None)
        if index is None:
            return

        reset_handlers = {
            'bm25': self._reset_bm25_index,
            'faiss': self._reset_faiss_index
        }
        
        retriever_type = self.retriever_types.get(retriever_name, '')
        for key, handler in reset_handlers.items():
            if key in retriever_type:
                handler(index, retriever_name)
                break

    def _reset_bm25_index(self, index: Any, retriever_name: str) -> None:
        """
        Reset BM25 index.
        
        Args:
            index (Any): The index instance.
            retriever_name (str): The name of the retriever.
        """
        logger.debug(f"Resetting BM25 index for {retriever_name}")
        setattr(index, '_index', None)
        
        if hasattr(index, '_initialize_index'):
            getattr(index, '_initialize_index')()
        
        if hasattr(index, 'config') and hasattr(index.config, 'index_path'):
            index_path = getattr(index.config, 'index_path', None)
            if index_path:
                os.makedirs(index_path, exist_ok=True)

    def _reset_faiss_index(self, index: Any, retriever_name: str) -> None:
        """
        Reset FAISS index.
        
        Args:
            index (Any): The index instance.
            retriever_name (str): The name of the retriever.
        """
        logger.debug(f"Resetting FAISS index for {retriever_name}")
        setattr(index, 'index', None)
        if hasattr(index, 'docstore'):
            index.docstore.clear()
        if hasattr(index, 'index_to_docstore_id'):
            index.index_to_docstore_id.clear()

    def _rebuild_index(self, retriever: BaseRetriever, retriever_name: str, documents: List[Document]) -> None:
        """
        Rebuild index - optimized version.
        
        Args:
            retriever (BaseRetriever): The retriever instance.
            retriever_name (str): The name of the retriever.
            documents (List[Document]): List of documents.
        """
        index = getattr(retriever, '_index', None)
        if index is None:
            return
            
        retriever_type = self.retriever_types.get(retriever_name, '')
        
        if 'bm25' in retriever_type:
            self._rebuild_bm25_index(retriever, retriever_name, documents)
        else:
            if hasattr(retriever, 'add_documents'):
                add_documents_method = getattr(retriever, 'add_documents')
                add_documents_method(documents)
    
    def _rebuild_bm25_index(self, retriever: BaseRetriever, retriever_name: str, documents: List[Document]) -> None:
        """
        Rebuild BM25 index.
        
        Args:
            retriever (BaseRetriever): The retriever instance.
            retriever_name (str): The name of the retriever.
            documents (List[Document]): List of documents.
        """
        index = getattr(retriever, '_index', None)
        if index is None:
            return
            
        if hasattr(index, 'close'):
            getattr(index, 'close')()

        if hasattr(index, 'config') and hasattr(index.config, 'index_path'):
            index_path = getattr(index.config, 'index_path', None)
            if index_path and os.path.exists(index_path):
                try:
                    shutil.rmtree(index_path)
                    logger.debug(f"Removed existing index directory: {index_path}")
                except OSError as e:
                    logger.warning(f"Failed to remove index directory: {e}, trying again...")
                    time.sleep(0.1)  # Brief wait
                    try:
                        shutil.rmtree(index_path)
                        logger.debug(f"Successfully removed index directory on retry: {index_path}")
                    except OSError:
                        logger.warning(f"Still failed to remove directory, continuing anyway...")
                os.makedirs(index_path, exist_ok=True)

        retriever._index = retriever.config.index_config.build()

        bm25_index = getattr(retriever, '_index', None)
        if bm25_index is not None:
            reset_attrs = ['_index', '_schema', '_tokenizers_registered']
            for attr in reset_attrs:
                if hasattr(bm25_index, attr):
                    setattr(bm25_index, attr, None)

        if bm25_index is not None and hasattr(bm25_index, 'from_documents'):
            from_documents_method = getattr(bm25_index, 'from_documents')
            from_documents_method(documents)

        if hasattr(retriever, 'reload_searcher'):
            reload_searcher_method = getattr(retriever, 'reload_searcher')
            reload_searcher_method()
        elif hasattr(retriever, 'searcher'):
            setattr(retriever, 'searcher', None)
    
    def save_index(self, retriever_name: str, index_path: str, index_name: str = "index") -> None:
        """
        Save index.
        
        Args:
            retriever_name (str): The name of the retriever.
            index_path (str): Path to save the index.
            index_name (str, optional): Name of the index. Defaults to "index".
            
        Raises:
            AttributeError: If the retriever does not support the save_index operation.
            Exception: If failed to save the index.
        """
        retriever = self._get_retriever(retriever_name)
        
        if not hasattr(retriever, 'save_index'):
            raise AttributeError(f"Retriever '{retriever_name}' does not support save_index operation")
        
        try:
            save_index_method = getattr(retriever, 'save_index')
            save_index_method(index_path, index_name)
            logger.info(f"Saved index for retriever {retriever_name} to {index_path}")
            
        except Exception as e:
            logger.error(f"Failed to save index for retriever {retriever_name}: {e}")
            raise
    
    def load_index(self, retriever_name: str, index_path: Optional[str] = None) -> None:
        """
        Load index.
        
        Args:
            retriever_name (str): The name of the retriever.
            index_path (Optional[str], optional): Path to the index. Defaults to None.
            
        Raises:
            AttributeError: If the retriever does not support the load_index operation.
            Exception: If failed to load the index.
        """
        retriever = self._get_retriever(retriever_name)
        
        if not hasattr(retriever, 'load_index'):
            raise AttributeError(f"Retriever '{retriever_name}' does not support load_index operation")
        
        try:
            load_index_method = getattr(retriever, 'load_index')
            load_index_method(index_path)
            logger.info(f"Loaded index for retriever {retriever_name}")
            
        except Exception as e:
            logger.error(f"Failed to load index for retriever {retriever_name}: {e}")
            raise
    
    def list_retrievers(self) -> List[str]:
        """
        Get a list of all retriever names.
        
        Returns:
            List[str]: List of retriever names.
        """
        return list(self.retrievers.keys())
        
    def get_retriever_info(self, retriever_name: str) -> Dict[str, Any]:
        """
        Get retriever information.
        
        Args:
            retriever_name (str): The name of the retriever.
            
        Returns:
            Dict[str, Any]: Dictionary containing retriever information.
        """
        retriever = self._get_retriever(retriever_name)
        
        info = {
            "name": retriever_name,
            "type": retriever.get_name(),
            "class": retriever.__class__.__name__,
            "config": self.configs.get(retriever_name, {})
        }
        
        if hasattr(retriever, 'get_vectorstore_info'):
            get_vectorstore_info_method = getattr(retriever, 'get_vectorstore_info')
            info.update(get_vectorstore_info_method())
        elif hasattr(retriever, 'get_multipath_info'):
            get_multipath_info_method = getattr(retriever, 'get_multipath_info')
            info.update(get_multipath_info_method())
        
        return info
    
    def remove_retriever(self, retriever_name: str) -> bool:
        """
        Remove a retriever.
        
        Args:
            retriever_name (str): The name of the retriever.
            
        Returns:
            bool: True if removal is successful, False otherwise.
        """
        if retriever_name in self.retrievers:
            retriever = self.retrievers[retriever_name]
            if hasattr(retriever, 'close'):
                try:
                    close_method = getattr(retriever, 'close')
                    close_method()
                except Exception as e:
                    logger.warning(f"Failed to close retriever {retriever_name}: {e}")
            
            del self.retrievers[retriever_name]
            if retriever_name in self.configs:
                del self.configs[retriever_name]
            if retriever_name in self.retriever_types:
                del self.retriever_types[retriever_name]
                
            logger.info(f"Removed retriever: {retriever_name}")
            return True
        return False


# Create global API instance
api = RetrievalAPI()