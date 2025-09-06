import asyncio
import logging
from typing import Any, Callable, Dict, Iterable, List, Optional
import uuid
import warnings
from pydantic import BaseModel, ConfigDict, Field, model_validator, field_validator
import numpy as np
import jieba
from rank_bm25 import BM25Okapi
from core.utils.data_model import Document
from core.retrieval.base import BaseRetriever

logger = logging.getLogger(__name__)

def default_preprocessing_func(text: str) -> List[str]:
    """Default text preprocessing function, only effective for English text
    
    Args:
        text: Input text
        
    Returns:
        List of words after tokenization
    """
    return text.split()


def jieba_preprocessing_func(text: str) -> List[str]:
    """Jieba-based text preprocessing function, suitable for Chinese text

    Args:
        text: Input text
        
    Returns:
        List of words after tokenization
    """
    return list(jieba.cut(text)) 


class BM25Retriever(BaseRetriever):
    """
    BM25Retriever is a document retriever based on the BM25 algorithm, suitable for efficient text relevance ranking in scenarios such as information retrieval, question answering systems, and knowledge bases.

    This class implements BM25 retrieval for document collections by integrating the rank_bm25 library, supporting dynamic document addition, deletion, and batch index building operations.
    It is suitable for scenarios where the document collection is relatively static and high retrieval speed is required. For scenarios with frequent document additions and deletions, it is recommended to use vector retrieval (such as VectorStoreRetriever).

    Key features:
    - Supports rapid construction of BM25 retrievers from text lists or Document object lists.
    - Supports custom tokenization/preprocessing functions to adapt to different languages and tokenization needs.
    - Supports dynamic addition and deletion of documents (each operation rebuilds the index, suitable for small to medium-sized datasets).
    - Can obtain retrieval scores, top-k documents and scores, retriever configuration information, etc.
    - Compatible with asynchronous document addition/deletion, facilitating large-scale data processing.
    - Validates parameters through Pydantic to ensure configuration safety.

    Main parameters:
        vectorizer (Any): BM25 vectorizer instance (typically BM25Okapi).
        docs (List[Document]): List of document objects currently held by the retriever.
        k (int): Default number of relevant documents to return.
        preprocess_func (Callable): Text tokenization/preprocessing function, defaults to space tokenization.
        bm25_params (Dict): Parameters passed to BM25Okapi (such as k1, b, etc.).

    Core methods:
        - from_texts/from_documents: Build retriever from raw text or Document.
        - _get_relevant_documents: Retrieve the top k documents most relevant to the query.
        - get_scores: Get BM25 scores for the query against all documents.
        - get_top_k_with_scores: Get top-k documents and their scores.
        - add_documents/delete_documents: Dynamically add/delete documents and rebuild the index.
        - get_bm25_info: Get retriever configuration information and statistics.
        - update_k: Dynamically adjust the number of documents returned.

    Performance considerations:
        - Defaults to space tokenization, tokenization functions can be customized
        - Each document addition/deletion rebuilds the BM25 index, suitable for scenarios with smaller document volumes or infrequent updates.
        - For large document volumes or frequent updates, it is recommended to use a vector retrieval solution.
        - Supports asynchronous operations, facilitating large-scale data processing.

    Typical usage:
        >>> retriever = BM25Retriever.from_texts(["text1", "text2"], k=3)
        >>> results = retriever._get_relevant_documents("query statement")
        >>> retriever.add_documents([Document(content="new document")])
        >>> retriever.delete_documents(ids=["doc_id"])
        >>> info = retriever.get_bm25_info()

    Attributes:
        vectorizer: BM25 vectorizer instance
        docs: Document list
        k: Number of documents to return
        preprocess_func: Text tokenization function
        bm25_params: BM25 algorithm parameters
    """
    
    def __init__(self, vectorizer=None, docs=None, k=5, preprocess_func=default_preprocessing_func, 
                 bm25_params=None, search_kwargs=None, **kwargs):
        """Initialize BM25 Retriever
        
        Args:
            vectorizer: BM25 vectorizer
            docs: Document list  
            k: Number of documents to return, must be greater than 0
            preprocess_func: Preprocessing function
            bm25_params: BM25 parameters
            search_kwargs: Search parameters
            **kwargs: Other parameters
        """
        # Initialize BaseRetriever
        super().__init__(search_kwargs=search_kwargs or {}, **kwargs)
        
        # Initialize BM25 specific attributes
        self.vectorizer = vectorizer
        self.docs = docs or []
        self.k = k
        self.preprocess_func = preprocess_func
        self.bm25_params = bm25_params or {}
        self.processed_texts = []
        
        # Validate k value
        if self.k <= 0:
            raise ValueError(f"k must be greater than 0, current value: {self.k}")
            
        # Validate preprocess_func
        if not callable(self.preprocess_func):
            raise ValueError("preprocess_func must be a callable function")
        
        # Issue preprocessing function warning (only when using default function)
        if self._contains_chinese() and self.preprocess_func == default_preprocessing_func:
            warnings.warn(
                "Chinese text detected, but the default English tokenizer is being used. "
                "It is recommended to install jieba and set preprocess_func=jieba_preprocessing_func for better Chinese retrieval performance.",
                UserWarning,
                stacklevel=2
            )
    
    def _contains_chinese(self) -> bool:
        """Check if the documents contain Chinese text"""
        for doc in self.docs:
            if any('\u4e00' <= char <= '\u9fff' for char in doc.content):
                warnings.warn(
                    "Chinese text detected, please use jieba tokenizer.",
                    UserWarning,
                    stacklevel=2
                )
                return True
        return False
    
    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Optional[Iterable[Dict[str, Any]]] = None,
        ids: Optional[Iterable[str]] = None,
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> "BM25Retriever":
        """Create BM25Retriever from text list
        
        Args:
            texts: Text list
            metadatas: Metadata list, optional
            ids: ID list, optional
            bm25_params: BM25 algorithm parameters, optional
            preprocess_func: Preprocessing function
            **kwargs: Other parameters
            
        Returns:
            BM25Retriever instance
            
        Raises:
            ImportError: If rank_bm25 is not installed
            ValueError: If parameters do not match
        """
        # Convert to list
        texts_list = list(texts)
        if not texts_list:
            raise ValueError("texts cannot be empty")
        
        # Process metadata and IDs
        metadatas_list = list(metadatas) if metadatas is not None else [{} for _ in texts_list]
        ids_list = list(ids) if ids is not None else [str(uuid.uuid4()) for _ in texts_list]
        
        if len(metadatas_list) != len(texts_list):
            raise ValueError("Metadata length does not match text length")
        if len(ids_list) != len(texts_list):
            raise ValueError("ID length does not match text length")
        
        # Preprocess text
        logger.info(f"Preprocessing {len(texts_list)} texts...")
        texts_processed = [preprocess_func(text) for text in texts_list]
        
        # Create BM25 vectorizer
        bm25_params = bm25_params or {}
        logger.info(f"Creating BM25 vectorizer with parameters: {bm25_params}")
        vectorizer = BM25Okapi(texts_processed, **bm25_params)
        
        # Create document objects
        docs = [
            Document(content=text, metadata=metadata, id=doc_id)
            for text, metadata, doc_id in zip(texts_list, metadatas_list, ids_list)
        ]
        
        logger.info(f"Successfully created BM25Retriever with {len(docs)} documents")

        instance = cls(
            vectorizer=vectorizer,
            docs=docs,
            preprocess_func=preprocess_func,
            bm25_params=bm25_params,
            **kwargs
        )
        instance.processed_texts = texts_processed

        if instance._contains_chinese() and preprocess_func == default_preprocessing_func:
            warnings.warn(
                "Chinese documents added, but the default English tokenizer is being used. "
                "It is recommended to install jieba and set preprocess_func=jieba_preprocessing_func for better Chinese retrieval performance.",
                UserWarning,
                stacklevel=2
            )
            
        return instance
    
    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        bm25_params: Optional[Dict[str, Any]] = None,
        preprocess_func: Callable[[str], List[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> "BM25Retriever":
        """Create BM25Retriever from document list
        
        Args:
            documents: Document list
            bm25_params: BM25 algorithm parameters, optional
            preprocess_func: Preprocessing function
            **kwargs: Other parameters
            
        Returns:
            BM25Retriever instance
        """
        docs_list = list(documents)
        if not docs_list:
            raise ValueError("documents cannot be empty")
        
        texts = [doc.content for doc in docs_list]
        metadatas = [doc.metadata for doc in docs_list]
        ids = [doc.id if doc.id else str(uuid.uuid4()) for doc in docs_list]
        
        return cls.from_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            bm25_params=bm25_params,
            preprocess_func=preprocess_func,
            **kwargs,
        )
    
    def _get_relevant_documents(self, query: str, **kwargs: Any) -> List[Document]:
        """Get the top k documents related to the query

        Args:
            query: Query string
            **kwargs: Other parameters, may contain 'k' to override the default return count

        Returns:
            List of relevant documents

        Raises:
            ValueError: If vectorizer is not initialized
        """
        if self.vectorizer is None:
            raise ValueError("BM25 vectorizer not initialized")

        if not self.docs:
            logger.warning("Document list is empty, returning empty result")
            return []

        # Get number of documents to return
        k = min(kwargs.get('k', self.k), len(self.docs))
        if k <= 0:
            return []

        try:
            # Preprocess query
            processed_query = self.preprocess_func(query)
            logger.debug(f"Preprocessed query: {processed_query}")

            # Get scores for all documents
            scores = self.vectorizer.get_scores(processed_query)
            # Get indices of top k documents with highest scores

            if k == 1:
                top_index = np.argmax(scores)
                return [self.docs[top_index]]
            
            scores_array = np.array(scores)
            top_indices = np.argpartition(scores_array, -k)[-k:]
            top_indices = top_indices[np.argsort(-scores_array[top_indices])]
            
            # Return top k documents
            top_docs = [self.docs[idx] for idx in top_indices]
            logger.debug(f"Found {len(top_docs)} relevant documents")
            return top_docs

        except Exception as e:
            logger.error(f"Error during BM25 search: {e}")
            raise

    def get_scores(self, query: str) -> List[float]:
        """Get BM25 scores for the query against all documents
        
        Args:
            query: Query string
            
        Returns:
            List of BM25 scores for all documents
        """
        if self.vectorizer is None:
            raise ValueError("BM25 vectorizer not initialized")
        
        processed_query = self.preprocess_func(query)
        scores = self.vectorizer.get_scores(processed_query)
        return scores.tolist()
    
    def get_top_k_with_scores(self, query: str, k: Optional[int] = None) -> List[tuple[Document, float]]:
        """Get top-k documents and their scores
        
        Args:
            query: Query string
            k: Number of documents to return, if None uses the instance's k value
            
        Returns:
            List of (document, score) tuples
        """
        if self.vectorizer is None or not self.docs:
            return []
        
        k = min(k or self.k, len(self.docs))
        if k <= 0:
            return []
        
        # Get all scores
        scores = self.get_scores(query)
        scores_array = np.array(scores)

        if k == 1:
            top_index = np.argmax(scores_array)
            return [(self.docs[top_index], scores_array[top_index])]

        top_indices = np.argpartition(scores_array, -k)[-k:]
        top_indices = top_indices[np.argsort(-scores_array[top_indices])]
        
        # Return documents and scores
        results = []
        for idx in top_indices:
            results.append((self.docs[idx], scores_array[idx]))
        
        return results
    
    def add_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Add new documents to the retriever
        
        Warning: This operation rebuilds the entire BM25 index, which may be slow on large document collections.
        For frequent document update operations, consider using VectorStoreRetriever.
        
        Args:
            documents: List of documents to add
            **kwargs: Other parameters
                rebuild_threshold: Document count threshold, warnings will be issued if exceeded (default 1000)
            
        Returns:
            List of added document IDs
            
        Raises:
            ImportError: If rank_bm25 is not installed
            RuntimeWarning: If document count exceeds recommended threshold
        """
        if not documents:
            return []
        
        # Check document count, issue performance warning
        rebuild_threshold = kwargs.get('rebuild_threshold', 1000)
        total_docs = len(self.docs) + len(documents)

        if total_docs > rebuild_threshold:
            warnings.warn(
                f"Rebuilding BM25 index with {total_docs} documents, this may be slow. "
                f"For large or frequently updated document collections, it is recommended to use VectorStoreRetriever.",
                RuntimeWarning,
                stacklevel=2
            )
        
        # Add documents to existing list
        self.docs.extend(documents)
        
        # Preprocess new documents and add to cache
        new_texts = [doc.content for doc in documents]
        new_processed = [self.preprocess_func(text) for text in new_texts]
        self.processed_texts.extend(new_processed)
        
        self.vectorizer = BM25Okapi(self.processed_texts, **self.bm25_params)

        # Chinese processing optimization: Check if Chinese tokenizer is needed after adding
        if self._contains_chinese() and self.preprocess_func == default_preprocessing_func:
            warnings.warn(
                "Chinese documents added, but the default English tokenizer is being used. "
                "It is recommended to install jieba and set preprocess_func=jieba_preprocessing_func for better Chinese retrieval performance.",
                UserWarning,
                stacklevel=2
            )
        
        logger.info(f"Added {len(documents)} documents, rebuilt BM25 index")
        
        # Return IDs of added documents (if any)
        return [doc.id for doc in documents if doc.id is not None]
    
    async def aadd_documents(self, documents: List[Document], **kwargs: Any) -> List[str]:
        """Asynchronously add documents
        
        Warning: This operation rebuilds the entire BM25 index, which may be slow on large document collections.
        """
        try:
            return await asyncio.to_thread(self.add_documents, documents, **kwargs)
        except AttributeError:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.add_documents, documents, **kwargs)
    
    def delete_documents(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """Delete documents
        
        Warning: This operation rebuilds the entire BM25 index, which may be slow on large document collections.
        For frequent document update operations, consider using VectorStoreRetriever.
        
        Args:
            ids: List of document IDs to delete, if None deletes all documents
            **kwargs: Other parameters
                rebuild_threshold: Document count threshold, warnings will be issued if exceeded (default 1000)
            
        Returns:
            Whether deletion was successful
        """
        if ids is None:
            # Delete all documents
            self.docs.clear()
            self.processed_texts.clear()
            self.vectorizer = None
            logger.info("Deleted all documents")
            return True
        
        # Delete documents with specified IDs
        ids_set = set(ids)
        original_count = len(self.docs)

        # Delete documents and cached processed texts simultaneously
        new_docs = []
        new_processed = []
        for i, doc in enumerate(self.docs):
            if doc.id not in ids_set:
                new_docs.append(doc)
                new_processed.append(self.processed_texts[i])

        deleted_count = original_count - len(new_docs)
        if deleted_count == 0:
            return False
        
        self.docs = new_docs
        self.processed_texts = new_processed
        
        # Rebuild index
        if self.docs:
            self.vectorizer = BM25Okapi(self.processed_texts, **self.bm25_params)
        else:
            self.vectorizer = None
        
        return True
    
    async def adelete_documents(self, ids: Optional[List[str]] = None, **kwargs: Any) -> bool:
        """Asynchronously delete documents
        
        Args:
            ids: List of document IDs to delete
            **kwargs: Other parameters
            
        Returns:
            Whether deletion was successful
        """
        try:
            return await asyncio.to_thread(self.delete_documents, ids, **kwargs)
        except AttributeError:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, self.delete_documents, ids, **kwargs)
    
    def get_document_count(self) -> int:
        """Get document count"""
        return len(self.docs)
    
    def get_bm25_info(self) -> Dict[str, Any]:
        """Get BM25 retriever information"""
        return {
            "document_count": len(self.docs),
            "k": self.k,
            "bm25_params": self.bm25_params,
            "preprocess_func": self.preprocess_func.__name__ if hasattr(self.preprocess_func, '__name__') else str(self.preprocess_func)
        }
    
    def update_k(self, k: int) -> None:
        """Update number of documents to return
        
        Args:
            k: New number of documents to return
        """
        if k <= 0:
            raise ValueError("k must be greater than 0")
        self.k = k