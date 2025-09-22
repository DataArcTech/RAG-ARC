import logging
import json
from typing import List, Dict, Any, Optional, Literal, Annotated, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pydantic import Field, ConfigDict

from framework.module import AbstractModule
from framework.config import AbstractConfig

from encapsulation.database.relational_db.data_schema import ChunksStatus
from encapsulation.data_model.data_model import Document

# Import configuration classes
from config.encapsulaiton.faiss_config import FaissIndexConfig
from config.database.bm25_config import BM25IndexBuilderConfig
from config.llm.huggingface_config import HuggingFaceEmbedConfig

logger = logging.getLogger(__name__)


# Database configuration classes
class PostgreSQLDBConfig(AbstractConfig):
    """PostgreSQL database configuration"""
    type: Literal["postgresql"] = "postgresql"
    host: str = "localhost"
    port: int = 5432
    database: str
    user: str
    password: str
    pool_size: int = 10
    max_overflow: int = 20
    echo_sql: bool = False

    def build(self):
        from encapsulation.database.relational_db.postgresql import PostgreSQLDB
        return PostgreSQLDB(self)


class LocalDBConfig(AbstractConfig):
    """Local file database configuration"""
    type: Literal["local"] = "local"
    base_path: str
    cleanup_empty_dirs: bool = True

    def build(self):
        from encapsulation.database.file_db.local import LocalDB
        return LocalDB(self)


class IndexManagerConfig(AbstractConfig):
    """IndexManager configuration class

    Unified index manager configuration for managing multiple index types construction and maintenance.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["index_manager"] = "index_manager"

    # Database configurations
    relational_db_config: Annotated[PostgreSQLDBConfig, Field(description="Relational database config for storing chunks metadata")]
    file_db_config: Annotated[LocalDBConfig, Field(description="File database config for storing chunk data")]

    # Indexer configuration mapping
    indexer_configs: Dict[str, Union[FaissIndexConfig, BM25IndexBuilderConfig]] = Field(
        description="Indexer configuration mapping, key is index_type, value is corresponding indexer config"
    )

    # Embedding model configuration (for FAISS index)
    embedding_config: Optional[HuggingFaceEmbedConfig] = Field(
        default=None,
        description="Embedding model configuration for FAISS index vector computation"
    )

    # Batch processing configuration
    batch_size: int = Field(
        default=100,
        description="Batch size, number of chunks processed at once"
    )
    max_concurrent_builds: int = Field(
        default=3,
        description="Maximum concurrent builds, number of index types built simultaneously"
    )

    # Retry configuration
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts"
    )
    retry_delay: float = Field(
        default=1.0,
        description="Retry delay in seconds"
    )
    
    def build(self) -> "IndexManager":
        return IndexManager(self)



class IndexManager(AbstractModule):
    """
    Unified Index Manager

    Responsible for reading unindexed chunk data from database, calling appropriate indexers
    for construction, and updating metadata status. Supports unified management of multiple index types.

    Core functionalities:
    - Index building: Build indexes from chunks in CHUNKED status
    - Index rebuilding: Force rebuild indexes of specified types
    - Status management: Update chunks index status
    - Statistics: Get statistics for each index type
    """

    def __init__(self, config: IndexManagerConfig):
        super().__init__(config)
        self._relational_db = None
        self._file_db = None
        self._indexers: Dict[str, Any] = {}
        self._embedding_model = None
        self._initialize_dependencies()

    def _initialize_dependencies(self):
        """Initialize dependency components"""
        # Initialize database connections
        self._initialize_databases()
        # Initialize embedding model (if configured)
        self._initialize_embedding_model()
        # Initialize indexer instances
        self._initialize_indexers()

    def _initialize_databases(self):
        """Initialize database connections"""
        try:
            # Directly use config objects to build database instances
            self._relational_db = self.config.relational_db_config.build()
            self._file_db = self.config.file_db_config.build()

            logger.info("Database connections initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize databases: {e}")
            raise


    def _initialize_embedding_model(self):
        """Initialize embedding model (for FAISS index)"""
        if self.config.embedding_config is None:
            logger.info("No embedding model configured, FAISS indexing will be skipped")
            return

        try:
            # Directly use config object to build embedding model
            self._embedding_model = self.config.embedding_config.build()
            logger.info(f"Initialized embedding model: {self.config.embedding_config.model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            # Don't raise exception, allow continuing but skip FAISS indexing

    def _initialize_indexers(self):
        """Initialize indexers of each type based on configuration"""
        for index_type, indexer_config in self.config.indexer_configs.items():
            try:
                # Directly use config object to build indexer
                self._indexers[index_type] = indexer_config.build()
                logger.info(f"Initialized {index_type} indexer: {indexer_config.index_path}")
            except Exception as e:
                logger.error(f"Failed to initialize indexer {index_type}: {e}")
                # Don't raise exception, allow partial indexer initialization failure

    def build_pending_indexes(self) -> Dict[str, int]:
        """
        Build all pending indexes

        Returns:
            Dict[str, int]: Build count statistics for each index type
        """
        # Get chunks metadata with CHUNKED status
        unindexed_chunks = self._relational_db.list_chunks_metadata(
            status=ChunksStatus.CHUNKED
        )

        if not unindexed_chunks:
            logger.info("No pending chunks to index")
            return {}

        logger.info(f"Found {len(unindexed_chunks)} chunks pending indexing")

        # Group by index_type
        chunks_by_type = self._group_chunks_by_index_type(unindexed_chunks)
        build_stats = {}

        # Concurrently build different types of indexes
        with ThreadPoolExecutor(max_workers=self.config.max_concurrent_builds) as executor:
            futures = {}
            for index_type, chunks in chunks_by_type.items():
                if index_type in self._indexers:
                    future = executor.submit(self._build_index_for_type, index_type, chunks)
                    futures[future] = index_type
                else:
                    logger.warning(f"No indexer available for type: {index_type}")
                    build_stats[index_type] = 0

            # Collect results
            for future in as_completed(futures):
                index_type = futures[future]
                try:
                    count = future.result()
                    build_stats[index_type] = count
                    logger.info(f"Built {count} indexes for type: {index_type}")
                except Exception as e:
                    logger.error(f"Failed to build indexes for type {index_type}: {e}")
                    build_stats[index_type] = 0

        return build_stats

    def _group_chunks_by_index_type(self, chunks_metadata: List[Any]) -> Dict[str, List[Any]]:
        """Group chunks by index type"""
        groups = {}
        for chunk_meta in chunks_metadata:
            # Determine index_type based on configuration or chunk metadata
            index_type = self._determine_index_type(chunk_meta)
            if index_type not in groups:
                groups[index_type] = []
            groups[index_type].append(chunk_meta)
        return groups

    def _determine_index_type(self, chunk_meta: Any) -> str:
        """Determine the index type that chunk should use"""
        # Can decide which index to use based on chunk characteristics, configuration, etc.
        # Simplified to default strategy here
        if hasattr(chunk_meta, 'index_type') and chunk_meta.index_type:
            return chunk_meta.index_type

        # Default strategy: use the first available index type
        available_types = list(self._indexers.keys())
        if available_types:
            return available_types[0]

        # If no indexer available, return default value
        return "faiss"

    def _build_index_for_type(self, index_type: str, chunks_metadata: List[Any]) -> int:
        """Build index for specific type"""
        indexer = self._indexers[index_type]
        built_count = 0

        # Batch processing build
        for i in range(0, len(chunks_metadata), self.config.batch_size):
            batch = chunks_metadata[i:i + self.config.batch_size]
            retry_count = 0

            while retry_count <= self.config.max_retries:
                try:
                    # Load chunk data
                    documents = self._load_documents_from_chunks(batch)
                    if not documents:
                        logger.warning(f"No documents loaded from batch {i//self.config.batch_size + 1}")
                        break

                    # Build index
                    if hasattr(indexer, 'add'):
                        # For indexes that need embedding vectors (like FAISS), need to compute embedding vectors
                        if index_type == "faiss":
                            if self._embedding_model is None:
                                logger.warning(f"FAISS indexer requires embedding model, but none configured. Skipping.")
                                break

                            # Compute embedding vectors
                            texts = [doc.content for doc in documents]
                            embeddings = self._embedding_model.embed_documents(texts)
                            indexer.add(documents, embeddings)
                        else:
                            indexer.add(documents)
                    elif hasattr(indexer, 'from_documents'):
                        indexer.from_documents(documents)
                    else:
                        logger.error(f"Indexer {index_type} does not support add or from_documents")
                        break

                    # Update metadata status
                    self._update_chunks_status(batch, ChunksStatus.INDEXED, index_type)
                    built_count += len(batch)
                    logger.info(f"Built index for {len(batch)} chunks of type {index_type}")
                    break  # Success, break out of retry loop

                except Exception as e:
                    retry_count += 1
                    if retry_count <= self.config.max_retries:
                        logger.warning(f"Failed to build index batch for type {index_type} (attempt {retry_count}): {e}")
                        import time
                        time.sleep(self.config.retry_delay)
                    else:
                        logger.error(f"Failed to build index batch for type {index_type} after {self.config.max_retries} retries: {e}")
                        # Mark failed status
                        self._update_chunks_status(batch, ChunksStatus.FAILED, index_type)

        return built_count

    def _load_documents_from_chunks(self, chunks_metadata: List[Any]) -> List[Document]:
        """Load Document objects from chunks metadata"""
        documents = []
        for chunk_meta in chunks_metadata:
            try:
                # Load chunk data from file database
                chunk_data = self._file_db.retrieve(chunk_meta.blob_key)

                # Parse JSON data to Document objects
                chunks_json = json.loads(chunk_data.decode('utf-8'))
                for chunk in chunks_json.get('chunks', []):
                    doc = Document(
                        id=chunk.get('id'),
                        content=chunk.get('content'),
                        metadata={
                            **chunk.get('metadata', {}),
                            'chunks_id': chunk_meta.chunks_id,
                            'source_parsed_content_id': chunk_meta.source_parsed_content_id
                        }
                    )
                    documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to load documents from chunk {chunk_meta.chunks_id}: {e}")

        return documents

    def _update_chunks_status(self, chunks_metadata: List[Any], status: ChunksStatus, index_type: str):
        """Update chunks status"""
        for chunk_meta in chunks_metadata:
            try:
                # Update status and index type
                updates = {
                    "status": status,
                    "index_type": index_type
                }

                # Save to database
                success = self._relational_db.update_chunks_metadata(chunk_meta.chunks_id, updates)
                if not success:
                    logger.warning(f"Failed to update chunk status {chunk_meta.chunks_id}")
            except Exception as e:
                logger.error(f"Failed to update chunk status {chunk_meta.chunks_id}: {e}")

    def rebuild_index(self, index_type: str, force: bool = False) -> int:
        """Rebuild index of specified type

        Args:
            index_type: Index type to rebuild
            force: Whether to force rebuild (reset all index status of this type)

        Returns:
            int: Number of rebuilt indexes
        """
        if index_type not in self._indexers:
            logger.error(f"Indexer not available for type: {index_type}")
            return 0

        if force:
            # Force rebuild: reset all index status of this type
            chunks = self._relational_db.list_chunks_metadata()
            reset_count = 0
            for chunk in chunks:
                if hasattr(chunk, 'index_type') and chunk.index_type == index_type:
                    updates = {
                        "status": ChunksStatus.CHUNKED,
                        "index_type": None
                    }
                    if self._relational_db.update_chunks_metadata(chunk.chunks_id, updates):
                        reset_count += 1

            logger.info(f"Reset {reset_count} chunks for index type {index_type}")

        # Rebuild
        unindexed_chunks = self._relational_db.list_chunks_metadata(status=ChunksStatus.CHUNKED)
        relevant_chunks = [chunk for chunk in unindexed_chunks
                          if self._determine_index_type(chunk) == index_type]

        if relevant_chunks:
            return self._build_index_for_type(index_type, relevant_chunks)
        else:
            logger.info(f"No chunks found for rebuilding index type {index_type}")
            return 0

    def get_index_statistics(self) -> Dict[str, Dict[str, int]]:
        """Get index statistics

        Returns:
            Dict[str, Dict[str, int]]: Statistics for each index type
        """
        stats = {}

        for index_type in self._indexers.keys():
            try:
                # Get all chunks of this type
                all_chunks = self._relational_db.list_chunks_metadata()
                type_chunks = [chunk for chunk in all_chunks
                              if hasattr(chunk, 'index_type') and chunk.index_type == index_type]

                stats[index_type] = {
                    'total': len(type_chunks),
                    'indexed': len([c for c in type_chunks if c.status == ChunksStatus.INDEXED]),
                    'pending': len([c for c in type_chunks if c.status == ChunksStatus.CHUNKED]),
                    'failed': len([c for c in type_chunks if c.status == ChunksStatus.FAILED])
                }
            except Exception as e:
                logger.error(f"Failed to get statistics for index type {index_type}: {e}")
                stats[index_type] = {
                    'total': 0,
                    'indexed': 0,
                    'pending': 0,
                    'failed': 0
                }

        return stats

    def get_indexer_health(self) -> Dict[str, Dict[str, Any]]:
        """Get indexer health status

        Returns:
            Dict[str, Dict[str, Any]]: Health status of each indexer
        """
        health_status = {}

        for index_type, indexer in self._indexers.items():
            try:
                if hasattr(indexer, 'health_check'):
                    health_status[index_type] = indexer.health_check()
                else:
                    # Basic health check
                    health_status[index_type] = {
                        "status": "unknown",
                        "exists": hasattr(indexer, 'index_exists') and indexer.index_exists(),
                        "stats": indexer.get_vector_db_info() if hasattr(indexer, 'get_vector_db_info') else {}
                    }
            except Exception as e:
                logger.error(f"Health check failed for indexer {index_type}: {e}")
                health_status[index_type] = {
                    "status": "error",
                    "error": str(e),
                    "exists": False,
                    "stats": {}
                }

        return health_status