import logging
from typing import Any, Dict, TYPE_CHECKING

from framework.module import AbstractModule

from core.file_management.index_manager_deletion import _IndexManagerDeletionMixin
from core.file_management.index_manager_pipeline import _IndexManagerPipelineMixin
from core.file_management.index_manager_status import _IndexManagerStatusMixin


if TYPE_CHECKING:
    from config.core.file_management.index_manager_config import IndexManagerConfig

logger = logging.getLogger(__name__)


class IndexManager(_IndexManagerPipelineMixin, _IndexManagerStatusMixin, _IndexManagerDeletionMixin, AbstractModule):
    """
    This class orchestrates the complete indexing pipeline:
    1. Retrieves file content using file_id from FileStorage
    2. Parses the file using StandardParser
    3. Chunks the parsed content using configured chunker
    4. Indexes the chunks using configured indexers
    5. Stores parsed content and chunks back to storage modules
    """

    def __init__(self, config: "IndexManagerConfig"):
        super().__init__(config)

        # Build storage instances
        self.file_storage = config.file_storage_config.build()
        self.parsed_content_storage = config.parsed_content_storage_config.build()
        self.chunk_storage = config.chunk_storage_config.build()

        # Build parser
        self.parser = self.config.parser_config.build()
        logger.info(f"Initialized parser: {type(self.parser).__name__}")

        # Build chunker
        self.chunker = self.config.chunker_config.build()
        logger.info(f"Initialized chunker: {self.chunker.get_chunker_info()['strategy']}")

        # Build indexers
        self.indexers = []
        for indexer_config in self.config.indexer_configs:
            indexer = indexer_config.build()
            self.indexers.append(indexer)
            logger.info(f"Initialized indexer: {type(indexer).__name__}")

        logger.info(f"IndexManager initialized with {len(self.indexers)} indexers")

    async def index_file(self, file_id: str, **kwargs: Any) -> Dict[str, Any]:
        """
        Async method for indexing a file by file_id.
        This is the main entry point for external usage.

        Args:
            file_id: The ID of the file to index
            **kwargs: Additional arguments passed to parser, chunker, and indexers.
                     Reserved: `progress` callable for stage updates.

        Returns:
            Dict containing indexing results
        """
        # Validate file_id
        if file_id is None or not isinstance(file_id, str) or not file_id.strip():
            error_msg = "file_id must be a non-empty string"
            logger.error(error_msg)
            return {
                "success": False,
                "file_id": file_id,
                "error_message": error_msg,
            }

        return await self.process_file(file_id, **kwargs)

