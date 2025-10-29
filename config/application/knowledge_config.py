from application.knowledge.module import Knowledge
from framework.config import AbstractConfig
from config.core.file_management.index_manager_config import IndexManagerConfig
from config.core.file_management.storage.file_storage import FileStorageConfig

from typing import Literal


class KnowledgeConfig(AbstractConfig):
    type: Literal["knowledge"] = "knowledge"
    index_manager_config: IndexManagerConfig
    file_storage_config: FileStorageConfig
    max_concurrent_indexing: int = 5  # Maximum number of concurrent indexing operations

    def build(self):
        return Knowledge(self)