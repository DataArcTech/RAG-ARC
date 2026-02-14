import os
from typing import Literal

from pydantic import Field

from config.encapsulation.database.file_db.local_config import LocalDBConfig
from encapsulation.io.io_manager import IOManager
from framework.config import AbstractConfig


class IOManagerConfig(AbstractConfig):
    type: Literal["io_manager"] = "io_manager"

    file_db_config: LocalDBConfig
    default_namespace: str = Field(
        default_factory=lambda: str(os.getenv("IO_STORE_DEFAULT_NAMESPACE", "io") or "io").strip() or "io",
        description="Default namespace (prefix) for IO-managed keys.",
    )

    def build(self) -> IOManager:
        return IOManager(self)

