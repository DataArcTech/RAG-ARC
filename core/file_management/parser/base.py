from abc import abstractmethod
from typing import List, Optional, Dict, Any

from framework.module import AbstractModule


class AbstractParser(AbstractModule):
    """
    Abstract base class for document parsing strategies.

    This class defines the interface for different parsing strategies that can handle
    various document types and parsing requirements.
    """

    @abstractmethod
    async def parse_file(
        self,
        file_data: bytes,
        filename: str,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Parse a file from binary data using this strategy.

        Args:
            file_data: Binary content of the file
            filename: Name of the file (used for extension detection and output naming)
            **kwargs: Strategy-specific parsing options

        Returns:
            List of parsing result dictionaries

        Raises:
            Exception: If parsing fails
        """
        pass