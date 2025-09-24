from abc import abstractmethod
from typing import List, Optional

from framework.module import AbstractModule


class ParserBase(AbstractModule):
    """Abstract base class for document parsers"""
    
    @abstractmethod
    def parse_file(
        self,
        file_data: bytes,
        filename: str,
        **kwargs
    ) -> List[dict]:
        """
        Parse a file from binary data

        Args:
            file_data: Binary content of the file
            filename: Name of the file (used for extension detection and output naming)
            **kwargs: Additional parsing options

        Returns:
            List of parsing result dictionaries
        """
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """
        Get list of supported file extensions
        
        Returns:
            List of supported file extensions (e.g., ['.pdf', '.jpg', '.png'])
        """
        pass