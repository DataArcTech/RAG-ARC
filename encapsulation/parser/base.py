from abc import abstractmethod
from typing import List, Optional

from framework.module import AbstractModule


class ParserBase(AbstractModule):
    """Abstract base class for document parsers"""
    
    @abstractmethod
    def parse_file(
        self, 
        input_path: str,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> List[dict]:
        """
        Parse a single file (PDF, image, etc.)
        
        Args:
            input_path: Path to input file
            output_dir: Optional override for output directory
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