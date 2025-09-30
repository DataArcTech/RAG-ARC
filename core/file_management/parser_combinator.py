from typing import List, Optional, Dict, Any, TYPE_CHECKING
import logging
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

if TYPE_CHECKING:
    from config.core.file_management.parser_combinator_config import ParserCombinatorConfig

logger = logging.getLogger(__name__)


class ParserCombinator():
    """
    Standard document parser that directly delegates to encapsulation parsers.

    This parser provides a unified interface to the encapsulation layer parsers
    (DotsOCR, Native) with automatic parser selection based on file type.

    Features:
    - Automatic parser selection based on file extension
    - Direct delegation to encapsulation parsers
    - Minimal processing overhead
    - Support for all file types supported by encapsulation parsers
    """

    def __init__(self, config: "ParserCombinatorConfig"):
        """Initialize ParserCobinator with parser subconfig"""
        super().__init__(config)
        # Build parser immediately from subconfig
        parser_config = getattr(self.config, 'parser', None)
        if parser_config is not None:
            logger.debug(f"Parser specified, build parser by config")
            self.parser = parser_config.build()
        else:
            # No parser config, will need to auto-select per file extension
            logger.debug(f"Parser not specified, auto select available parser")
            self.parser = None

    def parse_file(
        self,
        file_data: bytes,
        filename: str,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Parse a file from binary data using appropriate encapsulation parser.

        Args:
            file_data: Binary content of the file
            filename: Name of the file (used for extension detection and output naming)
            **kwargs: Additional parsing options passed to encapsulation parser

        Returns:
            List of parsing result dictionaries from encapsulation parser

        Raises:
            ValueError: If file type not supported or parser not available
            Exception: If parsing fails
        """
        # Get file extension
        file_ext = Path(filename).suffix.lower()

        # Use configured parser or auto-select by extension
        if self.parser is not None:
            parser = self.parser
        else:
            parser = self._select_parser_by_extension(file_ext)

        # Parse using configured parser
        try:
            logger.info(f"Parsing {filename} using {parser.__class__.__name__}")
            results = parser.parse_file(
                file_data=file_data,
                filename=filename,
                **kwargs
            )
            logger.info(f"Successfully parsed {filename}, got {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Failed to parse {filename}: {str(e)}")
            raise

    def _select_parser_by_extension(self, file_ext: str):
        """Auto-select parser based on file extension"""
        # Try to build available parsers and check their supported extensions
        available_parsers = []

        # Try DotsOCR parser
        try:
            from config.encapsulation.parser.dots_ocr import DotsOCRConfig
            # Build dots_ocr parser using default config
            dots_ocr_config = DotsOCRConfig()
            dots_ocr_parser = dots_ocr_config.build()
            supported_extensions = dots_ocr_parser.get_supported_extensions()
            if file_ext in supported_extensions:
                available_parsers.append(("dots_ocr", dots_ocr_parser))
        except Exception as e:
            logger.debug(f"DotsOCR parser not available: {e}")

        # Try Native parser
        try:
            from config.encapsulation.parser.native import NativeParserConfig
            # Build native parser using default config
            native_config = NativeParserConfig()
            native_parser = native_config.build()
            supported_extensions = native_parser.get_supported_extensions()
            if file_ext in supported_extensions:
                available_parsers.append(("native", native_parser))
        except Exception as e:
            logger.debug(f"Native parser not available: {e}")

        if not available_parsers:
            raise ValueError(
                f"No parser available for file type '{file_ext}'. "
                f"Please ensure DotsOCR or Native parser are properly configured."
            )

        # Use first compatible parser
        parser_name, parser_instance = available_parsers[0]
        logger.info(f"Auto-selected parser '{parser_name}' for file type '{file_ext}'")
        return parser_instance