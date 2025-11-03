from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING
import logging
import os
from pathlib import Path

from core.file_management.parser.base import AbstractParser
from framework.module import AbstractModule

if TYPE_CHECKING:
    from config.core.file_management.parser_combinator_config import ParserCombinatorConfig

logger = logging.getLogger(__name__)


class ParserCombinator(AbstractModule):
    """
    Parser combinator that combines OCR parser and Native parser.

    This combinator provides intelligent parser selection based on file type:
    - OCR Parser (DotsOCR/VLM): For PDF and image files (.pdf, .jpg, .jpeg, .png)
    - Native Parser: For office documents and text files (.docx, .xlsx, .pptx, .html, .txt, etc.)

    Features:
    - Automatic parser selection based on file extension
    - Fallback mechanism if primary parser fails
    - Support for all file types from both parsers
    - Configurable parser instances

    Architecture:
        ParserCombinator
        ├── OCR Parser (for PDF/images)
        └── Native Parser (for office/text documents)
    """

    def __init__(self, config: "ParserCombinatorConfig"):
        """Initialize ParserCombinator with OCR and Native parsers"""
        super().__init__(config)

        # Get base output directory from config
        base_output_dir = getattr(self.config, 'base_output_dir', './data/parsed_files')
        base_output_dir = os.path.abspath(base_output_dir)
        logger.info(f"ParserCombinator base output directory: {base_output_dir}")

        # Create base directory
        os.makedirs(base_output_dir, exist_ok=True)

        # Build OCR parser if configured
        ocr_parser_config = getattr(self.config, 'ocr_parser', None)
        if ocr_parser_config is not None:
            logger.info(f"Building OCR parser: {ocr_parser_config.type}")

            # Set output directory for OCR parser based on type
            if ocr_parser_config.type == "dots_ocr_parser":
                ocr_output_dir = os.path.join(base_output_dir, "dots_ocr")
                os.environ['DOTSOCR_OUTPUT_DIR'] = ocr_output_dir
                logger.info(f"DotsOCR output directory: {ocr_output_dir}")
            elif ocr_parser_config.type == "vlm_ocr_parser":
                ocr_output_dir = os.path.join(base_output_dir, "vlm_ocr")
                os.environ['VLMOCR_OUTPUT_DIR'] = ocr_output_dir
                logger.info(f"VLM OCR output directory: {ocr_output_dir}")

            self.ocr_parser = ocr_parser_config.build()
        else:
            logger.warning("OCR parser not configured, PDF/image files will not be supported")
            self.ocr_parser = None

        # Build Native parser if configured
        native_parser_config = getattr(self.config, 'native_parser', None)
        if native_parser_config is not None:
            logger.info(f"Building Native parser: {native_parser_config.type}")

            # Set output directory for Native parser
            native_output_dir = os.path.join(base_output_dir, "native")
            os.environ['NATIVE_PARSER_OUTPUT_DIR'] = native_output_dir
            logger.info(f"Native parser output directory: {native_output_dir}")

            self.native_parser = native_parser_config.build()
        else:
            logger.warning("Native parser not configured, office/text files will not be supported")
            self.native_parser = None

        # Validate at least one parser is configured
        if self.ocr_parser is None and self.native_parser is None:
            raise ValueError("At least one parser (OCR or Native) must be configured")

        # Build extension to parser mapping
        self._build_extension_mapping()

    def _build_extension_mapping(self):
        """Build mapping from file extensions to parsers"""
        self.extension_to_parser: Dict[str, Tuple[str, AbstractParser]] = {}

        # Map OCR parser extensions
        if self.ocr_parser:
            ocr_extensions = self.ocr_parser.get_supported_extensions()
            for ext in ocr_extensions:
                self.extension_to_parser[ext] = ('ocr', self.ocr_parser)
            logger.info(f"OCR parser supports: {ocr_extensions}")

        # Map Native parser extensions
        if self.native_parser:
            native_extensions = self.native_parser.get_supported_extensions()
            for ext in native_extensions:
                self.extension_to_parser[ext] = ('native', self.native_parser)
            logger.info(f"Native parser supports: {native_extensions}")

        # Log all supported extensions
        all_extensions = list(self.extension_to_parser.keys())
        logger.info(f"ParserCombinator supports {len(all_extensions)} file types: {all_extensions}")

    def get_supported_extensions(self) -> List[str]:
        """Get all supported file extensions from both parsers"""
        return list(self.extension_to_parser.keys())

    async def parse_file(
        self,
        file_data: bytes,
        filename: str,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Parse a file from binary data using appropriate parser.

        Automatically selects the correct parser based on file extension:
        - OCR parser for PDF and images
        - Native parser for office documents and text files

        Args:
            file_data: Binary content of the file
            filename: Name of the file (used for extension detection and output naming)
            **kwargs: Additional parsing options passed to the selected parser

        Returns:
            List of parsing result dictionaries from the selected parser

        Raises:
            ValueError: If file type not supported by any configured parser
            Exception: If parsing fails
        """
        # Get file extension
        file_ext = Path(filename).suffix.lower()

        # Select parser based on extension
        if file_ext not in self.extension_to_parser:
            supported = ', '.join(self.get_supported_extensions())
            raise ValueError(
                f"File extension '{file_ext}' not supported. "
                f"Supported extensions: {supported}"
            )

        parser_type, parser = self.extension_to_parser[file_ext]

        # Parse using selected parser
        try:
            logger.info(f"Parsing {filename} using {parser_type} parser ({parser.__class__.__name__})")
            results = await parser.parse_file(
                file_data=file_data,
                filename=filename,
                **kwargs
            )
            logger.info(f"Successfully parsed {filename} with {parser_type} parser, got {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Failed to parse {filename} with {parser_type} parser: {str(e)}")
            raise

    def get_parser_info(self) -> Dict[str, Any]:
        """
        Get information about configured parsers and their supported file types.

        Returns:
            Dictionary containing parser configuration and supported extensions
        """
        info = {
            "ocr_parser": {
                "configured": self.ocr_parser is not None,
                "type": self.ocr_parser.__class__.__name__ if self.ocr_parser else None,
                "supported_extensions": self.ocr_parser.get_supported_extensions() if self.ocr_parser else []
            },
            "native_parser": {
                "configured": self.native_parser is not None,
                "type": self.native_parser.__class__.__name__ if self.native_parser else None,
                "supported_extensions": self.native_parser.get_supported_extensions() if self.native_parser else []
            },
            "all_supported_extensions": self.get_supported_extensions()
        }
        return info