from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING
import logging
from pathlib import Path

from core.file_management.parser.base import AbstractParser
from framework.module import AbstractModule
from core.utils.path_guard import require_writable_dir

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
    - Support for all file types from configured parsers
    - Configurable parser instances

    Architecture:
        ParserCombinator
        ├── OCR Parser (for PDF/images)
        └── Native Parser (for office/text documents)
    """

    def __init__(self, config: "ParserCombinatorConfig"):
        """Initialize ParserCombinator with OCR and Native parsers"""
        super().__init__(config)

        base_output_dir = getattr(self.config, "base_output_dir", None)
        if not isinstance(base_output_dir, str) or not base_output_dir.strip():
            raise ValueError("ParserCombinator requires config.base_output_dir")
        base_output_dir_virtual = str(base_output_dir).strip()
        if not base_output_dir_virtual.startswith("io://"):
            raise ValueError("ParserCombinator config.base_output_dir must be an io:// virtual path")
        # Ensure underlying directory exists and is writable (mapped by io_manager -> LocalDB in Phase 1).
        require_writable_dir(base_output_dir_virtual)
        self.base_output_dir = base_output_dir_virtual
        logger.info("ParserCombinator base output directory: %s", self.base_output_dir)

        # Build OCR parser if configured
        ocr_parser_config = getattr(self.config, 'ocr_parser', None)
        if ocr_parser_config is not None:
            logger.info(f"Building OCR parser: {ocr_parser_config.type}")
            ocr_subdir = self._resolve_ocr_output_subdir(ocr_parser_config.type)
            resolved_output_dir = f"{self.base_output_dir.rstrip('/')}/{ocr_subdir}"
            require_writable_dir(resolved_output_dir)
            ocr_cfg_output = getattr(ocr_parser_config, "output_dir", None)
            if ocr_cfg_output is None:
                ocr_parser_config = ocr_parser_config.model_copy(update={"output_dir": resolved_output_dir})
            else:
                resolved_output_dir = str(ocr_cfg_output).strip()
                if not resolved_output_dir.startswith("io://"):
                    raise ValueError("OCR parser config.output_dir must be an io:// virtual path")
                require_writable_dir(resolved_output_dir)
                ocr_parser_config = ocr_parser_config.model_copy(update={"output_dir": resolved_output_dir})
            self.ocr_parser = ocr_parser_config.build()
        else:
            logger.warning("OCR parser not configured, PDF/image files will not be supported")
            self.ocr_parser = None

        # Build Native parser if configured
        native_parser_config = getattr(self.config, 'native_parser', None)
        if native_parser_config is not None:
            logger.info(f"Building Native parser: {native_parser_config.type}")
            native_subdir = getattr(self.config, "native_output_subdir", None)
            if not isinstance(native_subdir, str) or not native_subdir.strip():
                raise ValueError("ParserCombinator config.native_output_subdir is required when native_parser is set")
            resolved_output_dir = f"{self.base_output_dir.rstrip('/')}/{native_subdir}"
            require_writable_dir(resolved_output_dir)
            native_cfg_output = getattr(native_parser_config, "output_dir", None)
            if native_cfg_output is None:
                native_parser_config = native_parser_config.model_copy(update={"output_dir": resolved_output_dir})
            else:
                resolved_output_dir = str(native_cfg_output).strip()
                if not resolved_output_dir.startswith("io://"):
                    raise ValueError("Native parser config.output_dir must be an io:// virtual path")
                require_writable_dir(resolved_output_dir)
                native_parser_config = native_parser_config.model_copy(update={"output_dir": resolved_output_dir})
            self.native_parser = native_parser_config.build()
        else:
            logger.warning("Native parser not configured, office/text files will not be supported")
            self.native_parser = None

        # Validate at least one parser is configured
        if self.ocr_parser is None and self.native_parser is None:
            raise ValueError("At least one parser (OCR or Native) must be configured")

        # Build extension to parser mapping
        self._build_extension_mapping()

    def _resolve_ocr_output_subdir(self, parser_type: str) -> str:
        token = str(parser_type or "").strip()
        if token == "dots_ocr_parser":
            subdir = getattr(self.config, "dots_ocr_output_subdir", None)
        elif token == "vlm_ocr_parser":
            subdir = getattr(self.config, "vlm_ocr_output_subdir", None)
        elif token == "mineru_parser":
            subdir = getattr(self.config, "mineru_output_subdir", None)
        else:
            raise ValueError(f"Unknown OCR parser type '{token}'")
        if not isinstance(subdir, str) or not subdir.strip():
            raise ValueError(f"ParserCombinator config missing output subdir for OCR parser type '{token}'")
        return subdir

    @staticmethod
    def _build_extension_mapping_from_parsers(
        *,
        ocr_parser: Optional[AbstractParser],
        native_parser: Optional[AbstractParser],
    ) -> Dict[str, Tuple[str, AbstractParser]]:
        """
        Build mapping from file extensions to parsers.

        Precedence: OCR overrides native for overlapping extensions (e.g., .pdf).
        """
        mapping: Dict[str, Tuple[str, AbstractParser]] = {}

        if native_parser:
            native_extensions = native_parser.get_supported_extensions()
            for ext in native_extensions:
                mapping[ext] = ("native", native_parser)

        if ocr_parser:
            ocr_extensions = ocr_parser.get_supported_extensions()
            for ext in ocr_extensions:
                mapping[ext] = ("ocr", ocr_parser)

        return mapping

    def _build_extension_mapping(self):
        """Build mapping from file extensions to parsers"""
        self.extension_to_parser = self._build_extension_mapping_from_parsers(
            ocr_parser=self.ocr_parser,
            native_parser=self.native_parser,
        )

        if self.native_parser:
            logger.info(f"Native parser supports: {self.native_parser.get_supported_extensions()}")
        if self.ocr_parser:
            logger.info(f"OCR parser supports: {self.ocr_parser.get_supported_extensions()}")

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
            self._annotate_parse_results(
                results,
                parser_family=parser_type,
                parser_instance=parser,
            )
            logger.info(f"Successfully parsed {filename} with {parser_type} parser, got {len(results)} results")
            return results

        except Exception as exc:
            logger.error(f"Failed to parse {filename} with {parser_type} parser: {str(exc)}")

            fallback = self._resolve_mineru_fallback(
                parser_family=parser_type,
                parser_instance=parser,
                file_ext=file_ext,
            )
            if fallback is None:
                raise

            fallback_family, fallback_parser = fallback
            logger.warning(
                "Falling back to %s parser for %s after MinerU failure: %s",
                fallback_family,
                filename,
                str(exc),
                exc_info=True,
            )
            try:
                results = await fallback_parser.parse_file(file_data=file_data, filename=filename, **kwargs)
                self._annotate_parse_results(
                    results,
                    parser_family=fallback_family,
                    parser_instance=fallback_parser,
                )
                self._annotate_fallback_results(
                    results,
                    from_parser=parser,
                    to_parser=fallback_parser,
                    reason=str(exc) or exc.__class__.__name__,
                )
                logger.info(
                    "Fallback parse succeeded for %s (from=%s to=%s) results=%s",
                    filename,
                    self._parser_label(parser),
                    self._parser_label(fallback_parser),
                    len(results) if isinstance(results, list) else None,
                )
                return results
            except Exception as fallback_exc:
                logger.error(
                    "Fallback parse failed for %s after MinerU failure: primary=%s fallback=%s",
                    filename,
                    str(exc),
                    str(fallback_exc),
                    exc_info=True,
                )
                raise RuntimeError(
                    f"Failed to parse {filename}: mineru_error={exc!s}; fallback_error={fallback_exc!s}"
                ) from fallback_exc

    @staticmethod
    def _parser_label(parser_instance: AbstractParser) -> str:
        """Stable parser label used for storage keys / observability."""
        name = parser_instance.__class__.__name__
        # Keep these labels stable for downstream storage keys and debug tooling.
        mapping = {
            "DotsOCRParser": "dots_ocr",
            "VLMOcrParser": "vlm_ocr",
            "MinerUParser": "mineru",
            "NativeParser": "native",
        }
        return mapping.get(name, name.lower())

    def _resolve_mineru_fallback(
        self,
        *,
        parser_family: str,
        parser_instance: AbstractParser,
        file_ext: str,
    ) -> Tuple[str, AbstractParser] | None:
        """Return fallback parser when MinerU fails and fallback is enabled."""

        if str(parser_family or "").strip().lower() != "ocr":
            return None
        enabled = bool(getattr(self.config, "mineru_fallback_to_native_on_failure", False))
        if not enabled:
            return None
        if self._parser_label(parser_instance) != "mineru":
            return None
        native = getattr(self, "native_parser", None)
        if native is None:
            return None
        try:
            supported = set(native.get_supported_extensions() or [])
        except Exception:
            supported = set()
        if file_ext not in supported:
            return None
        return ("native", native)

    @staticmethod
    def _annotate_fallback_results(
        results: Any,
        *,
        from_parser: AbstractParser,
        to_parser: AbstractParser,
        reason: str,
    ) -> None:
        if not isinstance(results, list):
            return
        fallback_meta = {
            "enabled": True,
            "from": from_parser.__class__.__name__,
            "from_label": ParserCombinator._parser_label(from_parser),
            "to": to_parser.__class__.__name__,
            "to_label": ParserCombinator._parser_label(to_parser),
            "reason": str(reason or "").strip() or "unknown",
        }
        for item in results:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
                item["metadata"] = meta
            meta.setdefault("parser_fallback", fallback_meta)

    @classmethod
    def _annotate_parse_results(
        cls,
        results: Any,
        *,
        parser_family: str,
        parser_instance: AbstractParser,
    ) -> None:
        """Inject parser provenance into parse result metadata for downstream storage/debug."""
        if not isinstance(results, list):
            return
        label = cls._parser_label(parser_instance)
        parser_name = parser_instance.__class__.__name__
        for item in results:
            if not isinstance(item, dict):
                continue
            meta = item.get("metadata")
            if not isinstance(meta, dict):
                meta = {}
                item["metadata"] = meta
            meta.setdefault("parser_family", str(parser_family))
            meta.setdefault("parser_name", str(parser_name))
            meta.setdefault("parser_label", str(label))

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
