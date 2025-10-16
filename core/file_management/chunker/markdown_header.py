from typing import List, Optional, Dict, Any, TYPE_CHECKING
import logging

from .base import AbstractChunker

if TYPE_CHECKING:
    from config.core.file_management.chunker.chunker_config import MarkdownHeaderChunkerConfig

logger = logging.getLogger(__name__)


class MarkdownHeaderChunker(AbstractChunker):
    """
    MarkdownHeaderChunker is a text chunker that splits markdown text based on header structure and hierarchy.

    This class implements structure-aware markdown splitting by identifying markdown headers and creating chunks based on document organization,
    preserving header hierarchy information for better context understanding and maintaining the logical flow of markdown content.

    Key features:
    - Header-based document structure recognition and preservation
    - Configurable header level splitting (H1, H2, H3, etc.)
    - Code block detection and preservation to avoid splitting code
    - Header content inclusion/exclusion options
    - Hierarchical context maintenance with header metadata
    - Support for both fenced (```) and tilded (~~~) code blocks
    - Optional sub-chunking for oversized sections

    Main parameters (from config):
        headers_to_split_on (List[str]): List of header markers to split on, defaults to ["#", "##"]
        strip_headers (bool): Whether to remove headers from chunk content, defaults to True
        chunk_size (int): Maximum chunk size for sub-splitting, 0 disables sub-splitting, defaults to 0

    Core methods:
        - chunk_text: Main chunking method using markdown header structure
        - get_chunker_info: Get chunker configuration information
        - _split_markdown_text: Execute markdown parsing and splitting logic
        - _chunk_content: Sub-split content if it exceeds chunk_size

    Performance considerations:
        - Preserves document structure and readability over strict size limits
        - Code block detection prevents splitting of code content
        - Header hierarchy tracking provides rich context metadata
        - For documents without clear header structure, consider using other chunking strategies
        - Sub-chunking can help manage very large sections while preserving context

    Typical usage:
        >>> config = MarkdownHeaderChunkerConfig(headers_to_split_on=["#", "##", "###"])
        >>> chunker = MarkdownHeaderChunker(config=config)
        >>> chunks = chunker.chunk_text(markdown_content)
        >>> for chunk in chunks:
        ...     logger.info(f"Header: {chunk['metadata']['header']}")
        ...     logger.info(f"Content: {chunk['content']}")

    Header strategies:
        - Top-level only: ["#"] - Split only on H1 headers
        - Two levels: ["#", "##"] - Split on H1 and H2 headers
        - Detailed: ["#", "##", "###", "####"] - Split on multiple header levels
    """

    def __init__(self, config: "MarkdownHeaderChunkerConfig"):
        """Initialize MarkdownHeaderChunker with config"""
        super().__init__(config)

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """
        Split markdown text into chunks based on header structure.

        This method analyzes markdown content to identify headers at specified levels and creates chunks
        that respect the document's logical structure. Each chunk preserves information about its
        associated header context, enabling better understanding of content hierarchy.

        Args:
            text: Markdown text content to be chunked
            metadata: Optional source document metadata, will be passed to each chunk
            **kwargs: Runtime parameter overrides
                headers_to_split_on (List[str]): Custom header markers, overrides config value
                strip_headers (bool): Whether to remove headers, overrides config value
                chunk_size (int): Maximum chunk size for sub-splitting, overrides config value

        Returns:
            List of chunk dictionaries, each containing:
            - content: Text content of the chunk (with or without headers based on strip_headers)
            - metadata: Chunk-specific metadata
                - chunk_id: Unique identifier for the chunk
                - chunk_index: Index of chunk in sequence
                - start_idx: Starting position in original text (approximate)
                - header: Header information dictionary
                    - level: Header level (number of # characters)
                    - name: Header text content
                - strategy: Chunking strategy identifier ('markdown_header')
            - source_metadata: Original document metadata (if provided)

        Raises:
            Exception: If markdown parsing or chunking process fails
        """
        # Get config parameters
        headers_to_split_on = getattr(self.config, 'headers_to_split_on', ["#", "##"])
        strip_headers = getattr(self.config, 'strip_headers', True)
        chunk_size = getattr(self.config, 'chunk_size', 0)

        # Allow runtime parameter overrides
        headers = kwargs.get('headers_to_split_on', headers_to_split_on)
        strip = kwargs.get('strip_headers', strip_headers)
        size = kwargs.get('chunk_size', chunk_size)

        try:
            chunks = self._split_markdown_text(text, headers, strip, size)

            # Convert to standardized format
            result = []
            for i, chunk in enumerate(chunks):
                chunk_dict = {
                    'content': chunk['content'],
                    'metadata': {
                        'chunk_id': i,
                        'chunk_index': i,
                        'start_idx': text.find(chunk['content']) if chunk['content'] in text else 0,
                        'header': chunk['Header'],
                        'strategy': 'markdown_header'
                    }
                }

                # Add source metadata if provided
                if metadata:
                    chunk_dict['source_metadata'] = metadata.copy()

                result.append(chunk_dict)

            logger.info(f"Split markdown text into {len(result)} chunks using header strategy")
            return result

        except Exception as e:
            logger.error(f"Failed to chunk markdown text: {str(e)}")
            raise

    def get_chunker_info(self) -> Dict[str, Any]:
        """
        Get information about this chunker's configuration and capabilities.

        Returns:
            Dictionary containing detailed chunker information:
            - strategy: Chunking strategy type ('markdown_header')
            - headers_to_split_on: List of header markers used for splitting
            - strip_headers: Whether headers are removed from content
            - chunk_size: Maximum chunk size for sub-splitting (0 = disabled)
            - supported_features: List of supported feature capabilities
            - parameters: Complete copy of current configuration parameters
        """
        headers_to_split_on = getattr(self.config, 'headers_to_split_on', ["#", "##"])
        strip_headers = getattr(self.config, 'strip_headers', True)
        chunk_size = getattr(self.config, 'chunk_size', 0)

        return {
            'strategy': 'markdown_header',
            'headers_to_split_on': headers_to_split_on,
            'strip_headers': strip_headers,
            'chunk_size': chunk_size,
            'supported_features': ['header_preservation', 'hierarchical_structure', 'code_block_handling'],
            'parameters': {
                'headers_to_split_on': headers_to_split_on,
                'strip_headers': strip_headers,
                'chunk_size': chunk_size
            }
        }

    def _chunk_content(self, content: str, chunk_size: int) -> List[str]:
        """
        Split content into chunks of specified size if needed.

        This method is used when a markdown section exceeds the specified chunk_size limit.
        It performs simple character-based splitting while trying to maintain readability.

        Args:
            content: Content to be potentially split
            chunk_size: Maximum size per chunk (0 disables splitting)

        Returns:
            List of content chunks (single item if no splitting needed)
        """
        if chunk_size <= 0:
            return [content]
        return [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]

    def _split_markdown_text(
        self,
        text: str,
        headers_to_split_on: List[str],
        strip_headers: bool,
        chunk_size: int
    ) -> List[Dict]:
        """
        Core markdown splitting logic with header detection and code block handling.

        This method implements the main parsing logic for markdown documents, identifying headers,
        tracking header hierarchy, and properly handling code blocks to avoid inappropriate splitting.

        Args:
            text: Markdown text to parse and split
            headers_to_split_on: List of header markers to recognize
            strip_headers: Whether to exclude headers from chunk content
            chunk_size: Maximum chunk size for optional sub-splitting

        Returns:
            List of dictionaries containing chunk content and header metadata
        """
        lines = text.split("\n")
        results = []

        current_content = []
        current_header = {"level": 0, "name": ""}
        header_stack = []
        in_code_block = False
        opening_fence = ""

        for line in lines:
            stripped_line = line.strip()
            stripped_line = "".join(filter(str.isprintable, stripped_line))

            # Code block detection
            if not in_code_block:
                if stripped_line.startswith("```") and stripped_line.count("```") == 1:
                    in_code_block = True
                    opening_fence = "```"
                elif stripped_line.startswith("~~~"):
                    in_code_block = True
                    opening_fence = "~~~"
            else:
                if stripped_line.startswith(opening_fence):
                    in_code_block = False
                    opening_fence = ""

            if in_code_block:
                current_content.append(line)
                continue

            matched_header = None
            for sep in headers_to_split_on:
                if stripped_line.startswith(sep) and (
                    len(stripped_line) == len(sep) or stripped_line[len(sep)] == " "
                ):
                    matched_header = sep
                    break

            if matched_header:
                # Save current content
                if current_content:
                    section_text = "\n".join(current_content).strip()
                    for chunk in self._chunk_content(section_text, chunk_size):
                        results.append({
                            "content": chunk,
                            "Header": current_header.copy()
                        })
                    current_content = []

                # Update header stack
                current_level = matched_header.count("#")
                while header_stack and header_stack[-1]["level"] >= current_level:
                    header_stack.pop()

                header_name = stripped_line[len(matched_header):].strip()
                current_header = {"level": current_level, "name": header_name}
                header_stack.append(current_header)

                if not strip_headers:
                    current_content.append(stripped_line + "\n")
            else:
                current_content.append(line)

        # Handle remaining content
        if current_content:
            section_text = "\n".join(current_content).strip()
            for chunk in self._chunk_content(section_text, chunk_size):
                results.append({
                    "content": chunk,
                    "Header": current_header.copy()
                })

        return results