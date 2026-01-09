from pathlib import Path

from config.core.file_management.chunker.chunker_config import MarkdownHeaderChunkerConfig
from core.file_management.chunker.markdown_header import MarkdownHeaderChunker
from core.file_management.parser.native import extract_html_structured_content, html_structured_content_to_markdown


def test_html_markdown_order_allows_header_chunking() -> None:
    html = Path("test/test2.html").read_text(encoding="utf-8")
    content = extract_html_structured_content(html)
    md = html_structured_content_to_markdown(content)

    chunker = MarkdownHeaderChunker(MarkdownHeaderChunkerConfig())
    chunks = chunker.chunk_text(md)

    assert any(
        "The Daily Bugle: J. Jonah Jameson's Crusade" in str(chunk.get("content") or "")
        and "freelance photographer" in str(chunk.get("content") or "")
        for chunk in chunks
        if isinstance(chunk, dict)
    )

