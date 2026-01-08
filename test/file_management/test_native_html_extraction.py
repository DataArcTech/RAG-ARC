from pathlib import Path

from core.file_management.parser.native import extract_html_structured_content, html_structured_content_to_markdown


def test_native_html_extraction_includes_div_excerpt_blocks() -> None:
    html = Path("test/test2.html").read_text(encoding="utf-8")
    content = extract_html_structured_content(html)
    paragraphs = content.get("paragraphs") or []

    joined = "\n".join(str(item) for item in paragraphs)
    assert "The Daily Bugle is New York's most prominent newspaper" in joined
    assert "J. Jonah Jameson" in joined
    assert "freelance photographer" in joined


def test_native_html_extraction_emits_ordered_blocks_for_chunking() -> None:
    html = Path("test/test2.html").read_text(encoding="utf-8")
    content = extract_html_structured_content(html)
    blocks = content.get("blocks") or []
    assert isinstance(blocks, list) and blocks

    heading_idx = next(
        idx
        for idx, block in enumerate(blocks)
        if isinstance(block, dict)
        and block.get("type") == "heading"
        and "Daily Bugle" in str(block.get("text") or "")
    )
    paragraph_idx = next(
        idx
        for idx, block in enumerate(blocks)
        if isinstance(block, dict)
        and block.get("type") == "paragraph"
        and str(block.get("text") or "").startswith("The Daily Bugle is New York's most prominent newspaper")
    )
    assert heading_idx < paragraph_idx

    markdown = html_structured_content_to_markdown(content)
    assert markdown.find("## The Daily Bugle: J. Jonah Jameson's Crusade") < markdown.find(
        "The Daily Bugle is New York's most prominent newspaper"
    )
