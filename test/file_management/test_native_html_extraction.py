from pathlib import Path

from core.file_management.parser.native import extract_html_structured_content


def test_native_html_extraction_includes_div_excerpt_blocks() -> None:
    html = Path("test/test2.html").read_text(encoding="utf-8")
    content = extract_html_structured_content(html)
    paragraphs = content.get("paragraphs") or []

    joined = "\n".join(str(item) for item in paragraphs)
    assert "The Daily Bugle is New York's most prominent newspaper" in joined
    assert "J. Jonah Jameson" in joined
    assert "freelance photographer" in joined

