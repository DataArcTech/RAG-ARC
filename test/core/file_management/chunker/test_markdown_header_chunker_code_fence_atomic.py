from config.core.file_management.chunker.chunker_config import MarkdownHeaderChunkerConfig


def test_markdown_header_chunker_does_not_split_inside_fenced_code_block():
    code_body = "line\n" * 200
    code_block = "```python\n" + code_body + "```\n"
    text = "# Title\n\n" + ("A" * 200) + "\n\n" + code_block + ("B" * 200) + "\n"

    chunker = MarkdownHeaderChunkerConfig(strip_headers=False).build()
    chunks = chunker.chunk_text(text, chunk_size=120)

    contents = [c["content"] for c in chunks]
    containing_start = [c for c in contents if "```python" in c]
    assert len(containing_start) == 1
    chunk = containing_start[0]
    assert "```python" in chunk
    assert "\nline\n" in ("\n" + chunk + "\n")
    assert chunk.count("```") >= 2
