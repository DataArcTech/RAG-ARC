from config.core.file_management.chunker.chunker_config import RecursiveChunkerConfig


def test_recursive_chunker_keeps_fenced_code_block_atomic():
    code_body = "x = 1\n" * 50
    code_block = "```python\n" + code_body + "```\n"
    text = "Intro text.\n" + code_block + "Outro text.\n"

    chunker = RecursiveChunkerConfig(chunk_size=50, chunk_overlap=0).build()
    chunks = chunker.chunk_text(text)

    contents = [c["content"] for c in chunks]
    containing_start = [c for c in contents if "```python" in c]
    assert len(containing_start) == 1
    assert code_block.strip() in containing_start[0]
    assert containing_start[0].count("```") >= 2
