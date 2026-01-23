from config.core.file_management.chunker.chunker_config import TokenChunkerConfig


def test_token_chunker_keeps_markdown_image_atomic():
    chunker = TokenChunkerConfig(chunk_size=5, chunk_overlap=0).build()
    text = "\n".join(
        [
            "Intro " * 20,
            "![Alt text](images/demo.jpg)",
            "Outro " * 20,
        ]
    )
    chunks = chunker.chunk_text(text)
    image_chunks = [c for c in chunks if c.get("metadata", {}).get("segment_type") == "image"]
    assert image_chunks, "expected an image segment chunk"
    assert image_chunks[0]["content"].strip() == "![Alt text](images/demo.jpg)"


def test_token_chunker_keeps_markdown_table_atomic():
    chunker = TokenChunkerConfig(chunk_size=5, chunk_overlap=0).build()
    table = "\n".join(
        [
            "| A | B |",
            "|---|---|",
            "| 1 | 2 |",
            "| 3 | 4 |",
        ]
    )
    text = "\n".join(["Intro " * 20, table, "Outro " * 20])
    chunks = chunker.chunk_text(text)
    table_chunks = [c for c in chunks if c.get("metadata", {}).get("segment_type") == "table"]
    assert table_chunks, "expected a table segment chunk"
    assert "| A | B |" in table_chunks[0]["content"]
    assert "| 3 | 4 |" in table_chunks[0]["content"]


def test_token_chunker_keeps_html_table_atomic():
    chunker = TokenChunkerConfig(chunk_size=5, chunk_overlap=0).build()
    table = "\n".join(
        [
            "<table>",
            "<tr><th>A</th></tr>",
            "<tr><td>1</td></tr>",
            "</table>",
        ]
    )
    text = "\n".join(["Intro " * 20, table, "Outro " * 20])
    chunks = chunker.chunk_text(text)
    table_chunks = [c for c in chunks if c.get("metadata", {}).get("segment_type") == "table"]
    assert table_chunks, "expected an HTML table segment chunk"
    assert "<table>" in table_chunks[0]["content"]
    assert "</table>" in table_chunks[0]["content"]


def test_token_chunker_splits_html_table_and_trailing_image():
    chunker = TokenChunkerConfig(chunk_size=5, chunk_overlap=0).build()
    line = "<table><tr><td>A</td></tr></table> ![cap](images/x.jpg)"
    text = "\n".join(["Intro " * 20, line, "Outro " * 20])
    chunks = chunker.chunk_text(text)
    kinds = [c.get("metadata", {}).get("segment_type") for c in chunks]
    assert "table" in kinds
    assert "image" in kinds


def test_token_chunker_keeps_url_atomic_with_context():
    probe = TokenChunkerConfig(chunk_size=5, chunk_overlap=0).build()
    url = "https://example.com/path?x=1"
    before = "BEFORE"
    after = "AFTER"
    context_tokens = max(
        2,
        len(probe._encode(before)),
        len(probe._encode(after)),
    )
    prefix = ("prefix " * 20) + f"{before} "
    suffix = "suffix " * 20
    url_tokens = len(probe._encode(url))
    prefix_tokens = len(probe._encode(prefix))
    chunk_size = prefix_tokens + max(1, url_tokens // 2)

    chunker = TokenChunkerConfig(
        chunk_size=chunk_size,
        chunk_overlap=5,
        url_atomic_context_tokens=context_tokens,
    ).build()
    text = f"{prefix}{url} {after} {suffix}"
    chunks = chunker.chunk_text(text)
    url_chunks = [c["content"] for c in chunks if "http" in c["content"]]
    assert url_chunks, "expected url chunk"
    for content in url_chunks:
        assert url in content
        assert before in content
        assert after in content
