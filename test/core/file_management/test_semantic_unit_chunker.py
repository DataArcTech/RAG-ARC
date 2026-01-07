from config.core.file_management.chunker.chunker_config import (
    SemanticUnitChunkerConfig,
    TokenChunkerConfig,
)
from core.file_management.index_manager import IndexManager


def test_semantic_unit_chunker_emits_table_anchor_and_slice():
    chunker = SemanticUnitChunkerConfig(
        level="basic",
        table_small_max_tokens=1,  # force "large table" path deterministically
        table_slice_max_tokens=10_000,
        table_slice_overlap_rows=0,
        fallback_chunker_config=TokenChunkerConfig(chunk_size=200, chunk_overlap=0),
    ).build()

    markdown = "\n".join(
        [
            "Intro paragraph.",
            "",
            "Table: Example",
            "| A | B |",
            "|---|---|",
            "| 1 | 2 |",
            "| 3 | 4 |",
            "",
            "Outro paragraph.",
        ]
    )

    chunks = chunker.chunk_text(markdown, metadata={"source_file_id": "file-1"})

    table_anchors = [
        c
        for c in chunks
        if c.get("metadata", {}).get("chunk_role") == "anchor"
        and c.get("metadata", {}).get("semantic_unit_type") == "table"
    ]
    table_slices = [
        c
        for c in chunks
        if c.get("metadata", {}).get("chunk_role") == "slice"
        and c.get("metadata", {}).get("semantic_unit_type") == "table"
    ]

    assert table_anchors, "expected at least one table anchor"
    assert table_slices, "expected at least one table slice"

    anchor = table_anchors[0]
    slice_chunk = table_slices[0]
    assert anchor["metadata"]["semantic_unit_id"] == slice_chunk["metadata"]["semantic_unit_id"]
    assert slice_chunk["metadata"]["anchor_chunk_id"] is None
    assert "| A | B |" in (anchor.get("content") or "")
    assert "| 1 | 2 |" not in (anchor.get("content") or "")
    assert "| 1 | 2 |" in (slice_chunk.get("content") or "")


def test_index_manager_backfills_anchor_chunk_id():
    chunks = [
        {
            "content": "anchor",
            "metadata": {
                "chunk_role": "anchor",
                "semantic_unit_id": "file-1:table:1",
            },
        },
        {
            "content": "slice",
            "metadata": {
                "chunk_role": "slice",
                "semantic_unit_id": "file-1:table:1",
                "anchor_chunk_id": None,
            },
        },
    ]
    chunk_ids = ["ANCHOR_ID", "SLICE_ID"]

    IndexManager._backfill_anchor_chunk_ids(chunks, chunk_ids)

    assert chunks[1]["metadata"]["anchor_chunk_id"] == "ANCHOR_ID"


def test_index_manager_persists_backfilled_anchor_chunk_id_to_storage():
    chunks = [
        {
            "content": "anchor",
            "metadata": {
                "chunk_role": "anchor",
                "semantic_unit_id": "file-1:table:1",
            },
        },
        {
            "content": "slice",
            "metadata": {
                "chunk_role": "slice",
                "semantic_unit_id": "file-1:table:1",
                "anchor_chunk_id": None,
            },
        },
    ]
    chunk_ids = ["ANCHOR_ID", "SLICE_ID"]

    IndexManager._backfill_anchor_chunk_ids(chunks, chunk_ids)

    class FakeChunkStorage:
        def __init__(self):
            self.writes = {}

        def overwrite_chunk_json(self, chunk_id, chunk_dict):  # noqa: ANN001
            self.writes[chunk_id] = chunk_dict
            return True

    fake = FakeChunkStorage()
    updated = IndexManager._persist_backfilled_anchor_chunk_ids(fake, chunks, chunk_ids)

    assert updated == 1
    assert fake.writes["SLICE_ID"]["metadata"]["anchor_chunk_id"] == "ANCHOR_ID"


def test_semantic_unit_chunker_standard_emits_code_chunk_as_single_anchor():
    chunker = SemanticUnitChunkerConfig(
        level="standard",
        code_small_max_tokens=1,
        code_anchor_preview_lines=1,
        fallback_chunker_config=TokenChunkerConfig(chunk_size=200, chunk_overlap=0),
    ).build()

    markdown = "\n".join(
        [
            "Intro paragraph.",
            "",
            "Code: Example",
            "```python",
            "def add(a, b):",
            "    return a + b",
            "```",
            "",
            "Outro paragraph.",
        ]
    )

    chunks = chunker.chunk_text(markdown, metadata={"source_file_id": "file-1"})

    code_anchors = [
        c
        for c in chunks
        if c.get("metadata", {}).get("chunk_role") == "anchor"
        and c.get("metadata", {}).get("semantic_unit_type") == "code"
    ]

    assert code_anchors, "expected at least one code anchor"
    assert code_anchors[0]["metadata"]["code_language"] == "python"
    anchor_content = code_anchors[0].get("content") or ""
    assert "def add(a, b)" in anchor_content
    assert "return a + b" in anchor_content


def test_semantic_unit_chunker_parses_fenced_code_with_extra_info_string():
    chunker = SemanticUnitChunkerConfig(
        level="standard",
        code_small_max_tokens=1,
        fallback_chunker_config=TokenChunkerConfig(chunk_size=200, chunk_overlap=0),
    ).build()

    markdown = "\n".join(
        [
            "Intro paragraph.",
            "",
            "```python linenums",
            "def add(a, b):",
            "    return a + b",
            "```",
            "",
            "Outro paragraph.",
        ]
    )

    chunks = chunker.chunk_text(markdown, metadata={"source_file_id": "file-1"})

    code_anchors = [
        c
        for c in chunks
        if c.get("metadata", {}).get("chunk_role") == "anchor"
        and c.get("metadata", {}).get("semantic_unit_type") == "code"
    ]

    assert code_anchors, "expected at least one code anchor"
    assert code_anchors[0]["metadata"]["code_language"] == "python"


def test_semantic_unit_chunker_standard_emits_list_anchor_and_slice():
    chunker = SemanticUnitChunkerConfig(
        level="standard",
        list_small_max_tokens=1,  # force slicing path deterministically
        list_slice_max_tokens=10_000,
        list_slice_overlap_items=0,
        list_anchor_preview_items=1,
        fallback_chunker_config=TokenChunkerConfig(chunk_size=200, chunk_overlap=0),
    ).build()

    markdown = "\n".join(
        [
            "Intro paragraph.",
            "",
            "List: Example",
            "- item 1",
            "- item 2",
            "- item 3",
            "",
            "Outro paragraph.",
        ]
    )

    chunks = chunker.chunk_text(markdown, metadata={"source_file_id": "file-1"})

    list_anchors = [
        c
        for c in chunks
        if c.get("metadata", {}).get("chunk_role") == "anchor"
        and c.get("metadata", {}).get("semantic_unit_type") == "list"
    ]
    list_slices = [
        c
        for c in chunks
        if c.get("metadata", {}).get("chunk_role") == "slice"
        and c.get("metadata", {}).get("semantic_unit_type") == "list"
    ]

    assert list_anchors, "expected at least one list anchor"
    assert list_slices, "expected at least one list slice"
    assert list_slices[0]["metadata"]["anchor_chunk_id"] is None
    anchor_content = list_anchors[0].get("content") or ""
    slice_content = list_slices[0].get("content") or ""
    assert "- item 1" in anchor_content
    assert "- item 3" not in anchor_content
    assert "- item 3" in slice_content


def test_semantic_unit_chunker_standard_emits_math_anchor():
    chunker = SemanticUnitChunkerConfig(
        level="standard",
        fallback_chunker_config=TokenChunkerConfig(chunk_size=200, chunk_overlap=0),
    ).build()

    markdown = "\n".join(
        [
            "Before math.",
            "",
            "$$",
            "E = mc^2",
            "$$",
            "",
            "After math.",
        ]
    )

    chunks = chunker.chunk_text(markdown, metadata={"source_file_id": "file-1"})
    math_anchors = [
        c
        for c in chunks
        if c.get("metadata", {}).get("chunk_role") == "anchor"
        and c.get("metadata", {}).get("semantic_unit_type") == "math"
    ]
    assert math_anchors, "expected a math anchor"


def test_semantic_unit_chunker_emits_image_as_atomic_anchor():
    chunker = SemanticUnitChunkerConfig(
        level="basic",
        fallback_chunker_config=TokenChunkerConfig(chunk_size=50, chunk_overlap=0),
    ).build()

    markdown = "\n".join(
        [
            "Intro paragraph.",
            "",
            "![Figure: Premium refund table](images/abc.jpg)",
            "",
            "Outro paragraph.",
        ]
    )

    chunks = chunker.chunk_text(markdown, metadata={"source_file_id": "file-1"})
    image_chunks = [
        c
        for c in chunks
        if c.get("metadata", {}).get("chunk_role") == "anchor"
        and c.get("metadata", {}).get("semantic_unit_type") == "image"
    ]
    assert image_chunks, "expected at least one image chunk"
    assert "![Figure: Premium refund table](images/abc.jpg)" in (image_chunks[0].get("content") or "")
    assert image_chunks[0]["metadata"]["is_atomic"] is True


def test_semantic_unit_chunker_treats_html_table_as_atomic():
    chunker = SemanticUnitChunkerConfig(
        level="basic",
        fallback_chunker_config=TokenChunkerConfig(chunk_size=50, chunk_overlap=0),
    ).build()

    markdown = "\n".join(
        [
            "Intro",
            "<table>",
            "<tr><th>A</th><th>B</th></tr>",
            "<tr><td>1</td><td>2</td></tr>",
            "</table>",
            "Outro",
        ]
    )

    chunks = chunker.chunk_text(markdown, metadata={"source_file_id": "file-1"})
    table_chunks = [
        c
        for c in chunks
        if c.get("metadata", {}).get("chunk_role") == "anchor"
        and c.get("metadata", {}).get("semantic_unit_type") == "table"
    ]
    assert table_chunks, "expected an HTML table chunk"
    assert table_chunks[0]["metadata"].get("is_atomic") is True
    content = (table_chunks[0].get("content") or "")
    # Chunker may normalize HTML tables to pipe tables for better downstream processing.
    assert ("<table>" in content) or ("| A | B |" in content)


def test_semantic_unit_chunker_splits_html_table_and_trailing_image():
    chunker = SemanticUnitChunkerConfig(
        level="basic",
        fallback_chunker_config=TokenChunkerConfig(chunk_size=50, chunk_overlap=0),
    ).build()

    markdown = "\n".join(
        [
            "Intro",
            "<table><tr><td>A</td></tr></table> ![cap](images/x.jpg)",
            "Outro",
        ]
    )

    chunks = chunker.chunk_text(markdown, metadata={"source_file_id": "file-1"})
    kinds = {(c.get("metadata") or {}).get("semantic_unit_type") for c in chunks}
    assert "table" in kinds
    assert "image" in kinds


def test_semantic_unit_chunker_does_not_use_html_table_line_as_image_caption():
    chunker = SemanticUnitChunkerConfig(
        level="basic",
        fallback_chunker_config=TokenChunkerConfig(chunk_size=50, chunk_overlap=0),
    ).build()

    markdown = "\n".join(
        [
            "<table>",
            "<tr><td>A</td></tr>",
            "</table>",
            "![cap](images/x.jpg)",
        ]
    )

    chunks = chunker.chunk_text(markdown, metadata={"source_file_id": "file-1"})
    image_chunks = [
        c
        for c in chunks
        if c.get("metadata", {}).get("chunk_role") == "anchor"
        and c.get("metadata", {}).get("semantic_unit_type") == "image"
    ]
    assert image_chunks, "expected an image chunk"
    assert (image_chunks[0].get("content") or "").strip() == "![cap](images/x.jpg)"
