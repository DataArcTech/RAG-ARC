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


def test_semantic_unit_chunker_standard_emits_code_anchor_and_slice():
    chunker = SemanticUnitChunkerConfig(
        level="standard",
        code_small_max_tokens=1,  # force slicing path deterministically
        code_slice_max_tokens=10_000,
        code_slice_overlap_lines=0,
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
    code_slices = [
        c
        for c in chunks
        if c.get("metadata", {}).get("chunk_role") == "slice"
        and c.get("metadata", {}).get("semantic_unit_type") == "code"
    ]

    assert code_anchors, "expected at least one code anchor"
    assert code_slices, "expected at least one code slice"
    assert code_slices[0]["metadata"]["anchor_chunk_id"] is None
    assert code_anchors[0]["metadata"]["code_language"] == "python"


def test_semantic_unit_chunker_parses_fenced_code_with_extra_info_string():
    chunker = SemanticUnitChunkerConfig(
        level="standard",
        code_small_max_tokens=1,  # force slicing path deterministically
        code_slice_max_tokens=10_000,
        code_slice_overlap_lines=0,
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
    code_slices = [
        c
        for c in chunks
        if c.get("metadata", {}).get("chunk_role") == "slice"
        and c.get("metadata", {}).get("semantic_unit_type") == "code"
    ]

    assert code_anchors, "expected at least one code anchor"
    assert code_slices, "expected at least one code slice"
    assert code_anchors[0]["metadata"]["code_language"] == "python"


def test_semantic_unit_chunker_standard_emits_list_anchor_and_slice():
    chunker = SemanticUnitChunkerConfig(
        level="standard",
        list_small_max_tokens=1,  # force slicing path deterministically
        list_slice_max_tokens=10_000,
        list_slice_overlap_items=0,
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
