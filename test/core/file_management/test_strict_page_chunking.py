from core.file_management.pageindex.strict_page_chunking import (
    build_page_markdown_from_content_list,
    strict_page_chunk,
)


class _DummyChunker:
    def chunk_text(self, *, text: str, metadata: dict, **_kwargs):  # noqa: ANN001
        return [
            {
                "content": text,
                "metadata": {
                    **metadata,
                    "semantic_unit_id": "unit-1",
                    "parent_unit_id": "unit-1",
                },
            }
        ]


def test_build_page_markdown_from_content_list_keeps_page_boundaries() -> None:
    content_list = [
        {"type": "text", "text": "Title", "page_idx": 0},
        {"type": "text", "text": "Body line", "page_idx": 0},
        {"type": "image", "image_caption": ["Figure 1"], "page_idx": 1},
        {"type": "table", "table_caption": ["Table 1"], "table_body": "<table><tr><td>A</td></tr></table>", "page_idx": 2},
    ]

    pages = build_page_markdown_from_content_list(content_list)

    assert sorted(pages.keys()) == [0, 1, 2]
    assert "Title" in pages[0]
    assert "Body line" in pages[0]
    assert "Figure 1" in pages[1]
    assert "Table 1" in pages[2]


def test_strict_page_chunk_forces_single_page_and_unique_semantic_ids() -> None:
    chunker = _DummyChunker()
    chunks = strict_page_chunk(
        chunker=chunker,
        chunk_metadata={"source_file_id": "f1"},
        page_texts={3: "Page three", 4: "Page four"},
    )

    assert len(chunks) == 2

    first = chunks[0]["metadata"]
    second = chunks[1]["metadata"]

    assert first["page_start"] == 3
    assert first["page_end"] == 3
    assert second["page_start"] == 4
    assert second["page_end"] == 4

    assert first["chunk_index"] == 0
    assert second["chunk_index"] == 1

    assert first["semantic_unit_id"].endswith(":p3")
    assert second["semantic_unit_id"].endswith(":p4")
    assert first["parent_unit_id"].endswith(":p3")
    assert second["parent_unit_id"].endswith(":p4")


def test_enrich_chunks_preserves_strict_page_metadata(tmp_path) -> None:
    import json

    from core.file_management.pageindex.service import PageIndexService

    md_text = "# Title\nHello world"
    md_path = tmp_path / "doc.md"
    md_path.write_text(md_text, encoding="utf-8")

    content_list = [
        {"type": "text", "text": "Title", "page_idx": 0},
        {"type": "text", "text": "Hello world", "page_idx": 0},
    ]
    (tmp_path / "doc_content_list.json").write_text(json.dumps(content_list), encoding="utf-8")

    service = PageIndexService(llm=None)
    ctx = service.build_context(
        file_id="f1",
        filename="doc.pdf",
        markdown=md_text,
        md_path=str(md_path),
        output_dir=str(tmp_path),
    )

    chunks = [
        {
            "content": "Hello world",
            "metadata": {"page_start": 4, "page_end": 4, "page_strict": True},
        }
    ]

    service.enrich_chunks(context=ctx, chunks=chunks)
    meta = chunks[0]["metadata"]
    assert meta["page_start"] == 4
    assert meta["page_end"] == 4
    assert meta.get("section_id")
