import pytest

from encapsulation.data_model.schema import Chunk

from core.graph_adapter.hipporag import HippoRAGGraphAdapter


def _chunk(*, chunk_id: str, file_id: str, semantic_unit_type: str, content: str = "x") -> Chunk:
    return Chunk(
        id=chunk_id,
        content=content,
        metadata={
            "source_file_id": file_id,
            "semantic_unit_type": semantic_unit_type,
            "score": 1.0,
        },
    )


def test_preserve_visual_evidence_inserts_image_chunk():
    file_id = "file-1"
    chunks = [
        _chunk(chunk_id="t1", file_id=file_id, semantic_unit_type="text", content="alpha"),
        _chunk(chunk_id="t2", file_id=file_id, semantic_unit_type="text", content="beta"),
        _chunk(chunk_id="t3", file_id=file_id, semantic_unit_type="text", content="gamma"),
        Chunk(
            id="img1",
            content="![](images/a.jpg)",
            metadata={
                "source_file_id": file_id,
                "semantic_unit_type": "image",
                "image_alts": ["优先通道申请条件表格"],
                "index_text": "优先通道申请条件表格",
                "score": 0.1,
            },
        ),
    ]

    enriched, diag = HippoRAGGraphAdapter._preserve_visual_evidence(
        chunks, requested_k=3, query="优先通道申请条件表格有哪些条件？"
    )
    assert diag.get("visual_evidence_enabled") is True
    assert diag.get("visual_evidence_in_topk") == 1
    assert len(enriched[:3]) == 3
    assert any(HippoRAGGraphAdapter._is_image_chunk(c) for c in enriched[:3])


def test_preserve_visual_evidence_does_not_pull_cross_file_images():
    chunks = [
        _chunk(chunk_id="t1", file_id="file-a", semantic_unit_type="text", content="alpha"),
        _chunk(chunk_id="t2", file_id="file-a", semantic_unit_type="text", content="beta"),
        _chunk(chunk_id="t3", file_id="file-a", semantic_unit_type="text", content="gamma"),
        Chunk(
            id="img-other",
            content="![](images/b.jpg)",
            metadata={
                "source_file_id": "file-b",
                "semantic_unit_type": "image",
                "image_alts": ["其他图片"],
                "index_text": "其他图片",
            },
        ),
    ]
    enriched, diag = HippoRAGGraphAdapter._preserve_visual_evidence(chunks, requested_k=3, query="优先通道申请条件表格")
    assert diag.get("visual_evidence_in_topk") in {0, None}
    assert not any(HippoRAGGraphAdapter._is_image_chunk(c) for c in enriched[:3])


def test_preserve_visual_evidence_noop_when_requested_k_non_positive():
    chunks = [_chunk(chunk_id="t1", file_id="file-a", semantic_unit_type="text", content="alpha")]
    enriched, diag = HippoRAGGraphAdapter._preserve_visual_evidence(chunks, requested_k=0, query="anything")
    assert enriched == chunks
    assert diag.get("visual_evidence_enabled") is True


def test_is_image_chunk_detects_markdown_images():
    chunk = Chunk(content="hello ![](images/x.png)", metadata={"source_file_id": "file-a"})
    assert HippoRAGGraphAdapter._is_image_chunk(chunk) is True
