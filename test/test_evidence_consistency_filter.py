from encapsulation.data_model.schema import Chunk

from core.utils.anchor_consistency import prune_anchors_by_query_text
from core.utils.evidence_consistency import filter_chunks_by_anchors


def _mk_chunk(*, file_id: str, filename: str, content: str = "x") -> Chunk:
    return Chunk(
        id=f"chunk-{file_id}-{filename}",
        content=content,
        metadata={"source_file_id": file_id, "filename": filename},
    )


def test_filter_chunks_by_anchors_filename_match() -> None:
    chunks = [
        _mk_chunk(file_id="A", filename="安达人寿 产品A 条款.md", content="..."),
        _mk_chunk(file_id="A", filename="安达人寿 产品A 条款.md", content="..."),
        _mk_chunk(file_id="B", filename="苏黎世保险 产品B 条款.md", content="..."),
    ]
    kept, info = filter_chunks_by_anchors(chunks=chunks, anchors=["安达人寿"], min_keep=1)
    assert info.enabled is True
    assert info.passed is True
    assert {c.metadata.get("source_file_id") for c in kept} == {"A"}


def test_filter_chunks_by_anchors_filename_fallback_terms(monkeypatch) -> None:
    # Make fallback deterministic in this unit test.
    monkeypatch.setenv("RAG_ANCHOR_FILENAME_FALLBACK_ENABLED", "1")
    monkeypatch.setenv("RAG_ANCHOR_FILENAME_FALLBACK_MIN_LEN", "2")
    monkeypatch.setenv("RAG_ANCHOR_FILENAME_FALLBACK_MAX_LEN", "4")
    monkeypatch.setenv("RAG_ANCHOR_FILENAME_FALLBACK_MAX_FILE_FRACTION", "0.3")
    monkeypatch.setenv("RAG_ANCHOR_FILENAME_FALLBACK_MAX_TERMS", "3")

    # No filename contains the full anchor, but one file contains a short distinctive CJK span.
    relevant = [
        _mk_chunk(file_id="F1", filename="AXA_人生盛利_产品说明.md", content="..."),
        _mk_chunk(file_id="F1", filename="AXA_人生盛利_产品说明.md", content="..."),
    ]
    distractors = [
        _mk_chunk(file_id=f"D{i}", filename=f"其他产品_{i}.md", content="...") for i in range(1, 9)
    ]
    chunks = relevant + distractors

    kept, info = filter_chunks_by_anchors(chunks=chunks, anchors=["財富Upgrade人生盛利"], min_keep=1)
    assert info.enabled is True
    assert info.passed is True
    assert {c.metadata.get("source_file_id") for c in kept} == {"F1"}


def test_prune_anchors_by_query_text() -> None:
    pruned = prune_anchors_by_query_text(
        anchors=["安达人寿", "苏黎世保险"],
        rewritten_query="回到安达人寿的产品：按年缴费是否有优惠？",
    )
    assert pruned == ["安达人寿"]
