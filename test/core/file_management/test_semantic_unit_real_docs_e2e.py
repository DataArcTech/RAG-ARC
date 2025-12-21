from pathlib import Path
import uuid

from config.core.file_management.chunker.chunker_config import SemanticUnitChunkerConfig, TokenChunkerConfig
from config.core.retrieval.multipath_config import MultiPathRetrieverConfig
from config.core.retrieval.tantivy_bm25_config import TantivyBM25RetrieverConfig
from config.encapsulation.database.bm25_config import BM25BuilderConfig
from core.file_management.index_manager import IndexManager
from encapsulation.data_model.schema import Chunk


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_semantic_unit_chunker_recognizes_task_list_in_real_doc():
    doc_path = _repo_root() / "docs-proj" / "semantic_unit_eval_complex.md"
    text = doc_path.read_text(encoding="utf-8", errors="replace")

    chunker = SemanticUnitChunkerConfig(
        level="standard",
        list_small_max_tokens=1,
        list_slice_max_tokens=10_000,
        list_slice_overlap_items=0,
        fallback_chunker_config=TokenChunkerConfig(chunk_size=800, chunk_overlap=0),
    ).build()

    chunks = chunker.chunk_text(text=text, metadata={"source_file_id": str(doc_path)}, level="standard")

    list_anchors = [
        c
        for c in chunks
        if c.get("metadata", {}).get("chunk_role") == "anchor"
        and c.get("metadata", {}).get("semantic_unit_type") == "list"
    ]
    assert list_anchors, "expected at least one list anchor in real doc"

    task_lists = [c for c in list_anchors if c.get("metadata", {}).get("list_type") == "task"]
    assert task_lists, "expected at least one task-list anchor in real doc"
    assert any("- [ ]" in (c.get("content") or "") or "- [x]" in (c.get("content") or "") for c in task_lists)


def test_semantic_unit_e2e_real_doc_bm25_anchor_backfill(tmp_path: Path):
    doc_path = _repo_root() / "docs-proj" / "semantic_unit_eval_complex.md"
    text = doc_path.read_text(encoding="utf-8", errors="replace")
    owner_id = str(uuid.uuid4())

    chunker_config = SemanticUnitChunkerConfig(
        level="advanced",
        fallback_chunker_config=TokenChunkerConfig(chunk_size=800, chunk_overlap=80),
    )
    chunker_config.table_small_max_tokens = 1
    chunker_config.code_small_max_tokens = 1
    chunker_config.list_small_max_tokens = 1
    chunker_config.list_anchor_preview_items = 1
    chunker = chunker_config.build()

    raw_chunks = chunker.chunk_text(
        text=text,
        metadata={"source_file_id": str(doc_path), "filename": doc_path.name, "owner_id": owner_id},
        level="advanced",
    )
    assert raw_chunks, "expected chunks from real doc"

    chunk_ids = [str(uuid.uuid4()) for _ in raw_chunks]
    IndexManager._backfill_anchor_chunk_ids(raw_chunks, chunk_ids)

    chunk_objects: list[Chunk] = []
    for chunk_id, chunk in zip(chunk_ids, raw_chunks):
        meta = (chunk.get("metadata") or {}).copy()
        src = chunk.get("source_metadata") or {}
        if src:
            meta.update(src)
        chunk_objects.append(
            Chunk(
                id=chunk_id,
                owner_id=str(meta.get("owner_id") or owner_id),
                content=str(chunk.get("content") or ""),
                metadata=meta,
            )
        )

    assert any((c.metadata or {}).get("chunk_role") == "anchor" for c in chunk_objects)
    assert any((c.metadata or {}).get("chunk_role") == "slice" for c in chunk_objects)
    assert any((c.metadata or {}).get("semantic_unit_type") == "table" for c in chunk_objects)
    assert any((c.metadata or {}).get("semantic_unit_type") == "code" for c in chunk_objects)
    assert any((c.metadata or {}).get("semantic_unit_type") == "list" for c in chunk_objects)
    assert any((c.metadata or {}).get("semantic_unit_type") == "math" for c in chunk_objects)
    assert any((c.metadata or {}).get("semantic_unit_type") == "blockquote" for c in chunk_objects)

    # Ensure every slice has a backfilled anchor_chunk_id.
    slice_missing_anchor = [
        c
        for c in chunk_objects
        if (c.metadata or {}).get("chunk_role") == "slice" and not str((c.metadata or {}).get("anchor_chunk_id") or "").strip()
    ]
    assert not slice_missing_anchor, "expected all slices to have anchor_chunk_id backfilled"

    bm25_dir = tmp_path / "bm25_index"
    bm25_dir.mkdir(parents=True, exist_ok=True)
    bm25_builder = BM25BuilderConfig(index_path=str(bm25_dir)).build()
    ok = bm25_builder.update_index(chunk_objects)
    assert ok, "expected BM25 indexing to succeed"

    multipath = MultiPathRetrieverConfig(
        retrievers=[
            TantivyBM25RetrieverConfig(
                index_config=BM25BuilderConfig(index_path=str(bm25_dir)),
                search_kwargs={"k": 6, "with_score": True, "use_phrase_query": False},
            )
        ],
        fusion_method="rrf",
        rrf_k=60,
        search_kwargs={"k": 6, "with_score": True},
    ).build()

    # Table slice hit should be replaced by table anchor, with matched slice rows merged into content.
    table_hits = multipath.invoke("Enterprise", k=6, owner_id=owner_id)
    assert table_hits, "expected table hits"
    table_top = table_hits[0]
    assert (table_top.metadata or {}).get("chunk_role") == "anchor"
    assert (table_top.metadata or {}).get("semantic_unit_type") == "table"
    assert "matched_slices" in (table_top.metadata or {})
    assert "Enterprise" in (table_top.content or "")

    # Code hit should preserve matched slice content even when returning the anchor.
    code_hits = multipath.invoke("get_user_name", k=6, owner_id=owner_id)
    assert code_hits, "expected hits"
    code_top = next(
        (chunk for chunk in code_hits if (chunk.metadata or {}).get("semantic_unit_type") == "code"),
        None,
    )
    assert code_top is not None, "expected at least one code result"
    assert (code_top.metadata or {}).get("chunk_role") == "anchor"
    matched_code = (code_top.metadata or {}).get("matched_slices") or []
    assert any("get_user_name" in (entry.get("content") or "") for entry in matched_code)
    assert "def get_user_name" in (code_top.content or "")

    # List hit should preserve matched slice content even when returning the anchor.
    list_hits = multipath.invoke("Verify indexes", k=6, owner_id=owner_id)
    assert list_hits, "expected hits"
    list_top = next(
        (
            chunk
            for chunk in list_hits
            if (chunk.metadata or {}).get("semantic_unit_type") == "list"
            and "Deployment Checklist" in (chunk.content or "")
        ),
        None,
    )
    assert list_top is not None, "expected a Deployment Checklist list result"
    assert (list_top.metadata or {}).get("chunk_role") == "anchor"
    matched_list = (list_top.metadata or {}).get("matched_slices") or []
    assert any("Verify indexes" in (entry.get("content") or "") for entry in matched_list)
