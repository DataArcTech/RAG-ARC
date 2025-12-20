from dataclasses import dataclass

from encapsulation.data_model.schema import Chunk
from core.retrieval.multipath import MultiPathRetriever
from config.output_limits import SEMANTIC_UNIT_MAX_MATCHED_SLICES, TABLE_MAX_MERGED_ROWS


class _DummyIndex:
    def __init__(self, chunks: list[Chunk]):
        self._chunks = {chunk.id: chunk for chunk in chunks if chunk.id}

    def get_by_ids(self, ids: list[str]) -> list[Chunk]:
        return [self._chunks[chunk_id] for chunk_id in ids if chunk_id in self._chunks]


class _DummyRetriever:
    def __init__(self, *, results: list[Chunk], index: _DummyIndex):
        self._results = results
        self._index = index

    def invoke(self, query: str, **kwargs):  # noqa: ARG002
        return list(self._results)

    @property
    def index(self):
        return self._index


@dataclass
class _DummyConfig:
    fusion_method: str = "rrf"
    rrf_k: int = 60
    weights: list[float] | None = None
    retrievers: list[object] = None  # only used for weights defaults
    search_kwargs: dict = None
    built_retrievers: list[object] | None = None
    fusion_instance: object | None = None


def test_multipath_backfills_anchor_and_attaches_slices():
    owner_id = "12345678-1234-5678-1234-567812345678"
    anchor_id = "ANCHOR_1"

    anchor = Chunk(
        id=anchor_id,
        owner_id=owner_id,
        content="Table: Example\n| A | B |\n|---|---|",
        metadata={"chunk_role": "anchor", "semantic_unit_type": "table", "anchor_is_summary": True},
    )
    slice_chunk = Chunk(
        id="SLICE_1",
        owner_id=owner_id,
        content="Table: Example\n| A | B |\n|---|---|\n| 1 | 2 |",
        metadata={"chunk_role": "slice", "semantic_unit_type": "table", "anchor_chunk_id": anchor_id, "score": 0.9},
    )

    dummy_index = _DummyIndex([anchor])
    dummy_retriever = _DummyRetriever(results=[slice_chunk], index=dummy_index)

    config = _DummyConfig(
        retrievers=[object()],
        search_kwargs={"k": 5, "with_score": True},
        built_retrievers=[dummy_retriever],
    )
    multipath = MultiPathRetriever(config)  # type: ignore[arg-type]

    results = multipath._get_relevant_chunks("example", k=1, owner_id=owner_id)

    assert len(results) == 1
    assert results[0].id == anchor_id
    assert "| 1 | 2 |" in results[0].content
    assert "matched_slices" in (results[0].metadata or {})
    assert results[0].metadata["matched_slices"][0]["id"] == "SLICE_1"
    assert "| 1 | 2 |" in str((results[0].metadata or {}).get("prompt_text") or "")


def test_multipath_does_not_mutate_index_backed_anchor():
    owner_id = "12345678-1234-5678-1234-567812345678"
    anchor_id = "ANCHOR_1"

    anchor = Chunk(
        id=anchor_id,
        owner_id=owner_id,
        content="Table: Example\n| A | B |\n|---|---|",
        metadata={"chunk_role": "anchor", "semantic_unit_type": "table", "anchor_is_summary": True},
    )
    slice_chunk = Chunk(
        id="SLICE_1",
        owner_id=owner_id,
        content="Table: Example\n| A | B |\n|---|---|\n| 1 | 2 |",
        metadata={"chunk_role": "slice", "semantic_unit_type": "table", "anchor_chunk_id": anchor_id, "score": 0.9},
    )

    dummy_index = _DummyIndex([anchor])
    dummy_retriever = _DummyRetriever(results=[slice_chunk], index=dummy_index)

    config = _DummyConfig(
        retrievers=[object()],
        search_kwargs={"k": 5, "with_score": True},
        built_retrievers=[dummy_retriever],
    )
    multipath = MultiPathRetriever(config)  # type: ignore[arg-type]

    first = multipath._get_relevant_chunks("example", k=1, owner_id=owner_id)[0]
    second = multipath._get_relevant_chunks("example", k=1, owner_id=owner_id)[0]

    assert first.id == anchor_id
    assert second.id == anchor_id

    # Index-backed anchor should not accumulate per-query evidence fields.
    stored_anchor = dummy_index._chunks[anchor_id]
    assert "matched_slices" not in (stored_anchor.metadata or {})
    assert "| 1 | 2 |" not in stored_anchor.content


def test_multipath_attaches_slices_when_anchor_also_returned():
    owner_id = "12345678-1234-5678-1234-567812345678"
    anchor_id = "ANCHOR_1"

    anchor = Chunk(
        id=anchor_id,
        owner_id=owner_id,
        content="Table: Example\n| A | B |\n|---|---|",
        metadata={"chunk_role": "anchor", "semantic_unit_type": "table", "anchor_is_summary": True},
    )
    slice_chunk = Chunk(
        id="SLICE_1",
        owner_id=owner_id,
        content="Table: Example\n| A | B |\n|---|---|\n| 1 | 2 |",
        metadata={"chunk_role": "slice", "semantic_unit_type": "table", "anchor_chunk_id": anchor_id, "score": 0.9},
    )

    dummy_index = _DummyIndex([anchor])
    dummy_retriever = _DummyRetriever(results=[anchor, slice_chunk], index=dummy_index)

    config = _DummyConfig(
        retrievers=[object()],
        search_kwargs={"k": 5, "with_score": True},
        built_retrievers=[dummy_retriever],
    )
    multipath = MultiPathRetriever(config)  # type: ignore[arg-type]

    results = multipath._get_relevant_chunks("example", k=2, owner_id=owner_id)

    assert len(results) == 1
    assert results[0].id == anchor_id
    assert "| 1 | 2 |" in results[0].content
    assert "matched_slices" in (results[0].metadata or {})
    assert results[0].metadata["matched_slices"][0]["id"] == "SLICE_1"


def test_multipath_limits_matched_slices_to_top_scores():
    owner_id = "12345678-1234-5678-1234-567812345678"
    anchor_id = "ANCHOR_1"

    anchor = Chunk(
        id=anchor_id,
        owner_id=owner_id,
        content="Table: Example\n| A | B |\n|---|---|",
        metadata={"chunk_role": "anchor", "semantic_unit_type": "table", "anchor_is_summary": True},
    )

    slices: list[Chunk] = []
    for idx in range(1, 6):
        score = float(idx) / 10.0
        slices.append(
            Chunk(
                id=f"SLICE_{idx}",
                owner_id=owner_id,
                content="\n".join(
                    [
                        "Table: Example",
                        "| A | B |",
                        "|---|---|",
                        f"| {idx} | value_{idx} |",
                    ]
                ),
                metadata={
                    "chunk_role": "slice",
                    "semantic_unit_type": "table",
                    "anchor_chunk_id": anchor_id,
                    "score": score,
                    "slice_index": idx,
                },
            )
        )

    dummy_index = _DummyIndex([anchor])
    dummy_retriever = _DummyRetriever(results=slices, index=dummy_index)

    config = _DummyConfig(
        retrievers=[object()],
        search_kwargs={"k": 10, "with_score": True},
        built_retrievers=[dummy_retriever],
    )
    multipath = MultiPathRetriever(config)  # type: ignore[arg-type]

    results = multipath._get_relevant_chunks("example", k=10, owner_id=owner_id)

    assert len(results) == 1
    out = results[0]
    matched = (out.metadata or {}).get("matched_slices") or []

    if SEMANTIC_UNIT_MAX_MATCHED_SLICES is not None:
        assert len(matched) == SEMANTIC_UNIT_MAX_MATCHED_SLICES
    else:
        assert len(matched) == 5

    matched_ids = [entry.get("id") for entry in matched]
    # RRF fusion rewrites per-chunk scores based on rank; earlier results rank higher.
    assert matched_ids[0] == "SLICE_1"
    assert "SLICE_5" not in matched_ids
    assert "| 1 | value_1 |" in out.content
    assert "| 5 | value_5 |" not in out.content


def test_multipath_limits_merged_table_rows():
    owner_id = "12345678-1234-5678-1234-567812345678"
    anchor_id = "ANCHOR_1"

    anchor = Chunk(
        id=anchor_id,
        owner_id=owner_id,
        content="Table: Example\n| A | B |\n|---|---|",
        metadata={"chunk_role": "anchor", "semantic_unit_type": "table", "anchor_is_summary": True},
    )

    slices: list[Chunk] = []
    # 3 slices, each with 15 unique rows => 45 total rows before limiting/dedup.
    for slice_idx in range(1, 4):
        rows = [f"| {slice_idx}-{row_idx} | v_{slice_idx}_{row_idx} |" for row_idx in range(1, 16)]
        content = "\n".join(["Table: Example", "| A | B |", "|---|---|", *rows])
        slices.append(
            Chunk(
                id=f"SLICE_{slice_idx}",
                owner_id=owner_id,
                content=content,
                metadata={
                    "chunk_role": "slice",
                    "semantic_unit_type": "table",
                    "anchor_chunk_id": anchor_id,
                    "score": 1.0,
                    "slice_index": slice_idx,
                },
            )
        )

    dummy_index = _DummyIndex([anchor])
    dummy_retriever = _DummyRetriever(results=slices, index=dummy_index)

    config = _DummyConfig(
        retrievers=[object()],
        search_kwargs={"k": 10, "with_score": True},
        built_retrievers=[dummy_retriever],
    )
    multipath = MultiPathRetriever(config)  # type: ignore[arg-type]

    out = multipath._get_relevant_chunks("example", k=10, owner_id=owner_id)[0]

    merged_rows = [
        line
        for line in out.content.splitlines()
        if line.strip().startswith("|") and not line.strip().startswith("|---")
    ]
    # Drop header row ("| A | B |") from the count.
    merged_data_rows = [line for line in merged_rows if line.strip() != "| A | B |"]

    if TABLE_MAX_MERGED_ROWS is not None:
        assert len(merged_data_rows) <= TABLE_MAX_MERGED_ROWS
    else:
        assert len(merged_data_rows) == 45


def test_multipath_merges_code_slice_into_anchor_content():
    owner_id = "12345678-1234-5678-1234-567812345678"
    anchor_id = "ANCHOR_CODE"

    anchor = Chunk(
        id=anchor_id,
        owner_id=owner_id,
        content="```python\n# preview only\n```",
        metadata={"chunk_role": "anchor", "semantic_unit_type": "code", "anchor_is_summary": True},
    )
    slice_chunk = Chunk(
        id="SLICE_CODE",
        owner_id=owner_id,
        content="```python\ndef foo():\n    return 1\n```",
        metadata={"chunk_role": "slice", "semantic_unit_type": "code", "anchor_chunk_id": anchor_id, "score": 0.9},
    )

    dummy_index = _DummyIndex([anchor])
    dummy_retriever = _DummyRetriever(results=[slice_chunk], index=dummy_index)

    config = _DummyConfig(
        retrievers=[object()],
        search_kwargs={"k": 5, "with_score": True},
        built_retrievers=[dummy_retriever],
    )
    multipath = MultiPathRetriever(config)  # type: ignore[arg-type]

    out = multipath._get_relevant_chunks("foo", k=1, owner_id=owner_id)[0]
    assert out.id == anchor_id
    assert out.metadata["chunk_role"] == "anchor"
    assert "matched_slices" in (out.metadata or {})
    assert "def foo" in out.content


def test_multipath_merges_list_slice_into_anchor_content():
    owner_id = "12345678-1234-5678-1234-567812345678"
    anchor_id = "ANCHOR_LIST"

    anchor = Chunk(
        id=anchor_id,
        owner_id=owner_id,
        content="Checklist\n- item A",
        metadata={"chunk_role": "anchor", "semantic_unit_type": "list", "anchor_is_summary": True},
    )
    slice_chunk = Chunk(
        id="SLICE_LIST",
        owner_id=owner_id,
        content="Checklist\n- item A\n- item B",
        metadata={"chunk_role": "slice", "semantic_unit_type": "list", "anchor_chunk_id": anchor_id, "score": 0.9},
    )

    dummy_index = _DummyIndex([anchor])
    dummy_retriever = _DummyRetriever(results=[slice_chunk], index=dummy_index)

    config = _DummyConfig(
        retrievers=[object()],
        search_kwargs={"k": 5, "with_score": True},
        built_retrievers=[dummy_retriever],
    )
    multipath = MultiPathRetriever(config)  # type: ignore[arg-type]

    out = multipath._get_relevant_chunks("item B", k=1, owner_id=owner_id)[0]
    assert out.id == anchor_id
    assert out.metadata["chunk_role"] == "anchor"
    assert "matched_slices" in (out.metadata or {})
    assert "item B" in out.content
