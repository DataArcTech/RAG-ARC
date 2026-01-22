from encapsulation.data_model.schema import Chunk

from core.utils.chunk_dedupe import dedupe_chunks


def test_dedupe_prefers_first_occurrence() -> None:
    chunks = [
        Chunk(
            id="uuid-1",
            content="A",
            metadata={"source_file_id": "file-1", "chunk_index": 0, "start_idx": 0, "end_idx": 10},
        ),
        # Same logical chunk (same file+range) but different runtime UUID.
        Chunk(
            id="uuid-2",
            content="A",
            metadata={"source_file_id": "file-1", "chunk_index": 0, "start_idx": 0, "end_idx": 10},
        ),
        Chunk(
            id="uuid-3",
            content="B",
            metadata={"source_file_id": "file-1", "chunk_index": 1, "start_idx": 11, "end_idx": 20},
        ),
    ]
    out = dedupe_chunks(chunks)
    assert [c.id for c in out] == ["uuid-1", "uuid-3"]


def test_dedupe_does_not_collapse_across_files() -> None:
    chunks = [
        Chunk(id=None, content="boilerplate", metadata={"source_file_id": "file-1", "chunk_index": 0}),
        Chunk(id=None, content="boilerplate", metadata={"source_file_id": "file-2", "chunk_index": 0}),
    ]
    out = dedupe_chunks(chunks)
    assert len(out) == 2


def test_dedupe_by_id_when_present() -> None:
    chunks = [
        Chunk(id="same", content="A", metadata={"source_file_id": "file-1", "chunk_index": 0}),
        Chunk(id="same", content="A", metadata={"source_file_id": "file-1", "chunk_index": 0}),
        Chunk(id="other", content="B", metadata={"source_file_id": "file-1", "chunk_index": 1}),
    ]
    out = dedupe_chunks(chunks)
    assert [c.id for c in out] == ["same", "other"]

