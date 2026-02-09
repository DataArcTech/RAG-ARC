from encapsulation.data_model.schema import Chunk


def test_prune_chunks_by_file_prefers_high_score_file_over_many_lows() -> None:
    from core.retrieval.file_pruning import prune_chunks_by_file

    # file_a: 1 very strong chunk
    # file_b: many low chunks; sum is higher but sqrt(n+1) should dampen
    chunks = []
    chunks.append(
        Chunk(
            id="a1",
            content="A strong relevant snippet",
            metadata={"source_file_id": "file-a", "filename": "A.pdf", "score": 10.0},
        )
    )
    for i in range(10):
        chunks.append(
            Chunk(
                id=f"b{i}",
                content="B weak snippet",
                metadata={"source_file_id": "file-b", "filename": "B.pdf", "score": 1.2},
            )
        )

    out, info = prune_chunks_by_file(chunks, enabled=True, max_files=1, max_chunks_per_file=6)
    assert info.enabled is True
    assert info.files_kept == 1
    assert len(out) == 1  # max_files=1, file-a should win
    assert (getattr(out[0], "metadata", {}) or {}).get("source_file_id") == "file-a"


def test_prune_chunks_by_file_limits_chunks_per_file_and_keeps_top_files() -> None:
    from core.retrieval.file_pruning import prune_chunks_by_file

    chunks = []
    # Two files, each with descending scores.
    for i, s in enumerate([5, 4, 3, 2, 1, 0.5, 0.1], start=1):
        chunks.append(Chunk(id=f"a{i}", content="A", metadata={"source_file_id": "file-a", "score": s}))
    for i, s in enumerate([4.9, 4.8, 4.7], start=1):
        chunks.append(Chunk(id=f"b{i}", content="B", metadata={"source_file_id": "file-b", "score": s}))

    out, info = prune_chunks_by_file(chunks, enabled=True, max_files=2, max_chunks_per_file=6)
    assert info.files_kept == 2
    # file-a limited to 6 chunks
    assert sum(1 for c in out if (c.metadata or {}).get("source_file_id") == "file-a") == 6
    # file-b keeps all 3 chunks
    assert sum(1 for c in out if (c.metadata or {}).get("source_file_id") == "file-b") == 3

