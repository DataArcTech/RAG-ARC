import pytest


class _DummyMetaStore:
    def __init__(self) -> None:
        self.status_updates: list[tuple[str, str]] = []

    def update_file_status(self, file_id: str, status, **_kwargs):  # noqa: ANN001
        # Store the raw enum/object; tests assert call ordering only.
        self.status_updates.append((file_id, str(status)))
        return True


class _DummyFileMeta:
    def __init__(self, *, filename: str, owner_id: str) -> None:
        self.filename = filename
        self.owner_id = owner_id


class _DummyFileStorage:
    def __init__(self, meta: _DummyFileMeta) -> None:
        self._meta = meta
        self.metadata_store = _DummyMetaStore()

    def get_file_metadata(self, file_id: str):  # noqa: ANN001
        if file_id == "missing":
            return None
        return self._meta


class _DummyParsedContentStorage:
    def store_parsed_content(self, *, source_file_id: str, parser_type: str, parsed_data: bytes, content_type: str, **_kwargs):  # noqa: ANN001
        assert source_file_id
        assert parser_type
        assert parsed_data
        assert content_type == "text/markdown"
        return "pc_1"


class _DummyChunkStorage:
    def __init__(self) -> None:
        self._stored: list[tuple[int, str]] = []

    def store_chunks_batch(  # noqa: ANN001
        self,
        *,
        source_parsed_content_id: str,
        chunker_type: str,
        chunks: list[dict],
        owner_id,
        validate_after_store: bool,
        **_kwargs,
    ):
        assert source_parsed_content_id == "pc_1"
        assert chunker_type
        assert owner_id
        assert validate_after_store is True or validate_after_store is False
        out: list[tuple[int, str]] = []
        for idx, _chunk in enumerate(chunks):
            out.append((idx, f"c_{idx}"))
        self._stored = out
        return out

    def overwrite_chunk_json(self, chunk_id: str, chunk_dict: dict) -> bool:  # noqa: ANN001
        # Used by anchor id backfill path; keep a permissive stub.
        return bool(chunk_id) and isinstance(chunk_dict, dict)


class _DummyChunker:
    def get_chunker_info(self):  # noqa: ANN001
        return {"strategy": "dummy"}

    def chunk_text(self, *, text: str, metadata: dict, **_kwargs):  # noqa: ANN001
        assert text
        assert metadata.get("source_file_id")
        assert metadata.get("parsed_content_id") == "pc_1"
        assert metadata.get("owner_id")
        return [
            {"content": "a", "metadata": dict(metadata)},
            {"content": "b", "metadata": dict(metadata)},
        ]


class _DummyIndexer:
    async def update_index(self, chunk_objects):  # noqa: ANN001
        # Return a non-empty list so indexing is considered successful.
        return [c.id for c in chunk_objects]


@pytest.mark.asyncio
async def test_process_file_from_parsed_markdown_smoke(caplog):
    from core.file_management.index_manager_pipeline import _IndexManagerPipelineMixin

    class _Mgr(_IndexManagerPipelineMixin):
        def __init__(self) -> None:
            self.file_storage = _DummyFileStorage(_DummyFileMeta(filename="x.pdf", owner_id="owner_1"))
            self.parsed_content_storage = _DummyParsedContentStorage()
            self.chunk_storage = _DummyChunkStorage()
            self.chunker = _DummyChunker()
            self.indexers = [_DummyIndexer()]
            self.pageindex_service = None

        # status hooks are synchronous (called via thread_pool.run_blocking)
        def _update_file_status_to_parsed(self, file_id: str, **_kwargs):  # noqa: ANN001
            return True

        def _update_file_status_to_indexed(self, file_id: str, **_kwargs):  # noqa: ANN001
            return True

        def _update_indexed_chunks_status(self, chunk_ids, indexing_results, **_kwargs):  # noqa: ANN001
            assert chunk_ids
            assert indexing_results
            return len(chunk_ids)

    mgr = _Mgr()
    with caplog.at_level("WARNING"):
        out = await mgr.process_file_from_parsed_markdown(file_id="f_1", parsed_markdown="# hello")
    assert out["success"] is True
    assert out["parsed_content_id"] == "pc_1"
    assert len(out["chunk_ids"]) == 2
    assert out["metadata"]["parser_type"] == "preparsed_markdown"
    assert out["metadata"]["chunker_type"] == "dummy"
    assert out["metadata"]["timings_s"]["total"] >= 0
    assert not any("Anchor chunk id backfill failed" in rec.message for rec in caplog.records)
