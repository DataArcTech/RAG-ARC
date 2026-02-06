import pytest

from core.file_management.extractor.metadata_keys import EXTRACTION_ERROR_KEY
from core.file_management.indexing.graph_indexing.pruned_hipporag_indexing import PrunedHippoRAGIndexer
from encapsulation.data_model.schema import Chunk, GraphData


class _Builder:
    def __init__(self, built):
        self._built = built

    def build(self):
        return self._built


class _DummyConfig:
    def __init__(self, *, extractor, graph_store):
        self.extractor_config = _Builder(extractor)
        self.graph_store_config = _Builder(graph_store)


class _DummyExtractor:
    async def __call__(self, chunks: list[Chunk]) -> list[Chunk]:
        out: list[Chunk] = []
        for chunk in chunks:
            if chunk.id == "chunk_ok":
                chunk.graph = GraphData(
                    entities=[{"id": "e1", "entity_name": "Ping An", "entity_type": "Company"}],
                    relations=[],
                    metadata={},
                )
            elif chunk.id == "chunk_empty":
                chunk.graph = GraphData()
            elif chunk.id == "chunk_failed":
                chunk.graph = GraphData(metadata={EXTRACTION_ERROR_KEY: {"message": "boom"}})
                chunk.metadata[EXTRACTION_ERROR_KEY] = {"message": "boom"}
            out.append(chunk)
        return out


class _DummyGraphStore:
    def __init__(self):
        self.seen: list[Chunk] = []

    def update_index(self, chunks: list[Chunk]) -> bool:
        self.seen = list(chunks)
        return True


@pytest.mark.asyncio
async def test_indexer_does_not_drop_empty_or_failed_graph_chunks():
    extractor = _DummyExtractor()
    store = _DummyGraphStore()
    indexer = PrunedHippoRAGIndexer(_DummyConfig(extractor=extractor, graph_store=store))

    chunks = [
        Chunk(id="chunk_ok", content="Ping An is an insurance company."),
        Chunk(id="chunk_empty", content="This passage has no extractable facts."),
        Chunk(id="chunk_failed", content="This passage triggers extraction failure."),
    ]

    indexed_ids = await indexer.update_index(chunks)
    if isinstance(indexed_ids, dict):
        indexed_ids = indexed_ids.get("indexed_ids")

    assert indexed_ids == ["chunk_ok", "chunk_empty", "chunk_failed"]
    assert [c.id for c in store.seen] == ["chunk_ok", "chunk_empty", "chunk_failed"]
    assert store.seen[1].graph.is_empty()
    assert store.seen[2].graph.is_empty()
    assert EXTRACTION_ERROR_KEY in store.seen[2].graph.metadata
