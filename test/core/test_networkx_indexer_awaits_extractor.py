import asyncio
from types import SimpleNamespace

import pytest

from encapsulation.data_model.schema import Chunk, GraphData
from core.file_management.indexing.graph_indexing.networkx_indexing import NetworkXGraphIndexer


class _FakeExtractor:
    async def __call__(self, chunks):
        await asyncio.sleep(0)
        for chunk in chunks:
            chunk.graph = GraphData(
                entities=[{"id": "e1", "entity_name": "Apple", "entity_type": "ORG", "attributes": {}}],
                relations=[],
            )
        return chunks


class _FakeExtractorConfig:
    def build(self):
        return _FakeExtractor()


class _FakeGraphStore:
    storage_path = None

    def __init__(self):
        self.added_chunks = []
        self.added_graph_data = []

    def add_chunk(self, chunk):
        self.added_chunks.append(chunk.id)

    def add_graph_data(self, graph_data, chunk_id):
        self.added_graph_data.append((chunk_id, len(graph_data.entities), len(graph_data.relations)))


class _FakeGraphStoreConfig:
    def build(self):
        return _FakeGraphStore()


@pytest.mark.asyncio
async def test_networkx_indexer_awaits_extractor():
    cfg = SimpleNamespace(extractor_config=_FakeExtractorConfig(), graph_store_config=_FakeGraphStoreConfig())
    indexer = NetworkXGraphIndexer(cfg)
    chunks = [Chunk(id="c1", content="Apple is a company.")]
    chunk_ids = await indexer.update_index(chunks)

    assert chunk_ids == ["c1"]
    assert indexer.networkx_store.added_chunks == ["c1"]
    assert indexer.networkx_store.added_graph_data
