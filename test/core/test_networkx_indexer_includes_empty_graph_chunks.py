import pytest

from encapsulation.data_model.schema import Chunk
from config.core.file_management.indexing.graph_indexing.networkx_indexing_config import NetworkXGraphIndexerConfig
from config.core.file_management.extractor.heuristic_cooccurrence_extractor_config import (
    HeuristicCooccurrenceExtractorConfig,
)
from config.encapsulation.database.graph_db.networkx_config import NetworkXConfig


@pytest.mark.asyncio
async def test_networkx_indexer_indexes_empty_graph_chunks() -> None:
    extractor = HeuristicCooccurrenceExtractorConfig(max_entities_per_chunk=8, max_cooccurrence_pairs_per_chunk=16)
    store = NetworkXConfig(storage_path=None, unify_entities_by_name=True, auto_save=False)
    cfg = NetworkXGraphIndexerConfig(extractor_config=extractor, graph_store_config=store, index_empty_graph_chunks=True)
    indexer = cfg.build()

    chunks = [
        Chunk(id="c_empty", content="no entities here", metadata={"source_file_id": "doc1"}),
        Chunk(id="c_has_entities", content="Alice met Bob.", metadata={"source_file_id": "doc1"}),
    ]
    chunk_ids = await indexer.update_index(chunks)

    assert set(chunk_ids) == {"c_empty", "c_has_entities"}
    assert "c_empty" in indexer.networkx_store.chunks
    assert "c_has_entities" in indexer.networkx_store.chunks
    assert indexer.networkx_store.graph.has_node("chunk_c_empty")
    assert indexer.networkx_store.graph.has_node("chunk_c_has_entities")

