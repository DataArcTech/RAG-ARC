import pytest

from pathlib import Path

from config.core.file_management.indexing.bm25_indexing_config import BM25IndexerConfig
from config.encapsulation.database.bm25_config import BM25BuilderConfig
from encapsulation.data_model.schema import Chunk
from encapsulation.database.index_scoping import owner_scoped_dir


@pytest.mark.asyncio
async def test_bm25_rebuild_when_meta_missing(tmp_path):
    config = BM25BuilderConfig(index_path=str(tmp_path))
    indexer = BM25IndexerConfig(index_config=config).build()

    first = Chunk(id="chunk-1", content="hello world", owner_id="owner", metadata={"owner_id": "owner"})
    await indexer.update_index([first])

    meta_dir = owner_scoped_dir(str(tmp_path), owner_id="owner", owner_dirname="owners", global_owner_name="__GLOBAL__")
    meta_path = Path(meta_dir) / "meta.json"
    assert meta_path.exists()

    meta_path.unlink()
    builder = indexer._builder_by_owner.get("owner")
    assert builder is not None
    assert builder._index is not None

    second = Chunk(id="chunk-2", content="second chunk", owner_id="owner", metadata={"owner_id": "owner"})
    await indexer.update_index([second])

    assert meta_path.exists()
