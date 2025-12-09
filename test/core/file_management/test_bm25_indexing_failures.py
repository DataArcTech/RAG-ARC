import asyncio
import threading
import pytest

from encapsulation.data_model.schema import Chunk
from core.file_management.indexing.bm25_indexing import BM25Indexer


class _FakeBuilder:
    def __init__(self):
        self._index = object()

    def update_index(self, chunks):
        return False

    def load_local(self):
        raise FileNotFoundError

    def from_chunks(self, chunks):
        return True

    def delete_index(self, chunk_ids):
        return True


class _FakeIndexConfig:
    def __init__(self, builder):
        self._builder = builder

    def build(self):
        return self._builder


class _FakeConfig:
    def __init__(self, builder):
        self.index_config = _FakeIndexConfig(builder)
        self.batch_size = 1
        self.flush_interval = 0.01


@pytest.mark.asyncio
async def test_bm25_update_index_propagates_flush_failure():
    builder = _FakeBuilder()
    config = _FakeConfig(builder)
    indexer = BM25Indexer(config)
    chunk = Chunk(id="chunk-1", content="foo")

    with pytest.raises(RuntimeError):
        await indexer.update_index([chunk])


class _SuccessBuilder(_FakeBuilder):
    def update_index(self, chunks):
        return True


@pytest.mark.asyncio
async def test_bm25_delete_mutates_pending_during_flush():
    builder = _SuccessBuilder()
    config = _FakeConfig(builder)
    indexer = BM25Indexer(config)
    chunk = Chunk(id="chunk-2", content="bar")

    lock_held = threading.Event()
    proceed = threading.Event()
    race_detected = threading.Event()

    original_build = indexer._build_or_update_index_sync

    def slow_build(chunks_list):
        lock_held.set()
        proceed.wait(timeout=5)
        return original_build(chunks_list)

    indexer._build_or_update_index_sync = slow_build

    original_remove = indexer._remove_pending_chunks

    def instrumented_remove(chunk_ids):
        if indexer._pending_lock.locked():
            race_detected.set()
        return original_remove(chunk_ids)

    indexer._remove_pending_chunks = instrumented_remove

    update_task = asyncio.create_task(indexer.update_index([chunk]))
    await asyncio.sleep(0)

    assert lock_held.wait(timeout=5), "flush lock was never acquired"

    deleter = threading.Thread(target=indexer.delete_chunks, args=([chunk.id],))
    deleter.start()
    deleter.join(timeout=5)
    assert not deleter.is_alive(), "delete_chunks thread did not finish"

    proceed.set()
    await update_task

    assert not race_detected.is_set(), "pending queue mutated while flush was holding the pending lock"
