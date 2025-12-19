import os
import sys
import uuid
import threading
import sqlite3
from dataclasses import dataclass

import numpy as np


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from core.retrieval.graph_retrieveal.pruned_hipporag import PrunedHippoRAGRetriever
from core.retrieval.graph_retrieveal.pruned_hipporag_neo4j import PrunedHippoRAGNeo4jRetriever


class _StubEmbeddingModel:
    def get_embedding_dimension(self) -> int:
        return 3


class _StubNeo4jStore:
    def __init__(self, chunks_by_owner: dict[str | None, list[str]]):
        self.embedding_model = _StubEmbeddingModel()
        self._chunks_by_owner = chunks_by_owner
        self.chunk_embeddings: dict[str, np.ndarray] = {}
        for chunk_ids in chunks_by_owner.values():
            if not chunk_ids:
                continue
            for chunk_id in chunk_ids:
                self.chunk_embeddings[chunk_id] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def get_cache_version(self):
        return 1

    def _execute_query(self, _query: str, params: dict | None = None):
        owner_id = None if not params else params.get("owner_id")
        chunk_ids = self._chunks_by_owner.get(owner_id, [])
        return [{"chunk_id": chunk_id} for chunk_id in chunk_ids]


@dataclass
class _StubGraphConfig:
    store: object
    storage_path: str = "/tmp"
    index_name: str = "stub"

    def build(self):
        return self.store


@dataclass
class _StubHippoConfig:
    graph_config: _StubGraphConfig
    llm_config: object | None = None
    type: str = "stub_hipporag"

    expansion_hops: int = 1
    include_chunk_neighbors: bool = False

    enable_pruning: bool = False
    max_neighbors: int = 5
    query_aware_multiplier: float = 0.0
    query_aware_min_k: int = 1
    query_aware_max_k: int = 5


def _run_mapping_race(build_fn, read_fn, owner_a, owner_b):
    proceed = threading.Event()
    ready_a = threading.Event()
    ready_b = threading.Event()
    results: dict[uuid.UUID, list[str]] = {}

    def worker(owner_id: uuid.UUID, ready: threading.Event):
        build_fn(owner_id)
        ready.set()
        proceed.wait(timeout=5)
        results[owner_id] = read_fn()

    thread_a = threading.Thread(target=worker, args=(owner_a, ready_a))
    thread_a.start()
    assert ready_a.wait(timeout=5)

    thread_b = threading.Thread(target=worker, args=(owner_b, ready_b))
    thread_b.start()
    assert ready_b.wait(timeout=5)

    proceed.set()
    thread_a.join(timeout=5)
    thread_b.join(timeout=5)
    assert not thread_a.is_alive()
    assert not thread_b.is_alive()

    return results


def test_pruned_hipporag_neo4j_thread_safe_owner_mappings():
    owner_a = uuid.uuid4()
    owner_b = uuid.uuid4()

    chunks_a = [f"{owner_a}_chunk_1", f"{owner_a}_chunk_2"]
    chunks_b = [f"{owner_b}_chunk_1", f"{owner_b}_chunk_2", f"{owner_b}_chunk_3"]

    store = _StubNeo4jStore(
        chunks_by_owner={
            str(owner_a): chunks_a,
            str(owner_b): chunks_b,
            None: chunks_a + chunks_b,
        }
    )
    config = _StubHippoConfig(graph_config=_StubGraphConfig(store=store))
    retriever = PrunedHippoRAGNeo4jRetriever(config)

    results = _run_mapping_race(
        build_fn=lambda owner_id: retriever._build_node_mappings(owner_id=owner_id),
        read_fn=lambda: list(retriever.passage_node_keys),
        owner_a=owner_a,
        owner_b=owner_b,
    )

    assert results[owner_a] == chunks_a
    assert results[owner_b] == chunks_b


def test_pruned_hipporag_igraph_thread_safe_owner_mappings(tmp_path):
    owner_a = uuid.uuid4()
    owner_b = uuid.uuid4()

    chunks_a = [f"{owner_a}_chunk_1", f"{owner_a}_chunk_2"]
    chunks_b = [f"{owner_b}_chunk_1", f"{owner_b}_chunk_2", f"{owner_b}_chunk_3"]

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE chunks (chunk_id TEXT, owner_id TEXT)")
    cursor.executemany(
        "INSERT INTO chunks (chunk_id, owner_id) VALUES (?, ?)",
        [(chunk_id, str(owner_a)) for chunk_id in chunks_a] + [(chunk_id, str(owner_b)) for chunk_id in chunks_b],
    )
    conn.commit()

    class _StubIGraphStore:
        def __init__(self):
            self.embedding_model = _StubEmbeddingModel()
            self.conn = conn
            self.node_to_idx = {chunk_id: idx for idx, chunk_id in enumerate(chunks_a + chunks_b)}

    store = _StubIGraphStore()
    graph_config = _StubGraphConfig(store=store, storage_path=str(tmp_path), index_name="stub_graph")
    config = _StubHippoConfig(graph_config=graph_config)
    retriever = PrunedHippoRAGRetriever(config)

    results = _run_mapping_race(
        build_fn=lambda owner_id: retriever._build_node_mappings(owner_id=owner_id),
        read_fn=lambda: list(retriever.passage_node_keys),
        owner_a=owner_a,
        owner_b=owner_b,
    )

    assert results[owner_a] == chunks_a
    assert results[owner_b] == chunks_b
