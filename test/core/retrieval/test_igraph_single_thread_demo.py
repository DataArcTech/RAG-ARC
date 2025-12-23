"""
Single-thread demo for the igraph (SQLite + igraph) HippoRAG backend.

This is intentionally not a concurrency test:
- The igraph store has shared mutable state (SQLite connection, igraph graph, caches).
- In production, concurrent indexing + retrieval should be guarded externally (lock/process separation),
  or routed to a backend that is designed for concurrent access.
"""

import os
import sys
import sqlite3
import uuid
from dataclasses import dataclass


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from core.retrieval.graph_retrieveal.pruned_hipporag import PrunedHippoRAGRetriever


@dataclass
class _StubConfig:
    type: str = "stub_hipporag"
    enable_llm_reranking: bool = False
    fact_retrieval_top_k: int = 5
    enable_pruning: bool = False
    include_chunk_neighbors: bool = False
    expansion_hops: int = 1
    damping_factor: float = 0.5
    passage_node_weight: float = 1.0
    max_neighbors: int = 5
    query_aware_multiplier: float = 0.0
    query_aware_min_k: int = 1
    query_aware_max_k: int = 5
    graph_config: object | None = None


def test_igraph_retriever_node_mappings_single_thread_demo():
    owner_a = uuid.uuid4()
    owner_b = uuid.uuid4()

    chunks_a = [f"{owner_a}_chunk_1", f"{owner_a}_chunk_2"]
    chunks_b = [f"{owner_b}_chunk_1"]

    # SQLite connection: check_same_thread=False allows access from multiple threads,
    # but the igraph backend is still best treated as single-thread unless you add your own locks.
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
            self.conn = conn
            # The retriever maps chunk_id -> graph index; for this demo we only need the mapping.
            self.node_to_idx = {chunk_id: idx for idx, chunk_id in enumerate(chunks_a + chunks_b)}

    retriever = object.__new__(PrunedHippoRAGRetriever)
    retriever.config = _StubConfig()
    retriever.graph_store = _StubIGraphStore()

    # Build mappings for owner A.
    retriever._build_node_mappings(owner_id=owner_a)
    assert retriever.passage_node_keys == chunks_a

    # Build mappings for owner B in the same thread.
    # This overwrites the per-thread mapping state (expected in single-thread sequential execution).
    retriever._build_node_mappings(owner_id=owner_b)
    assert retriever.passage_node_keys == chunks_b

