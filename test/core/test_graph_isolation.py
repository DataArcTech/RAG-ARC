import uuid
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from encapsulation.data_model.schema import Chunk, GraphData
from encapsulation.database.graph_db.pruned_hipporag_neo4j import PrunedHippoRAGNeo4jStore
from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id
from core.retrieval.graph_retrieveal.pruned_hipporag import PrunedHippoRAGRetriever
from core.retrieval.graph_retrieveal.pruned_hipporag_neo4j import PrunedHippoRAGNeo4jRetriever
from core.utils.owner_guard import is_admin_owner
from core.utils.rwlock import RWLock

RAW_STORE_CLASS = getattr(PrunedHippoRAGNeo4jStore, "__wrapped__", PrunedHippoRAGNeo4jStore)


class _FakeResult(list):
    """Simple iterable result placeholder for mocked Neo4j transactions."""


class _FakeTransaction:
    def __init__(self):
        self.run_calls = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def run(self, query, params=None):
        self.run_calls.append((query, params))
        if "UNWIND $entities AS entity" in query and params:
            return _FakeResult(
                [{'entity_id': entity['entity_id'], 'is_new': True} for entity in params.get('entities', [])]
            )
        return _FakeResult([])

    def commit(self):
        return True


class _FakeSession:
    def __init__(self, tx):
        self._tx = tx

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def begin_transaction(self):
        return self._tx


class _FakeDriver:
    def __init__(self):
        self.transactions = []

    def session(self, database=None):
        tx = _FakeTransaction()
        self.transactions.append(tx)
        return _FakeSession(tx)

    def close(self):
        return True


def _build_graph(entity_a: str, entity_b: str):
    return GraphData(
        entities=[
            {'entity_name': entity_a, 'entity_type': 'Entity'},
            {'entity_name': entity_b, 'entity_type': 'Entity'},
        ],
        relations=[[entity_a, 'related_to', entity_b]]
    )


def _make_store():
    store = object.__new__(RAW_STORE_CLASS)
    store._driver = _FakeDriver()
    store.database = "neo4j"
    store._cache_loaded = False
    store._graph_cache = {}
    store._entity_chunk_count_cache = {}
    store._rwlock = RWLock()
    store.chunk_embeddings = {}
    store._chunk_embeddings_array = None
    store._chunk_ids_list = None
    return store


def test_batch_add_chunks_and_graph_data_scopes_owner_metadata():
    """Verify that chunks, entities, mentions, and facts carry owner-specific metadata."""
    store = _make_store()
    owner_a = str(uuid.uuid4())
    owner_b = str(uuid.uuid4())

    chunk_a = Chunk(
        id="chunk-a",
        content="Alpha content",
        owner_id=owner_a,
        metadata={'owner_id': owner_a},
        graph=_build_graph("Alpha Corp", "Beta Corp")
    )
    chunk_b = Chunk(
        id="chunk-b",
        content="Beta content",
        owner_id=owner_b,
        metadata={'owner_id': owner_b},
        graph=_build_graph("Alpha Corp", "Beta Corp")
    )

    store._batch_add_chunks_and_graph_data([chunk_a, chunk_b])
    run_calls = store._driver.transactions[-1].run_calls

    chunk_payload = next(params for query, params in run_calls if "UNWIND $chunks" in query)['chunks']
    assert {c['owner_id'] for c in chunk_payload} == {owner_a, owner_b}

    entity_payload = next(params for query, params in run_calls if "UNWIND $entities" in query)['entities']
    owner_a_entities = [e for e in entity_payload if e['owner_id'] == owner_a]
    owner_b_entities = [e for e in entity_payload if e['owner_id'] == owner_b]
    assert owner_a_entities and owner_b_entities

    for record in owner_a_entities:
        type_key = record.get("entity_type_key") or "entity"
        recalculated = compute_mdhash_id(f"{record['entity_name_normalized']}|{type_key}", prefix='entity-', owner_id=owner_a)
        assert record['entity_id'] == recalculated

    for record in owner_b_entities:
        type_key = record.get("entity_type_key") or "entity"
        recalculated = compute_mdhash_id(f"{record['entity_name_normalized']}|{type_key}", prefix='entity-', owner_id=owner_b)
        assert record['entity_id'] == recalculated

    overlap = {
        e['entity_name_normalized']: e['entity_id']
        for e in owner_b_entities
    }
    for entity in owner_a_entities:
        if entity['entity_name_normalized'] in overlap:
            assert entity['entity_id'] != overlap[entity['entity_name_normalized']]

    mention_payload = next(params for query, params in run_calls if "UNWIND $mentions" in query)['mentions']
    assert {m['owner_id'] for m in mention_payload} == {owner_a, owner_b}

    fact_payload = next(params for query, params in run_calls if "UNWIND $facts" in query)['facts']
    assert {f['owner_id'] for f in fact_payload} == {owner_a, owner_b}
    sample_fact = fact_payload[0]
    expected_fact_id = compute_mdhash_id(
        f"{sample_fact['head_id']}|{sample_fact['relation_type']}|{sample_fact['tail_id']}",
        prefix='fact-',
        owner_id=sample_fact['owner_id'],
    )
    assert sample_fact['fact_id'] == expected_fact_id


def test_graph_cache_filters_neighbors_by_owner():
    """Ensure neighbor lookups never mix owners when cache is loaded."""
    store = _make_store()
    owner_a = uuid.uuid4()
    owner_b = uuid.uuid4()
    store._cache_loaded = True
    store._graph_cache = {
        store._owner_key(owner_a): {
            'entity-a': [('chunk-a', 0.9)],
            'chunk-a': [('entity-a', 0.9)],
        },
        store._owner_key(owner_b): {
            'entity-a': [('chunk-b', 0.8)],
            'chunk-b': [('entity-a', 0.8)],
        },
        store.OWNER_GLOBAL_KEY: {
            'shared-node': [('global-chunk', 0.7)],
        },
    }

    neighbors_a = store.get_neighbors_with_weights('entity-a', owner_id=owner_a)
    neighbors_b = store.get_neighbors_with_weights('entity-a', owner_id=owner_b)
    neighbors_global = store.get_neighbors_with_weights('entity-a', owner_id=None)
    assert neighbors_a == [('chunk-a', 0.9)]
    assert neighbors_b == [('chunk-b', 0.8)]
    assert sorted(neighbors_global) == [('chunk-a', 0.9), ('chunk-b', 0.8)]

    # batch lookup only returns neighbors for supplied owner
    batch = store.get_batch_neighbors_with_weights(['entity-a', 'shared-node'], owner_id=owner_a)
    assert batch['entity-a'] == [('chunk-a', 0.9)]
    assert batch['shared-node'] == []

    batch_global = store.get_batch_neighbors_with_weights(['entity-a', 'shared-node'], owner_id=None)
    assert sorted(batch_global['entity-a']) == [('chunk-a', 0.9), ('chunk-b', 0.8)]
    assert batch_global['shared-node'] == [('global-chunk', 0.7)]


def test_fact_scores_filtered_by_owner_id():
    """The Neo4j retriever should only keep facts belonging to the requested owner."""
    retriever = object.__new__(PrunedHippoRAGNeo4jRetriever)
    docstore = {
        'fact-a': Chunk(id='fact-a', content="('alpha','related','beta')", owner_id="owner-a"),
        'fact-b': Chunk(id='fact-b', content="('alpha','related','beta')", owner_id="owner-b"),
    }
    retriever.graph_store = SimpleNamespace(
        fact_faiss_db=SimpleNamespace(docstore=docstore)
    )

    with patch.object(
        PrunedHippoRAGRetriever,
        '_get_fact_scores_faiss',
        return_value=(np.array([0.9, 0.8], dtype=np.float32), ['fact-a', 'fact-b'])
    ):
        scores, fact_ids = retriever._get_fact_scores_faiss("test", owner_id="owner-a")
        assert np.allclose(scores, np.array([0.9], dtype=np.float32))
        assert fact_ids == ['fact-a']

        scores_all, fact_ids_all = retriever._get_fact_scores_faiss("test", owner_id=None)
        assert np.allclose(scores_all, np.array([0.9, 0.8], dtype=np.float32))
        assert fact_ids_all == ['fact-a', 'fact-b']


def test_retriever_all_owner_mode(monkeypatch):
    """Requests without owner_id should now return empty results."""
    monkeypatch.delenv("ADMIN_OWNER_ID", raising=False)
    retriever = object.__new__(PrunedHippoRAGRetriever)
    retriever.config = SimpleNamespace(
        enable_llm_reranking=False,
        fact_retrieval_top_k=5,
        enable_pruning=False,
        include_chunk_neighbors=False,
        expansion_hops=1,
        damping_factor=0.5,
        passage_node_weight=1.0,
        max_neighbors=5,
        query_aware_multiplier=0.0,
        query_aware_min_k=1,
        query_aware_max_k=5,
    )
    retriever.llm_client = None
    retriever.passage_node_keys = []
    retriever._build_node_mappings = lambda owner_id=None: None
    retriever._get_fact_scores_faiss = lambda query, owner_id=None: (np.array([]), [])
    retriever._dense_passage_retrieval = lambda query, top_k, owner_id=None: ["dense-fallback"]
    assert retriever.retrieve("test query", owner_id=None) == []


def test_admin_owner_detection(monkeypatch):
    """ADMIN_OWNER_ID env var should allow identifying admin requests."""
    admin_id = uuid.uuid4()
    monkeypatch.setenv("ADMIN_OWNER_ID", str(admin_id))
    assert is_admin_owner(admin_id) is True
    assert is_admin_owner(uuid.uuid4()) is False


def test_retrieve_requires_owner_scope():
    retriever = object.__new__(PrunedHippoRAGRetriever)
    result = retriever.retrieve("test", owner_id=None)
    assert result == []


def test_extract_entity_ids_retains_owner_scope(monkeypatch):
    monkeypatch.delenv("ADMIN_OWNER_ID", raising=False)
    retriever = object.__new__(PrunedHippoRAGRetriever)
    retriever.graph_store = SimpleNamespace(node_to_idx={
        compute_mdhash_id("alpha", prefix='entity-', owner_id='owner-a'): 1,
        compute_mdhash_id("alpha", prefix='entity-', owner_id='owner-b'): 2,
        compute_mdhash_id("beta", prefix='entity-', owner_id='owner-a'): 3,
        compute_mdhash_id("beta", prefix='entity-', owner_id='owner-b'): 4,
    })

    facts = [
        ("Alpha", "related", "Beta", "owner-a"),
        ("Alpha", "related", "Beta", "owner-b"),
    ]

    entity_ids = retriever._extract_entity_ids_from_facts(facts)
    assert compute_mdhash_id("alpha", prefix='entity-', owner_id='owner-a') in entity_ids
    assert compute_mdhash_id("alpha", prefix='entity-', owner_id='owner-b') in entity_ids

def test_compute_ppr_push_uses_owner_specific_cache(monkeypatch):
    """Push-based PPR should only see the cache/shards for the requested owner."""
    store = _make_store()
    owner_a = uuid.uuid4()
    owner_b = uuid.uuid4()
    store._cache_loaded = True
    store._graph_cache = {
        store._owner_key(owner_a): {'node-a': [('node-b', 1.0)]},
        store._owner_key(owner_b): {'node-a': [('node-c', 1.0)]},
    }

    captured_adjacencies = []

    def fake_extract(cache, nodes):
        captured_adjacencies.append(cache)
        return cache

    def fake_ppr(**kwargs):
        return {'node-a': 0.5}

    monkeypatch.setattr(
        "encapsulation.database.utils.ppr_push.extract_subgraph_adjacency",
        fake_extract
    )
    monkeypatch.setattr(
        "encapsulation.database.utils.ppr_push.ppr_push",
        fake_ppr
    )

    store.compute_ppr_push({'node-a'}, {'node-a': 1.0}, owner_id=owner_a)
    assert captured_adjacencies[0] == store._graph_cache[store._owner_key(owner_a)]

    store.compute_ppr_push({'node-a'}, {'node-a': 1.0}, owner_id=owner_b)
    assert captured_adjacencies[1] == store._graph_cache[store._owner_key(owner_b)]

    store.compute_ppr_push({'node-a'}, {'node-a': 1.0}, owner_id=None)
    assert captured_adjacencies[2]['node-a'] == [('node-b', 1.0), ('node-c', 1.0)]


def test_convert_to_chunks_restores_owner_type():
    """_convert_to_chunks should respect owner filtering and restore uuid objects."""
    retriever = object.__new__(PrunedHippoRAGNeo4jRetriever)
    owner_id = uuid.uuid4()

    class DummyStore:
        def _restore_owner_id(self, value):
            return None if value == RAW_STORE_CLASS.OWNER_GLOBAL_KEY else value

        def _execute_query(self, query, params=None):
            owner_filter = None
            if params and 'owner_id' in params:
                owner_filter = params['owner_id']

            rows = [
                {
                    'chunk_id': 'c1',
                    'content': 'alpha',
                    'owner_id': str(owner_id),
                    'metadata': '{}'
                },
                {
                    'chunk_id': 'c2',
                    'content': 'beta',
                    'owner_id': RAW_STORE_CLASS.OWNER_GLOBAL_KEY,
                    'metadata': '{}'
                },
            ]

            if owner_filter:
                return [row for row in rows if row['owner_id'] == owner_filter]
            return rows

    retriever.graph_store = DummyStore()

    chunks = retriever._convert_to_chunks(['c1', 'c2'], [0.1, 0.2], owner_id=owner_id)
    assert len(chunks) == 1
    assert chunks[0].id == 'c1'
    assert chunks[0].owner_id == owner_id

    chunks_all = retriever._convert_to_chunks(['c1', 'c2'], [0.1, 0.2], owner_id=None)
    assert len(chunks_all) == 2
