def test_batch_generate_embeddings_filters_fact_query_when_incremental():
    """Regression: incremental embedding generation must not scan all RELATES_TO facts.

    If `batch_generate_embeddings(chunk_ids=..., entity_ids=...)` scans the whole graph,
    it can trigger an O(N_graph) save of owner-scoped fact/entity FAISS artifacts and blow up
    memory/IO during ingestion benchmarks.
    """

    from encapsulation.database.graph_db.pruned_hipporag_neo4j_embeddings import _PrunedHippoRAGNeo4jEmbeddingsMixin

    class _Dummy(_PrunedHippoRAGNeo4jEmbeddingsMixin):
        def __init__(self) -> None:
            self._queries: list[tuple[str, object]] = []

            # Minimal attributes referenced by the mixin; they won't be used because we return no rows.
            self.chunk_embeddings = {}
            self.storage_path = "/tmp/graph"
            self.fact_faiss_db = object()
            self.entity_faiss_db = object()

        def _restore_owner_id(self, owner):  # noqa: ANN001
            return owner

        def _execute_query(self, query, params=None):  # noqa: ANN001
            self._queries.append((str(query), params))
            return []

    store = _Dummy()
    store.batch_generate_embeddings(chunk_ids=["c1", "c2"], entity_ids=["e1"])

    # Locate the fact query call.
    fact_calls = [q for q, _p in store._queries if "RELATES_TO" in q and "RETURN r.fact_id" in q]
    assert len(fact_calls) == 1
    fact_query = fact_calls[0]

    # Must have a WHERE clause in incremental mode.
    assert "WHERE" in fact_query
    assert "entity_ids" in fact_query or "$entity_ids" in fact_query
    assert "chunk_ids" in fact_query or "$chunk_ids" in fact_query


def test_batch_generate_embeddings_does_not_scan_all_entities_when_empty_entity_ids_provided():
    """If update_index passes entity_ids=[] (no new entities), entity embedding should be a no-op."""

    from encapsulation.database.graph_db.pruned_hipporag_neo4j_embeddings import _PrunedHippoRAGNeo4jEmbeddingsMixin

    class _Dummy(_PrunedHippoRAGNeo4jEmbeddingsMixin):
        def __init__(self) -> None:
            self._queries: list[tuple[str, object]] = []
            self.chunk_embeddings = {}
            self.storage_path = "/tmp/graph"
            self.fact_faiss_db = object()
            self.entity_faiss_db = object()

        def _restore_owner_id(self, owner):  # noqa: ANN001
            return owner

        def _execute_query(self, query, params=None):  # noqa: ANN001
            self._queries.append((str(query), params))
            return []

    store = _Dummy()
    store.batch_generate_embeddings(chunk_ids=["c1"], entity_ids=[])

    entity_calls = [q for q, _p in store._queries if "MATCH (e:Entity)" in q]
    # In this no-op case, we should not issue a full graph scan query for entities.
    assert not entity_calls
