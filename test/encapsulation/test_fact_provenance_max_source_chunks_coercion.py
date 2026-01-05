from encapsulation.database.graph_db.pruned_hipporag_neo4j_indexing import _coerce_fact_provenance_max_source_chunks


def test_fact_provenance_max_source_chunks_honors_zero() -> None:
    assert _coerce_fact_provenance_max_source_chunks(0, default=50, max_value=1000) == 0


def test_fact_provenance_max_source_chunks_clamps_and_falls_back() -> None:
    assert _coerce_fact_provenance_max_source_chunks(None, default=50, max_value=1000) == 50
    assert _coerce_fact_provenance_max_source_chunks("not-an-int", default=50, max_value=1000) == 50
    assert _coerce_fact_provenance_max_source_chunks(-3, default=50, max_value=1000) == 0
    assert _coerce_fact_provenance_max_source_chunks(5000, default=50, max_value=1000) == 1000

