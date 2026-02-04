from core.presentation.evidence import _trim_graph_snapshot, _triples_from_subgraph


def test_trim_graph_snapshot_prefers_entities_so_triples_survive() -> None:
    # Regression test:
    # When node_limit is small and chunks are abundant, we must still retain entity nodes,
    # otherwise entity-entity edges (triples) get filtered out entirely.
    snapshot = {
        "chunks": [{"id": f"chunk-{idx}", "ppr_score": 1.0} for idx in range(200)],
        "nodes": [
            {"id": f"entity-{idx}", "name": f"E{idx}", "type": "entity", "ppr_score": 0.5}
            for idx in range(20)
        ],
        "edges": [
            {"source": "E0", "target": "E1", "relation": "relates_to", "weight": 1.0},
            {"source": "E1", "target": "E2", "relation": "mentions", "weight": 0.1},
        ],
        "metadata": {},
    }

    trimmed = _trim_graph_snapshot(snapshot, node_limit=5, edge_limit=50)

    assert trimmed
    assert trimmed.get("nodes")
    # "mentions" edges are not treated as triples; keep at least one non-mentions edge.
    triples = _triples_from_subgraph(trimmed, limit=None)
    assert triples
    assert triples[0]["head"] == "E0"
    assert triples[0]["tail"] == "E1"
