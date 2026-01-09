from core.retrieval.graph_retrieveal.similarity_edges import filter_similarity_neighbors, filter_similarity_neighbor_tuples


def test_similarity_edges_filtered_after_hop_budget():
    neighbors = [
        {"neighbor_id": "e2", "relation_type": "RELATES_TO"},
        {"neighbor_id": "e3", "relation_type": "SIMILAR_TO", "similarity": 0.95},
    ]
    out = filter_similarity_neighbors(
        neighbors,
        hop=1,
        relation_name="SIMILAR_TO",
        max_hops=1,
        min_similarity=0.0,
        max_per_node=20,
    )
    assert [n.get("neighbor_id") for n in out] == ["e2"]


def test_similarity_edges_threshold_and_cap():
    neighbors = [
        {"neighbor_id": "e2", "relation_type": "RELATES_TO"},
        {"neighbor_id": "e3", "relation_type": "SIMILAR_TO", "similarity": 0.95},
        {"neighbor_id": "e4", "relation_type": "SIMILAR_TO", "similarity": 0.7},
        {"neighbor_id": "e5", "relation_type": "SIMILAR_TO", "similarity": 0.9},
    ]
    out = filter_similarity_neighbors(
        neighbors,
        hop=0,
        relation_name="SIMILAR_TO",
        max_hops=1,
        min_similarity=0.8,
        max_per_node=1,
    )
    ids = [n.get("neighbor_id") for n in out]
    assert ids[0] == "e2"
    assert ids[1] in {"e3", "e5"}
    assert len(ids) == 2


def test_similarity_edges_disabled_by_max_per_node_zero():
    neighbors = [
        {"neighbor_id": "e2", "relation_type": "SIMILAR_TO", "similarity": 0.95},
        {"neighbor_id": "e3", "relation_type": "RELATES_TO"},
    ]
    out = filter_similarity_neighbors(
        neighbors,
        hop=0,
        relation_name="SIMILAR_TO",
        max_hops=10,
        min_similarity=0.0,
        max_per_node=0,
    )
    assert [n.get("neighbor_id") for n in out] == ["e3"]


def test_similarity_edge_tuple_variant_filters_by_hop_and_threshold():
    neighbors = [
        ("e2", 10.0, "RELATES_TO"),
        ("e3", 0.95, "SIMILAR_TO"),
        ("e4", 0.7, "SIMILAR_TO"),
    ]
    out = filter_similarity_neighbor_tuples(
        neighbors,
        hop=0,
        relation_name="SIMILAR_TO",
        max_hops=1,
        min_similarity=0.8,
        max_per_node=10,
    )
    assert out == [("e2", 10.0, "RELATES_TO"), ("e3", 0.95, "SIMILAR_TO")]
