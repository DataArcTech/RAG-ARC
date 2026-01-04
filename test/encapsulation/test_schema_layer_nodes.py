from encapsulation.database.utils.schema_layer_nodes import build_schema_layer_payload


def test_build_schema_layer_payload_merges_by_layer_and_text() -> None:
    nodes, links = build_schema_layer_payload(
        mindmap_nodes=[
            {"level": "1", "content": "[concept] 保险条款"},
            {"level": "1.1", "content": "[process] 批单生效"},
            {"level": "2", "content": "[concept] 保险条款"},
        ],
        chunk_id="chunk-1",
        owner_id="owner-a",
        db_owner_id="owner-a",
        max_nodes=50,
    )
    # Two unique schema nodes: (concept,保险条款) and (process,批单生效)
    assert len(nodes) == 2
    assert len(links) == 3
    layers = sorted({n["layer"] for n in nodes})
    assert layers == ["concept", "process"]


def test_build_schema_layer_payload_respects_max_nodes() -> None:
    nodes, links = build_schema_layer_payload(
        mindmap_nodes=[
            {"level": "1", "content": "[concept] A"},
            {"level": "2", "content": "[concept] B"},
        ],
        chunk_id="chunk-1",
        owner_id="owner-a",
        db_owner_id="owner-a",
        max_nodes=1,
    )
    assert len(nodes) == 1
    assert len(links) == 1

