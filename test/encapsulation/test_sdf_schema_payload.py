from encapsulation.database.utils.sdf_schema_payload import build_sdf_schema_payload


def test_build_sdf_schema_payload_basic() -> None:
    sdf = {
        "doc_namespace": "doc-1",
        "events": [
            {
                "@id": "sdf-ev-a",
                "name": "理赔裁决",
                "description": "对理赔请求做出裁决。",
                "children_gate": "and",
                "temporal": {"effective_date": "2025-01-01"},
                "children": [{"child": "sdf-ev-b", "importance": 1.0}],
                "attributes": {"scope": "保险理赔"},
            },
            {
                "@id": "sdf-ev-b",
                "name": "生效期校验",
                "description": "判断事故时间是否在有效期内。",
                "children": [],
                "attributes": {},
            },
        ],
        "relations": [
            {
                "@id": "Relations/before",
                "wd_label": "before",
                "relationSubject": "sdf-ev-b",
                "relationObject": "sdf-ev-a",
            }
        ],
    }
    nodes, sub_edges, before_edges, links = build_sdf_schema_payload(
        sdf=sdf,
        chunk_id="chunk-1",
        db_owner_id="owner-1",
        max_events=10,
        max_relations=10,
        max_source_chunks=50,
    )
    assert len(nodes) == 2
    assert len(sub_edges) == 1
    assert len(before_edges) == 1
    assert len(links) == 2

    node_ids = {n["sdf_event_id"] for n in nodes}
    assert node_ids == {"sdf-ev-a", "sdf-ev-b"}
    assert sub_edges[0]["parent_id"] == "sdf-ev-a"
    assert sub_edges[0]["child_id"] == "sdf-ev-b"
    assert before_edges[0]["subject_id"] == "sdf-ev-b"
    assert before_edges[0]["object_id"] == "sdf-ev-a"


def test_build_sdf_schema_payload_respects_limits() -> None:
    sdf = {
        "doc_namespace": "doc-1",
        "events": [{"@id": f"sdf-ev-{i}", "name": f"e{i}", "children": [], "attributes": {}} for i in range(20)],
        "relations": [{"wd_label": "before", "relationSubject": "a", "relationObject": "b"} for _ in range(50)],
    }
    nodes, sub_edges, before_edges, links = build_sdf_schema_payload(
        sdf=sdf,
        chunk_id="chunk-1",
        db_owner_id="owner-1",
        max_events=5,
        max_relations=7,
        max_source_chunks=0,
    )
    assert len(nodes) == 5
    assert len(before_edges) == 7
    assert all(n.get("source_chunk_ids") == [] for n in nodes)
    assert all(e.get("source_chunk_ids") == [] for e in before_edges)

