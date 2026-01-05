import pytest

from core.knowledge_graph.sdf import hs_to_sdf_schema, parse_hs_blocks
from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id, text_processing


def test_parse_hs_blocks_basic() -> None:
    hs = """event: 保险责任裁决
event_id: ev1
description: 对事故是否属于保险责任做出结论。
participants: 生效期校验 ev1.1_P1
Gate: and
Relations: xxxx
attributes: xxxx

subevent: 生效期校验
event_id: ev1.1
description: 判断事故时间是否落在保单有效期内。
participants: xxxx
Gate: xxxx
Relations: xxxx
attributes: xxxx
"""
    events = parse_hs_blocks(hs)
    assert len(events) == 2
    assert events[0].raw_event_id == "ev1"
    assert events[0].children_gate == "and"
    assert events[0].participants and events[0].participants[0].raw_child_event_id == "ev1.1"


def test_hs_to_sdf_schema_stable_ids_and_default_temporal() -> None:
    hs = """event: 保险责任裁决
event_id: ev1
description: 对事故是否属于保险责任做出结论。
participants: 生效期校验 ev1.1_P1
Gate: and
Relations: xxxx
attributes: {"scope":"保险理赔","conditions":["已投保"],"exceptions":[]}

subevent: 生效期校验
event_id: ev1.1
description: 判断事故时间是否落在保单有效期内。
participants: xxxx
Gate: xxxx
Relations: xxxx
attributes: xxxx
"""
    hs_events = parse_hs_blocks(hs)
    sdf = hs_to_sdf_schema(
        hs_events=hs_events,
        owner_id="owner-1",
        doc_namespace="doc-1",
        schema_version="v0",
        default_temporal={"effective_date": "2025-01-01"},
    )
    assert sdf.get("sdfVersion") == "2.2"
    assert sdf.get("doc_namespace") == "doc-1"
    events = sdf.get("events") or []
    assert len(events) == 2

    root = events[0]
    assert root.get("temporal", {}).get("effective_date") == "2025-01-01"
    assert root.get("children_gate") == "and"
    assert root.get("children") and isinstance(root.get("children"), list)

    # ID is derived from doc namespace + normalized name path.
    expected_root_id = compute_mdhash_id(
        "doc-1|{}".format(text_processing("保险责任裁决")),
        prefix="sdf-ev-",
        owner_id="owner-1",
    )
    assert root.get("@id") == expected_root_id

    child_id = root["children"][0]["child"]
    assert child_id != expected_root_id
    assert any(e.get("@id") == child_id for e in events)


def test_hs_to_sdf_schema_preserves_before_relations() -> None:
    hs = """event: 理赔裁决
event_id: ev1
description: 对理赔请求做出裁决。
participants: 生效期校验 ev1.1_P1, 除外条款校验 ev1.2_P1
Gate: and
Relations: ev1.1>ev1.2
attributes: xxxx

subevent: 生效期校验
event_id: ev1.1
description: 校验事故时间是否在保单有效期内。
participants: xxxx
Gate: xxxx
Relations: xxxx
attributes: xxxx

subevent: 除外条款校验
event_id: ev1.2
description: 判断是否触发除外责任。
participants: xxxx
Gate: xxxx
Relations: xxxx
attributes: xxxx
"""
    sdf = hs_to_sdf_schema(hs_events=parse_hs_blocks(hs), owner_id="owner-1", doc_namespace="doc-1")
    rels = sdf.get("relations") or []
    assert len(rels) == 1
    assert rels[0].get("wd_label") == "before"
    assert rels[0].get("relationSubject") and rels[0].get("relationObject")

