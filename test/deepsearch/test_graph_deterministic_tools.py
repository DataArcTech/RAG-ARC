"""Regression suite for deterministic graph tools.

The tools under test are Neo4j-Cypher-backed deterministic operators exposed to DeepSearch
via the built-in tool catalog.
"""

import pytest

from core.deepsearch.tools import GraphOpsTool, ToolRunRequest
from core.graph_adapter.base import GraphAccessScope, GraphAdapterCapability, GraphAdapterMetadata
from core.knowledge_graph.schema import schema_from_dict


def _graph_ops_request(*, question: str, plan_step: str, adapter, access_scope, template: str, args: dict) -> ToolRunRequest:
    return ToolRunRequest(
        question=question,
        plan_step=plan_step,
        context_evidences=[],
        adapter=adapter,
        access_scope=access_scope,
        extra={"mode": "template", "template": template, "template_args": args},
    )


class _StubCypherAdapter:
    def __init__(self, rows_by_tool: dict[str, list[dict]], *, direction_sensitive_relations: list[str] | None = None):
        self._rows_by_tool = rows_by_tool
        self._metadata = GraphAdapterMetadata(
            adapter_name="hipporag",
            graph_type="hipporag",
            version="test",
            capabilities=(GraphAdapterCapability(name="cypher_query", modes=("neo4j",)),),
        )
        if direction_sensitive_relations:
            schema = schema_from_dict(
                {
                    "version": "v1",
                    "default_domain": "default",
                    "domains": {"default": {"direction_sensitive_relations": direction_sensitive_relations}},
                }
            )

            class _GraphStore:
                kg_schema = schema

            class _Retriever:
                graph_store = _GraphStore()

            self.retriever = _Retriever()

    async def prepare(self, question: str, *, access_scope=None) -> None:  # noqa: ARG002
        return None

    async def aquery_subgraph(self, query: str, *, channel: str = "graph", access_scope=None, query_options=None):  # noqa: ARG002
        return {"chunks": [], "nodes": [], "edges": [], "metadata": {"adapter": "hipporag"}}

    async def context_filter(self, data, *, filter_type: str = "semantic", access_scope=None):  # noqa: ARG002
        return data

    async def summarize(self, channel: str, data, *, access_scope=None):  # noqa: ARG002
        return "ok"

    async def chain_traverse(self, strategy, *, access_scope=None):  # noqa: ARG002
        return {"strategy": strategy.get("strategy"), "hops": 1, "visited": []}

    def cypher_capable(self) -> bool:
        return True

    async def acypher(self, cypher: str, params=None, *, access_scope=None):  # noqa: ARG002
        text = str(cypher or "")
        if "AS left_fact_ids" in text and "AS right_fact_ids" in text:
            return self._rows_by_tool.get("intersection", [])
        if "set_difference" in text or "WITH u, count(rel) AS hit_count" in text:
            return self._rows_by_tool.get("set_difference", [])
        if "count(DISTINCT COALESCE(n.entity_canonical_key" in text:
            return self._rows_by_tool.get("aggregate", [])
        if "AS head_candidates" in text and "AS tail_candidates" in text and "r.fact_id AS fact_id" in text:
            head = (params or {}).get("head")
            head_key = f"rule_check:{head}"
            if head_key in self._rows_by_tool:
                return self._rows_by_tool.get(head_key, [])
            return self._rows_by_tool.get("rule_check_fail", [])
        if "AS source_candidates" in text and "AS target_candidates" in text and "AS nodes" in text:
            return self._rows_by_tool.get("path_exists", [])
        if "AS leaf_candidates" in text and "AS chain" in text:
            return self._rows_by_tool.get("trace_to_root", [])
        if "relation_path_explore" in text:
            return self._rows_by_tool.get("relation_path_explore", [])
        if "relation_path_ground" in text:
            return self._rows_by_tool.get("relation_path_ground", [])
        if "n.entity_name AS neighbor" in text and "AS candidate_count" in text:
            return self._rows_by_tool.get("neighbors", [])
        if "facts_by_type" in text or "RETURN e.entity_name AS head" in text:
            return self._rows_by_tool.get("facts_by_type", [])
        if "t.entity_name AS term" in text and "candidate_count" in text:
            return self._rows_by_tool.get("expand_terms", [])
        if "latest_truth" in text or "ORDER BY sort_key DESC" in text:
            return self._rows_by_tool.get("latest_truth", [])
        if "SDF_HAS_SUBEVENT" in text and "candidate_count AS candidate_count" in text and "c.sdf_event_id AS child_event_id" in text:
            return self._rows_by_tool.get("sdf_children", [])
        if "before_list AS before" in text and "SDF_BEFORE" in text:
            return self._rows_by_tool.get("sdf_dependencies", [])
        return []

    def metadata(self):
        return self._metadata


@pytest.mark.asyncio
async def test_example_07_intersection_query_ddi() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "intersection": [
                {
                    "target": "cyp3a4",
                    "left_fact_ids": ["fact-1"],
                    "right_fact_ids": ["fact-2"],
                    "left_source_chunk_ids": [["chunk-a"]],
                    "right_source_chunk_ids": [["chunk-b"]],
                }
            ]
        }
    )
    tool = GraphOpsTool()
    req = _graph_ops_request(
        question="ddi?",
        plan_step="p7",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="intersection",
        args={"left": "Zenthorax", "right": "Vira-X", "left_predicates": ["INHIBITS"], "right_predicates": ["METABOLIZED_BY"]},
    )
    result = await tool.run(req)
    assert "intersection" in result.summary.lower()
    assert result.evidences


@pytest.mark.asyncio
async def test_example_11_set_difference_safe_products() -> None:
    adapter = _StubCypherAdapter(rows_by_tool={"set_difference": [{"entity": "可可喜悦"}]})
    tool = GraphOpsTool()
    req = _graph_ops_request(
        question="which products do not contain peanuts?",
        plan_step="p11",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="set_difference",
        args={"universe_type": "Product", "exclude": ["花生"], "predicates": ["CONTAINS", "TRACES_OF"]},
    )
    result = await tool.run(req)
    assert "set_difference" in result.summary.lower()
    assert result.evidences


@pytest.mark.asyncio
async def test_example_12_aggregate_counts_distinct() -> None:
    adapter = _StubCypherAdapter(rows_by_tool={"aggregate": [{"distinct_count": 3, "examples": ["apex", "beta-tech"]}]})
    tool = GraphOpsTool()
    req = _graph_ops_request(
        question="how many suppliers?",
        plan_step="p12",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="aggregate",
        args={"entity": "Project Zeus", "predicate": "HAS_SUPPLIER", "direction": "out"},
    )
    result = await tool.run(req)
    assert "distinct_count=3" in result.summary
    assert result.evidences


@pytest.mark.asyncio
async def test_example_10_rule_check_reports_failure() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "rule_check:84消毒液": [{"fact_id": "fact-1", "source_chunk_ids": ["chunk-a"], "text": "(84,has,次氯酸钠)"}],
            "rule_check:洁厕灵": [],
        }
    )
    tool = GraphOpsTool()
    req = _graph_ops_request(
        question="dangerous mix?",
        plan_step="p10",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="rule_check",
        args={
            "conditions": [
                {"head": "84消毒液", "predicate": "HAS_INGREDIENT", "tail": "次氯酸钠", "direction": "out"},
                {"head": "洁厕灵", "predicate": "HAS_INGREDIENT", "tail": "盐酸", "direction": "out"},
            ]
        },
    )
    result = await tool.run(req)
    assert "failed" in result.summary.lower()
    assert result.diagnostics.get("ok") is False


@pytest.mark.asyncio
async def test_example_10_rule_check_reports_success() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "rule_check:84消毒液": [{"fact_id": "fact-1", "source_chunk_ids": ["chunk-a"], "text": "(84,has,次氯酸钠)"}],
            "rule_check:洁厕灵": [{"fact_id": "fact-2", "source_chunk_ids": ["chunk-b"], "text": "(洁厕灵,has,盐酸)"}],
        }
    )
    tool = GraphOpsTool()
    req = _graph_ops_request(
        question="dangerous mix?",
        plan_step="p10-ok",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="rule_check",
        args={
            "conditions": [
                {"head": "84消毒液", "predicate": "HAS_INGREDIENT", "tail": "次氯酸钠", "direction": "out"},
                {"head": "洁厕灵", "predicate": "HAS_INGREDIENT", "tail": "盐酸", "direction": "out"},
            ]
        },
    )
    result = await tool.run(req)
    assert "ok" in result.summary.lower()
    assert result.diagnostics.get("ok") is True


@pytest.mark.asyncio
async def test_rule_check_fails_when_no_valid_conditions() -> None:
    adapter = _StubCypherAdapter(rows_by_tool={})
    tool = GraphOpsTool()
    req = _graph_ops_request(
        question="noop should fail",
        plan_step="p10-noop",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="rule_check",
        args={"conditions": [{}, {"head": "", "predicate": "", "tail": ""}]},
    )
    result = await tool.run(req)
    assert "failed" in result.summary.lower()
    assert result.diagnostics.get("ok") is False


@pytest.mark.asyncio
async def test_example_01_multi_hop_path_exists() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "path_exists": [
                {
                    "nodes": ["a公司", "b公司", "c公司"],
                    "predicates": ["OWNS", "OWNS"],
                    "fact_ids": ["fact-1", "fact-2"],
                    "source_chunk_ids": [["chunk-1"], ["chunk-2"]],
                }
            ]
        }
    )
    tool = GraphOpsTool()
    req = _graph_ops_request(
        question="A 是否间接控股 C？",
        plan_step="p01",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="path_exists",
        args={"source": "A公司", "target": "C公司", "predicates": ["OWNS"], "direction": "out", "max_hops": 3},
    )
    result = await tool.run(req)
    assert "ok=true" in result.summary.lower()
    assert result.evidences


@pytest.mark.asyncio
async def test_sdf_children_returns_children() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "sdf_children": [
                {
                    "candidate_count": 1,
                    "gate": "and",
                    "child": "生效期校验",
                    "child_event_id": "sdf-ev-b",
                    "importance": 1.0,
                    "source_chunk_ids": ["chunk-a"],
                }
            ]
        }
    )
    tool = GraphOpsTool()
    req = _graph_ops_request(
        question="保险责任裁决流程是什么？",
        plan_step="p-sdf-children",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="sdf_children",
        args={"event": "保险责任裁决", "doc_namespace": "doc-1", "limit": 10},
    )
    result = await tool.run(req)
    assert "children=1" in result.summary
    assert result.evidences
    assert result.evidences[0].provenance.get("children")


@pytest.mark.asyncio
async def test_sdf_dependencies_returns_neighbors() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "sdf_dependencies": [
                {
                    "candidate_count": 1,
                    "before": [{"name": "生效期校验", "event_id": "sdf-ev-b", "source_chunk_ids": ["chunk-a"]}],
                    "after": [{"name": "除外条款校验", "event_id": "sdf-ev-c", "source_chunk_ids": ["chunk-b"]}],
                }
            ]
        }
    )
    tool = GraphOpsTool()
    req = _graph_ops_request(
        question="哪些步骤在除外条款校验之前/之后？",
        plan_step="p-sdf-deps",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="sdf_dependencies",
        args={"event": "除外条款校验", "doc_namespace": "doc-1", "limit": 10},
    )
    result = await tool.run(req)
    assert "before=1" in result.summary
    assert "after=1" in result.summary
    assert result.evidences


@pytest.mark.asyncio
async def test_direction_sensitive_predicate_forces_directed_traversal() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "path_exists": [
                {
                    "nodes": ["a公司", "b公司", "c公司"],
                    "predicates": ["OWNS", "OWNS"],
                    "fact_ids": ["fact-1", "fact-2"],
                    "source_chunk_ids": [["chunk-1"], ["chunk-2"]],
                }
            ]
        },
        direction_sensitive_relations=["OWNS"],
    )
    tool = GraphOpsTool()
    req = _graph_ops_request(
        question="A 是否间接控股 C？",
        plan_step="p01-both",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="path_exists",
        args={"source": "A公司", "target": "C公司", "predicates": ["OWNS"], "direction": "both", "max_hops": 3},
    )
    result = await tool.run(req)
    assert result.evidences


@pytest.mark.asyncio
async def test_relation_path_explore_emits_sequences() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "relation_path_explore": [
                {
                    "candidate_count": 1,
                    "predicate_sequence": ["OWNS", "SUBSIDIARY_OF"],
                    "targets": ["B", "C"],
                    "fact_ids_samples": [["fact-1", "fact-2"]],
                    "source_chunk_ids_samples": [[["chunk-a"], ["chunk-b"]]],
                    "path_count": 2,
                }
            ]
        }
    )
    tool = GraphOpsTool()
    req = _graph_ops_request(
        question="Explore relation paths from A",
        plan_step="p1",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="relation_path_explore",
        args={"entity": "A", "max_hops": 3, "max_sequences": 10},
    )
    result = await tool.run(req)
    assert result.evidences
    provenance = result.evidences[0].provenance
    assert provenance.get("entity") == "a"
    assert provenance.get("relation_paths")


@pytest.mark.asyncio
async def test_relation_path_ground_preserves_ordered_sequence() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "relation_path_ground": [
                {
                    "candidate_count": 1,
                    "nodes": ["A", "B", "C"],
                    "predicates": ["OWNS", "SUBSIDIARY_OF"],
                    "fact_ids": ["fact-1", "fact-2"],
                    "source_chunk_ids": [["chunk-a"], ["chunk-b"]],
                }
            ]
        }
    )
    tool = GraphOpsTool()
    req = _graph_ops_request(
        question="Ground ordered path",
        plan_step="p2",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="relation_path_ground",
        args={"source": "A", "predicate_sequence": ["OWNS", "SUBSIDIARY_OF"], "max_paths": 5},
    )
    result = await tool.run(req)
    assert result.evidences
    provenance = result.evidences[0].provenance
    assert provenance.get("predicate_sequence") == ["OWNS", "SUBSIDIARY_OF"]
    assert provenance.get("frontier_entities") == ["C"]
    assert result.evidences[0].provenance.get("direction") == "out"


@pytest.mark.asyncio
async def test_example_02_entity_disambiguation_filter_by_type() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "facts_by_type": [
                {
                    "head": "捷豹路虎(公司)",
                    "predicate": "HAS_FACT",
                    "tail": "Q3营收增长12%",
                    "fact_id": "fact-1",
                    "source_chunk_ids": [["chunk-1"]],
                }
            ]
        }
    )
    tool = GraphOpsTool()
    req = _graph_ops_request(
        question="捷豹在第三季度表现如何？",
        plan_step="p02",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="facts_by_type",
        args={"entity_type": "Company", "predicates": ["HAS_FACT"], "limit": 10},
    )
    result = await tool.run(req)
    assert "type=Company" in result.summary
    assert result.evidences
    facts = (result.diagnostics or {}).get("facts") or []
    assert any(item.get("tail") == "Q3营收增长12%" for item in facts)


def test_example_03_schema_hallucination_rejects_unknown_relations() -> None:
    schema = schema_from_dict(
        {
            "version": "v1",
            "default_domain": "default",
            "domains": {
                "default": {
                    "allowed_relations": ["PARTNERS_WITH", "SUPPLIES", "DEVELOPS", "ACQUIRED"],
                    "relation_aliases": {"supplies": "SUPPLIES"},
                    "unknown_predicate_policy": "reject",
                }
            },
        }
    )
    domain = schema.for_domain("default")
    assert domain.normalize_predicate("SUPPLIES") == "SUPPLIES"
    assert domain.normalize_predicate("DISCUSSED_WITH") is None


@pytest.mark.asyncio
async def test_example_04_directionality_neighbors_in_out() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "neighbors": [
                {"neighbor": "Novacorp", "predicate": "OWNS", "fact_id": "fact-1", "source_chunk_ids": [["chunk-1"]]},
                {"neighbor": "TinyAI", "predicate": "OWNS", "fact_id": "fact-2", "source_chunk_ids": [["chunk-2"]]},
            ]
        }
    )
    tool = GraphOpsTool()
    # predecessors (in)
    req_in = _graph_ops_request(
        question="谁拥有 Stratos Global？",
        plan_step="p04-in",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="neighbors",
        args={"entity": "Stratos Global", "predicates": ["OWNS"], "direction": "in", "limit": 10},
    )
    result_in = await tool.run(req_in)
    assert "direction=in" in result_in.summary
    # successors (out)
    req_out = _graph_ops_request(
        question="Stratos Global 拥有谁？",
        plan_step="p04-out",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="neighbors",
        args={"entity": "Stratos Global", "predicates": ["OWNS"], "direction": "out", "limit": 10},
    )
    result_out = await tool.run(req_out)
    assert "direction=out" in result_out.summary


@pytest.mark.asyncio
async def test_example_05_fragmented_evidence_list_all_features() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "neighbors": [
                {"neighbor": "钛合金机身", "predicate": "HAS_FEATURE", "fact_id": "f1", "source_chunk_ids": [["c1"]]},
                {"neighbor": "浪涌保护", "predicate": "HAS_FEATURE", "fact_id": "f2", "source_chunk_ids": [["c2"]]},
                {"neighbor": "激光雷达避障", "predicate": "HAS_FEATURE", "fact_id": "f3", "source_chunk_ids": [["c3"]]},
                {"neighbor": "红色急停按钮", "predicate": "HAS_FEATURE", "fact_id": "f4", "source_chunk_ids": [["c4"]]},
                {"neighbor": "ISO-9001 认证", "predicate": "HAS_FEATURE", "fact_id": "f5", "source_chunk_ids": [["c5"]]},
            ]
        }
    )
    tool = GraphOpsTool()
    req = _graph_ops_request(
        question="列出 Model-X 的全部特性",
        plan_step="p05",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="neighbors",
        args={"entity": "Model-X", "predicates": ["HAS_FEATURE"], "direction": "out", "limit": 20},
    )
    result = await tool.run(req)
    neighbors = (result.diagnostics or {}).get("neighbors") or []
    assert len(neighbors) == 5


@pytest.mark.asyncio
async def test_example_06_hierarchy_trace_to_root_returns_chain() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "trace_to_root": [
                {"chain": ["Zeus-X 超级计算机", "核心处理单元", "量子芯片组", "Qubit Lattice"], "hops": 3}
            ]
        }
    )
    tool = GraphOpsTool()
    req = _graph_ops_request(
        question="追溯 Qubit Lattice 的层级路径",
        plan_step="p06",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="trace_to_root",
        args={"leaf": "Qubit Lattice", "predicates": ["CONTAINS"], "max_hops": 6},
    )
    result = await tool.run(req)
    chain = (result.diagnostics or {}).get("chain") or []
    assert chain == ["Zeus-X 超级计算机", "核心处理单元", "量子芯片组", "Qubit Lattice"]


@pytest.mark.asyncio
async def test_example_08_ontology_mapping_query_expansion() -> None:
    adapter = _StubCypherAdapter(rows_by_tool={"expand_terms": [{"term": "Cart-Flow-V2"}]})
    tool = GraphOpsTool()
    req = _graph_ops_request(
        question="结账服务错误率怎么样？",
        plan_step="p08",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="expand_terms",
        args={"concept": "结账服务", "predicates": ["IMPLEMENTS"], "direction": "in", "limit": 10},
    )
    result = await tool.run(req)
    terms = (result.diagnostics or {}).get("terms") or []
    assert "Cart-Flow-V2" in terms


@pytest.mark.asyncio
async def test_example_09_temporal_conflict_latest_truth() -> None:
    adapter = _StubCypherAdapter(
        rows_by_tool={
            "latest_truth": [
                {
                    "value": "0 天/周",
                    "predicate": "HAS_POLICY",
                    "sort_key": "2024",
                    "fact_id": "fact-2024",
                    "source_chunk_ids": [["chunk-2024"]],
                }
            ]
        }
    )
    tool = GraphOpsTool()
    req = _graph_ops_request(
        question="每周可远程办公几天？",
        plan_step="p09",
        adapter=adapter,
        access_scope=GraphAccessScope(scope_id="owner-1"),
        template="latest_truth",
        args={"topic": "远程办公政策", "predicates": ["HAS_POLICY"]},
    )
    result = await tool.run(req)
    assert "value=0 天/周" in result.summary


def test_example_12_entity_canonicalization_configurable() -> None:
    schema = schema_from_dict(
        {
            "version": "v1",
            "default_domain": "default",
            "domains": {
                "default": {
                    "entity_suffixes_to_strip": ["corp", "inc", "ltd", "llc"],
                    "entity_aliases": {"gamma logistics": "gamma logistics"},
                }
            },
        }
    )
    domain = schema.for_domain("default")
    assert domain.canonicalize_entity_name("Apex Corp") == "apex"
    assert domain.canonicalize_entity_name("Apex Inc") == "apex"
