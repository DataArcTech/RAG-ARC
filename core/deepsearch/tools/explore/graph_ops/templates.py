"""Template registry for graph.ops."""
from dataclasses import dataclass
from typing import Awaitable, Callable, Dict, Iterable

from core.deepsearch.tools.base import ToolResult, ToolRunRequest

from .templates_facts import (
    run_entity_concepts,
    run_expand_terms,
    run_facts_by_type,
    run_schema_nodes,
)
from .templates_reasoning import (
    run_aggregate,
    run_intersection,
    run_rule_check,
    run_set_difference,
)
from .templates_relation_paths import run_relation_path_explore, run_relation_path_ground
from .templates_sdf import run_sdf_children, run_sdf_dependencies
from .templates_temporal import run_latest_truth
from .templates_traversal import run_neighbors, run_path_exists, run_trace_to_root


TemplateHandler = Callable[[object, ToolRunRequest, dict], Awaitable[ToolResult]]


@dataclass(frozen=True)
class GraphOpsTemplate:
    name: str
    description: str
    handler: TemplateHandler


_TEMPLATES: Dict[str, GraphOpsTemplate] = {
    "path_exists": GraphOpsTemplate(
        name="path_exists",
        description="Multi-hop path existence (a→b→c) with predicate filters.",
        handler=run_path_exists,
    ),
    "neighbors": GraphOpsTemplate(
        name="neighbors",
        description="1-hop neighbor lookup with direction/predicate filters.",
        handler=run_neighbors,
    ),
    "trace_to_root": GraphOpsTemplate(
        name="trace_to_root",
        description="Hierarchy trace (leaf→root) for tree-like predicates.",
        handler=run_trace_to_root,
    ),
    "intersection": GraphOpsTemplate(
        name="intersection",
        description="Intersection/shared targets between two entities.",
        handler=run_intersection,
    ),
    "set_difference": GraphOpsTemplate(
        name="set_difference",
        description="Set difference (NOT/exclusion) over entity sets.",
        handler=run_set_difference,
    ),
    "aggregate": GraphOpsTemplate(
        name="aggregate",
        description="COUNT DISTINCT aggregation over relation targets.",
        handler=run_aggregate,
    ),
    "rule_check": GraphOpsTemplate(
        name="rule_check",
        description="AND-rule verification over required edges.",
        handler=run_rule_check,
    ),
    "relation_path_explore": GraphOpsTemplate(
        name="relation_path_explore",
        description="Explore relation-path predicates around a seed entity.",
        handler=run_relation_path_explore,
    ),
    "relation_path_ground": GraphOpsTemplate(
        name="relation_path_ground",
        description="Ground a relation-path predicate sequence into concrete paths.",
        handler=run_relation_path_ground,
    ),
    "facts_by_type": GraphOpsTemplate(
        name="facts_by_type",
        description="Fact lookup restricted by entity type.",
        handler=run_facts_by_type,
    ),
    "expand_terms": GraphOpsTemplate(
        name="expand_terms",
        description="Query expansion via related terms.",
        handler=run_expand_terms,
    ),
    "entity_concepts": GraphOpsTemplate(
        name="entity_concepts",
        description="Canonical/alias resolution for entities.",
        handler=run_entity_concepts,
    ),
    "schema_nodes": GraphOpsTemplate(
        name="schema_nodes",
        description="Mindmap schema node lookup by chunk or term.",
        handler=run_schema_nodes,
    ),
    "latest_truth": GraphOpsTemplate(
        name="latest_truth",
        description="Latest-truth selection using temporal attributes.",
        handler=run_latest_truth,
    ),
    "sdf_children": GraphOpsTemplate(
        name="sdf_children",
        description="SDF subevent lookup (children).",
        handler=run_sdf_children,
    ),
    "sdf_dependencies": GraphOpsTemplate(
        name="sdf_dependencies",
        description="SDF before/after dependencies.",
        handler=run_sdf_dependencies,
    ),
}


def template_registry() -> Dict[str, GraphOpsTemplate]:
    return dict(_TEMPLATES)


def template_names() -> list[str]:
    return sorted(_TEMPLATES.keys())


def template_descriptions() -> Dict[str, str]:
    return {name: tmpl.description for name, tmpl in _TEMPLATES.items()}


def iter_templates() -> Iterable[GraphOpsTemplate]:
    return _TEMPLATES.values()
