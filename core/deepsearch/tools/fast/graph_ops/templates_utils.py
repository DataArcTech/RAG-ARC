"""Shared helpers for graph.ops templates."""
from typing import Any, Iterable, List

from encapsulation.data_model.deepsearch import EvidenceChunk
from config.core.deepsearch.evidence_defaults import EVIDENCE_CLASS_GRAPH_INFERENCE
from core.deepsearch.utils.evidence_ids import derived_chunk_id
from core.deepsearch.utils.evidence_kinds import EVIDENCE_KIND_DERIVED, EvidenceKind
from core.knowledge_graph.schema import normalize_relation_token


def build_derived_evidence(
    *,
    tool_name: str,
    plan_step: str | None,
    label: str,
    content: str,
    provenance: dict[str, Any],
    kind: EvidenceKind = EVIDENCE_KIND_DERIVED,
    evidence_class: str | None = None,
) -> EvidenceChunk:
    if evidence_class:
        provenance.setdefault("evidence_class", evidence_class)
    else:
        provenance.setdefault("evidence_class", EVIDENCE_CLASS_GRAPH_INFERENCE)
    chunk_id = derived_chunk_id(tool_name=tool_name, plan_step=plan_step, label=label, content=content)
    return EvidenceChunk(
        chunk_id=chunk_id,
        source=tool_name,
        content=content,
        kind=kind,
        provenance=provenance,
    )


def normalize_string_list(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    values: List[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text:
            values.append(text)
    return values


def normalize_predicate_sequence(raw: Any) -> List[str]:
    values = normalize_string_list(raw)
    normalized: List[str] = []
    for item in values:
        token = normalize_relation_token(item)
        if token:
            normalized.append(token)
    return normalized


def unique_strings(values: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    output: List[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output
