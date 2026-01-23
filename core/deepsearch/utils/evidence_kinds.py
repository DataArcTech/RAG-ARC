"""Evidence kind helpers (primary/derived/diagnostic).

Centralizes constants and common operations so evidence governance does not turn into
scattered string literals across the codebase.
"""
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Literal, Tuple, TypeGuard

from encapsulation.data_model.deepsearch import EvidenceChunk

EvidenceKind = Literal["primary", "derived", "diagnostic"]

EVIDENCE_KIND_PRIMARY: EvidenceKind = "primary"
EVIDENCE_KIND_DERIVED: EvidenceKind = "derived"
EVIDENCE_KIND_DIAGNOSTIC: EvidenceKind = "diagnostic"

ALL_EVIDENCE_KINDS: Tuple[EvidenceKind, ...] = (
    EVIDENCE_KIND_PRIMARY,
    EVIDENCE_KIND_DERIVED,
    EVIDENCE_KIND_DIAGNOSTIC,
)


def _is_evidence_mapping(value: Any) -> TypeGuard[Mapping[str, Any]]:
    return isinstance(value, Mapping)


def coerce_evidence_kind(value: Any, *, default: EvidenceKind = EVIDENCE_KIND_PRIMARY) -> EvidenceKind:
    """Best-effort coercion for backwards-compatible traces."""

    if isinstance(value, str):
        token = value.strip().lower()
        if token in ALL_EVIDENCE_KINDS:
            return token  # type: ignore[return-value]
    return default


def get_evidence_kind(evidence: EvidenceChunk | Mapping[str, Any] | Any) -> EvidenceKind:
    if isinstance(evidence, EvidenceChunk):
        return coerce_evidence_kind(getattr(evidence, "kind", None))
    if _is_evidence_mapping(evidence):
        return coerce_evidence_kind(evidence.get("kind"))
    return EVIDENCE_KIND_PRIMARY


def is_primary_evidence(evidence: EvidenceChunk | Mapping[str, Any] | Any) -> bool:
    return get_evidence_kind(evidence) == EVIDENCE_KIND_PRIMARY


def is_derived_evidence(evidence: EvidenceChunk | Mapping[str, Any] | Any) -> bool:
    return get_evidence_kind(evidence) == EVIDENCE_KIND_DERIVED


def is_diagnostic_evidence(evidence: EvidenceChunk | Mapping[str, Any] | Any) -> bool:
    return get_evidence_kind(evidence) == EVIDENCE_KIND_DIAGNOSTIC


def count_evidences_by_kind(evidences: Iterable[EvidenceChunk | Mapping[str, Any] | Any]) -> Dict[EvidenceKind, int]:
    counts: Dict[EvidenceKind, int] = {
        EVIDENCE_KIND_PRIMARY: 0,
        EVIDENCE_KIND_DERIVED: 0,
        EVIDENCE_KIND_DIAGNOSTIC: 0,
    }
    for ev in evidences or []:
        kind = get_evidence_kind(ev)
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def count_primary_evidences(evidences: Iterable[EvidenceChunk | Mapping[str, Any] | Any]) -> int:
    return count_evidences_by_kind(evidences).get(EVIDENCE_KIND_PRIMARY, 0)


def filter_evidences_by_kind(
    evidences: Iterable[EvidenceChunk],
    *,
    allow: Tuple[EvidenceKind, ...] = (EVIDENCE_KIND_PRIMARY,),
) -> list[EvidenceChunk]:
    allowed = set(allow)
    return [ev for ev in evidences or [] if get_evidence_kind(ev) in allowed]

