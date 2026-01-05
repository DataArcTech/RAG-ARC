"""Hierarchical event schema utilities.

This module defines a minimal, testable data contract for:
- HS (Hierarchical Structure): a compact, line-based intermediate format suitable for LLM extraction.
- SDF (Schema Definition Format): a JSON(-LD)-like structure aligned with SHIELD/MFI examples(https://arxiv.org/abs/2408.05357).
"""
import json
import re
from typing import Any, Iterable, Optional

from pydantic import BaseModel, Field, ValidationError

from encapsulation.database.utils.pruned_hipporag_utils import compute_mdhash_id, text_processing


_IMPORTANCE_RE = re.compile(r"(?P<event_id>[A-Za-z0-9_.-]+)_P(?P<importance>[0-9]+(?:\\.[0-9]+)?)$")
_RELATION_RE = re.compile(r"^(?P<left>[A-Za-z0-9_.-]+)\s*>\s*(?P<right>[A-Za-z0-9_.-]+)$")


class HsParticipantRef(BaseModel):
    raw_child_event_id: str
    importance: float | None = None
    label: str | None = None


class HsRelation(BaseModel):
    raw_subject_event_id: str
    raw_object_event_id: str
    relation_type: str = "before"


class HsEvent(BaseModel):
    raw_event_id: str
    name: str
    description: str
    children_gate: str | None = None
    participants: list[HsParticipantRef] = Field(default_factory=list)
    relations: list[HsRelation] = Field(default_factory=list)
    attributes: dict[str, Any] = Field(default_factory=dict)


class SdfTemporal(BaseModel):
    effective_date: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None


class SdfChild(BaseModel):
    child: str
    importance: float | None = None


class SdfEvent(BaseModel):
    id: str = Field(alias="@id")
    name: str
    description: str | None = None
    participants: list[dict[str, Any]] = Field(default_factory=list)
    children: list[SdfChild] = Field(default_factory=list)
    children_gate: str | None = None
    temporal: SdfTemporal | None = None
    scope: str | None = None
    conditions: list[str] | None = None
    exceptions: list[str] | None = None
    priority: float | str | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True, "extra": "allow"}


class SdfRelation(BaseModel):
    id: str = Field(alias="@id")
    wd_node: str | None = None
    wd_label: str = "before"
    wd_description: str | None = None
    relationSubject: str
    relationObject: str
    attributes: dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True, "extra": "allow"}


class SdfSchema(BaseModel):
    context: list[Any] = Field(default_factory=lambda: ["sdf.s3.jsonld", {"cmu": "https://www.cmu.edu/"}], alias="@context")
    sdfVersion: str = "2.2"
    id: str = Field(alias="@id")
    version: str = "v0"
    events: list[SdfEvent] = Field(default_factory=list)
    relations: list[SdfRelation] = Field(default_factory=list)
    entities: list[dict[str, Any]] = Field(default_factory=list)

    model_config = {"populate_by_name": True, "extra": "allow"}


def parse_hs_blocks(text: str) -> list[HsEvent]:
    """Parse HS blocks emitted by an LLM prompt into structured events.

    This parser is intentionally strict on field keys and tolerant on whitespace.
    """
    raw = str(text or "").strip()
    if not raw:
        return []

    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    events: list[HsEvent] = []

    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        fields: dict[str, str] = {}
        for line in lines:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip()
            if key:
                fields[key] = val

        name = fields.get("event") or fields.get("subevent") or ""
        raw_event_id = fields.get("event_id") or ""
        description = fields.get("description") or ""
        if not (name and raw_event_id and description):
            continue

        participants = _parse_participants(fields.get("participants") or "")
        gate = _parse_gate(fields.get("Gate") or "")
        relations = _parse_relations(fields.get("Relations") or "")
        attributes = _parse_attributes(fields.get("attributes") or "")

        try:
            events.append(
                HsEvent(
                    raw_event_id=str(raw_event_id).strip(),
                    name=str(name).strip(),
                    description=str(description).strip(),
                    children_gate=gate,
                    participants=participants,
                    relations=relations,
                    attributes=attributes,
                )
            )
        except ValidationError:
            continue

    return events


def hs_to_sdf_schema(
    *,
    hs_events: Iterable[HsEvent],
    owner_id: str | None,
    doc_namespace: str | None,
    schema_version: str = "v0",
    default_temporal: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert parsed HS events into an SDF schema dict with stable IDs.

    ID stability policy:
    - IDs are derived from (doc_namespace + hierarchical path of normalized event names).
    - This is deterministic for a given extracted HS structure.
    """
    hs_list = list(hs_events or [])
    if not hs_list:
        return {}

    namespace = str(doc_namespace or "").strip() or "unknown_doc"
    schema_id = compute_mdhash_id(f"{namespace}|sdf", prefix="sdf-", owner_id=owner_id)

    by_raw_id: dict[str, HsEvent] = {e.raw_event_id: e for e in hs_list}

    def parent_raw_id(raw_id: str) -> str | None:
        token = str(raw_id or "").strip()
        if "." not in token:
            return None
        return token.rsplit(".", 1)[0]

    cache_path: dict[str, list[str]] = {}
    cache_id: dict[str, str] = {}

    def path_for(raw_id: str) -> list[str]:
        if raw_id in cache_path:
            return cache_path[raw_id]
        evt = by_raw_id.get(raw_id)
        current_name = text_processing(getattr(evt, "name", "") or "") if evt else ""
        if not current_name:
            current_name = text_processing(raw_id) or raw_id
        parent = parent_raw_id(raw_id)
        if parent and parent in by_raw_id:
            path = path_for(parent) + [current_name]
        else:
            path = [current_name]
        cache_path[raw_id] = path
        return path

    def stable_event_id(raw_id: str) -> str:
        if raw_id in cache_id:
            return cache_id[raw_id]
        path_key = ">".join(path_for(raw_id))
        key = f"{namespace}|{path_key}"
        eid = compute_mdhash_id(key, prefix="sdf-ev-", owner_id=owner_id)
        cache_id[raw_id] = eid
        return eid

    events_out: list[dict[str, Any]] = []
    rels_out: list[dict[str, Any]] = []

    for event in hs_list:
        eid = stable_event_id(event.raw_event_id)
        temporal = None
        attrs = dict(event.attributes or {})
        temporal_raw = attrs.get("temporal")
        if isinstance(temporal_raw, dict):
            temporal = temporal_raw
        elif default_temporal and event.raw_event_id == hs_list[0].raw_event_id:
            temporal = default_temporal

        scope = attrs.get("scope") if isinstance(attrs.get("scope"), str) else None
        conditions = attrs.get("conditions") if isinstance(attrs.get("conditions"), list) else None
        exceptions = attrs.get("exceptions") if isinstance(attrs.get("exceptions"), list) else None
        priority = attrs.get("priority")

        children: list[dict[str, Any]] = []
        for ref in event.participants:
            child_raw = str(ref.raw_child_event_id or "").strip()
            if not child_raw or child_raw not in by_raw_id:
                continue
            children.append({"child": stable_event_id(child_raw), "importance": ref.importance})

        out = {
            "@id": eid,
            "name": event.name,
            "description": event.description,
            "participants": [],
            "children": children,
            "children_gate": event.children_gate,
            "temporal": temporal,
            "scope": scope,
            "conditions": conditions,
            "exceptions": exceptions,
            "priority": priority,
            "attributes": attrs,
        }
        events_out.append(out)

        for rel in event.relations:
            left = str(rel.raw_subject_event_id or "").strip()
            right = str(rel.raw_object_event_id or "").strip()
            if not left or not right:
                continue
            if left not in by_raw_id or right not in by_raw_id:
                continue
            rels_out.append(
                {
                    "@id": "Relations/before",
                    "wd_node": "wd:Q79030196",
                    "wd_label": "before",
                    "wd_description": (
                        "qualifies something (inception or end of a thing, event, or date) "
                        "as happening previously to another thing"
                    ),
                    "relationSubject": stable_event_id(left),
                    "relationObject": stable_event_id(right),
                    "attributes": {},
                }
            )

    schema = {
        "@context": ["sdf.s3.jsonld", {"cmu": "https://www.cmu.edu/"}],
        "sdfVersion": "2.2",
        "@id": schema_id,
        "version": str(schema_version or "v0"),
        "events": events_out,
        "relations": rels_out,
        "entities": [],
        "doc_namespace": namespace,
    }

    validated = SdfSchema.model_validate(schema)
    return validated.model_dump(by_alias=True)


def _parse_gate(raw: str) -> str | None:
    token = str(raw or "").strip().lower()
    if not token or token == "xxxx":
        return None
    if token in {"and", "or", "xor"}:
        return token
    return None


def _parse_participants(raw: str) -> list[HsParticipantRef]:
    text = str(raw or "").strip()
    if not text or text.lower() == "xxxx":
        return []
    out: list[HsParticipantRef] = []
    for chunk in [c.strip() for c in text.split(",") if c.strip()]:
        tokens = chunk.split()
        last = tokens[-1] if tokens else chunk
        match = _IMPORTANCE_RE.match(last)
        if match:
            raw_child_id = match.group("event_id")
            importance = float(match.group("importance"))
            label = " ".join(tokens[:-1]).strip() if len(tokens) > 1 else None
            out.append(HsParticipantRef(raw_child_event_id=raw_child_id, importance=importance, label=label or None))
            continue
        # tolerate "<childId>_P1" without label
        match2 = _IMPORTANCE_RE.match(chunk)
        if match2:
            out.append(
                HsParticipantRef(
                    raw_child_event_id=match2.group("event_id"),
                    importance=float(match2.group("importance")),
                    label=None,
                )
            )
    return out


def _parse_relations(raw: str) -> list[HsRelation]:
    text = str(raw or "").strip()
    if not text or text.lower() in {"xxxx", "xxx"}:
        return []
    out: list[HsRelation] = []
    for chunk in [c.strip() for c in text.split(",") if c.strip()]:
        match = _RELATION_RE.match(chunk.replace(" ", ""))
        if not match:
            continue
        out.append(HsRelation(raw_subject_event_id=match.group("left"), raw_object_event_id=match.group("right")))
    return out


def _parse_attributes(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text or text.lower() == "xxxx":
        return {}
    if text.startswith("{") and text.endswith("}"):
        try:
            val = json.loads(text)
            if isinstance(val, dict):
                return val
        except Exception:
            return {"raw_attributes": text}
    return {"raw_attributes": text}
