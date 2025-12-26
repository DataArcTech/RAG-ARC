"""Citation post-processing for DeepSearch reports.

This module decouples citation/index generation from report writing by scanning
the generated report for inline evidence references (e.g. [ev1]) and producing a
structured evidence index that is consistent with the provided evidence bundle.
"""
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


class CitationAgent:
    """Normalize citations and build an evidence index after report generation."""

    _BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
    _CJK_BRACKET_RE = re.compile(r"【([^【】]+)】")

    def process(
        self,
        *,
        structured_report: Dict[str, Any],
        evidences: Sequence[Dict[str, Any]],
        graph_evidence: Mapping[str, Any] | None = None,
        max_chunk_index_items: int | None = None,
        citation_aliases: Mapping[str, str] | None = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        evidence_lookup = _build_chunk_lookup(evidences)
        known_chunk_ids = set(evidence_lookup)

        citation_hints = structured_report.get("citations") or []
        used_for_lookup, source_lookup = _coerce_citation_hints(citation_hints, citation_aliases=citation_aliases, known_ids=known_chunk_ids)

        ordered_ids, locations = _extract_inline_citations(structured_report, known_chunk_ids, citation_aliases=citation_aliases)
        for ev_id in known_chunk_ids:
            if ev_id in used_for_lookup and ev_id not in locations:
                locations[ev_id] = ["evidence_index"]

        citations: List[Dict[str, Any]] = []
        for ev_id in ordered_ids:
            evidence = evidence_lookup.get(ev_id) or {}
            citations.append(
                {
                    "evidence_id": ev_id,
                    "source_type": "chunk",
                    "source": str(evidence.get("source") or source_lookup.get(ev_id) or "").strip() or None,
                    "used_for": (used_for_lookup.get(ev_id) or "Referenced in report").strip(),
                    "confidence": 1.0,
                    "location_in_report": (locations.get(ev_id) or [""])[0] or None,
                }
            )

        if max_chunk_index_items is not None:
            citations = citations[: max(max_chunk_index_items, 0)]

        evidence_index = _build_evidence_index(
            evidences=evidences,
            citations=citations,
            graph_evidence=graph_evidence,
            max_chunk_items=max_chunk_index_items,
        )

        updated = dict(structured_report)
        updated["citations"] = citations
        updated["evidence_index"] = evidence_index

        audit = {
            "inline_citation_count": len(ordered_ids),
            "citation_count": len(citations),
            "evidence_index_count": len(evidence_index),
        }
        return updated, audit


def _build_chunk_lookup(evidences: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for item in evidences:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        lookup[chunk_id] = item
    return lookup


def _coerce_citation_hints(
    citations: Iterable[Any],
    *,
    citation_aliases: Mapping[str, str] | None,
    known_ids: set[str],
) -> Tuple[Dict[str, str], Dict[str, str]]:
    used_for: Dict[str, str] = {}
    source: Dict[str, str] = {}
    alias_map = {(str(key).strip().lower()): str(value) for key, value in (citation_aliases or {}).items() if str(key).strip()}
    for entry in citations:
        if not isinstance(entry, dict):
            continue
        ev_id = str(entry.get("evidence_id") or "").strip()
        if not ev_id:
            continue
        if ev_id not in known_ids and alias_map:
            mapped = alias_map.get(ev_id.lower())
            if mapped and mapped in known_ids:
                ev_id = mapped
        used = str(entry.get("used_for") or "").strip()
        if used:
            used_for[ev_id] = used
        src = entry.get("source")
        if src is not None and str(src).strip():
            source[ev_id] = str(src).strip()
    return used_for, source


def _extract_inline_citations(
    structured_report: Mapping[str, Any],
    known_ids: set[str],
    *,
    citation_aliases: Mapping[str, str] | None,
) -> Tuple[List[str], Dict[str, List[str]]]:
    ordered: List[str] = []
    seen: set[str] = set()
    locations: Dict[str, List[str]] = {}
    alias_map = {(str(key).strip().lower()): str(value) for key, value in (citation_aliases or {}).items() if str(key).strip()}

    def _resolve(token: str) -> str | None:
        candidate = token.strip()
        if not candidate:
            return None
        if candidate in known_ids:
            return candidate
        if alias_map:
            mapped = alias_map.get(candidate.lower())
            if mapped and mapped in known_ids:
                return mapped
        return None

    def _scan(text: str, location: str) -> None:
        raw_tokens: List[str] = []
        raw_tokens.extend(CitationAgent._BRACKET_RE.findall(text or ""))
        raw_tokens.extend(CitationAgent._CJK_BRACKET_RE.findall(text or ""))
        for raw in raw_tokens:
            candidates = re.split(r"[,\s]+", raw.strip())
            for candidate in candidates:
                token = _resolve(candidate)
                if not token:
                    continue
                locations.setdefault(token, []).append(location)
                if token in seen:
                    continue
                seen.add(token)
                ordered.append(token)

    summary = structured_report.get("summary")
    if isinstance(summary, str) and summary:
        _scan(summary, "summary")

    sections = structured_report.get("sections") or []
    if isinstance(sections, list):
        for idx, section in enumerate(sections, start=1):
            if not isinstance(section, dict):
                continue
            title = str(section.get("title") or "").strip()
            body = str(section.get("body_markdown") or "").strip()
            label = f"section:{idx}" if not title else f"section:{idx}:{title}"
            if body:
                _scan(body, label)

    return ordered, locations


def _build_evidence_index(
    *,
    evidences: Sequence[Dict[str, Any]],
    citations: Sequence[Dict[str, Any]],
    graph_evidence: Mapping[str, Any] | None,
    max_chunk_items: int | None,
) -> List[Dict[str, Any]]:
    citation_map: Dict[str, Dict[str, Any]] = {}
    for entry in citations:
        if not isinstance(entry, dict):
            continue
        ev_id = str(entry.get("evidence_id") or "").strip()
        if not ev_id:
            continue
        citation_map[ev_id] = entry
    cited_ids = set(citation_map)

    index: List[Dict[str, Any]] = []
    chunk_items = list(evidences)
    if max_chunk_items is not None:
        chunk_items = chunk_items[: max(max_chunk_items, 0)]
    for item in chunk_items:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        citation = citation_map.get(chunk_id) or {}
        index.append(
            {
                "evidence_id": chunk_id,
                "source_type": "chunk",
                "source": item.get("source"),
                "score": item.get("score"),
                "used_in_report": chunk_id in cited_ids,
                "used_for": citation.get("used_for") if citation else None,
                "location_in_report": citation.get("location_in_report") if citation else None,
                "confidence": citation.get("confidence") if citation else None,
            }
        )

    ge = graph_evidence or {}
    seeds = list(ge.get("seed_entities") or [])
    for seed in seeds:
        if not seed:
            continue
        index.append(
            {
                "evidence_id": f"entity:{seed}",
                "source_type": "entity",
                "source": "graph",
                "used_in_report": False,
                "preview": str(seed),
            }
        )

    # Triples are intentionally omitted from the public evidence index because they
    # tend to be a subset of the graph chain edges and can create redundant output.

    chain = list(ge.get("graph_chain") or [])
    for idx, edge in enumerate(chain, start=1):
        if not edge:
            continue
        index.append(
            {
                "evidence_id": f"path:{idx}",
                "source_type": "graph_path",
                "source": "graph",
                "used_in_report": False,
                "preview": str(edge),
            }
        )

    return index
