"""Render DeepSearch citations as compact evidence-number tags.

The internal pipeline uses bracket citations (e.g. [chunk_001]) so downstream
components can validate coverage and consistency. This module converts those
bracket citations into user-facing evidence-number tags (e.g. [E1]) and
generates a compact evidence index.
"""
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_CJK_BRACKET_RE = re.compile(r"【([^【】]+)】")

def _infer_ordered_ids_from_text(
    markdown_text: str,
    *,
    evidence_lookup: Mapping[str, Dict[str, Any]],
) -> List[str]:
    """Infer citation ordering directly from report text when structured citations are missing."""

    known_ids = set(evidence_lookup)
    if not known_ids:
        return []

    ordered: List[str] = []
    seen: set[str] = set()

    def _consider(candidate: str) -> None:
        token = (candidate or "").strip()
        if not token or token not in known_ids:
            return
        if token in seen:
            return
        seen.add(token)
        ordered.append(token)

    for raw in _BRACKET_RE.findall(markdown_text or ""):
        for candidate in re.split(r"[,\s]+", str(raw or "").strip()):
            _consider(candidate)
    for raw in _CJK_BRACKET_RE.findall(markdown_text or ""):
        for candidate in re.split(r"[,\s]+", str(raw or "").strip()):
            _consider(candidate)

    return ordered


def _build_evidence_lookup(evidences: Sequence[Dict[str, Any]] | None) -> Dict[str, Dict[str, Any]]:
    lookup: Dict[str, Dict[str, Any]] = {}
    for item in evidences or []:
        if not isinstance(item, dict):
            continue
        chunk_id = str(item.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        lookup[chunk_id] = item
    return lookup


def _ordered_evidence_ids(citations: Sequence[Dict[str, Any]] | None) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for entry in citations or []:
        if not isinstance(entry, dict):
            continue
        ev_id = str(entry.get("evidence_id") or "").strip()
        if not ev_id or ev_id in seen:
            continue
        seen.add(ev_id)
        ordered.append(ev_id)
    return ordered


def convert_bracket_citations_to_sup(
    markdown_text: str,
    *,
    citations: Sequence[Dict[str, Any]] | None,
    evidences: Sequence[Dict[str, Any]] | None = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Convert bracket citations in the markdown into evidence-number tags.

    Returns:
      - updated markdown text
      - reference entries (ordered, 1-based numbering)
    """

    evidence_lookup = _build_evidence_lookup(evidences)
    # Prefer the ordering implied by the rendered report text. This prevents emitting references that
    # are not actually cited (a frequent source of "references != citations" inconsistencies).
    text_ordered = _infer_ordered_ids_from_text(str(markdown_text or ""), evidence_lookup=evidence_lookup) if evidence_lookup else []
    citations_ordered = _ordered_evidence_ids(citations)

    synthesized_citations: List[Dict[str, Any]] | None = None
    ordered_ids: List[str] = []
    if text_ordered:
        text_set = set(text_ordered)
        ordered_ids.extend([ev_id for ev_id in citations_ordered if ev_id in text_set])
        for ev_id in text_ordered:
            if ev_id not in ordered_ids:
                ordered_ids.append(ev_id)
    else:
        ordered_ids = list(citations_ordered)
        if not ordered_ids and evidence_lookup:
            ordered_ids = list(text_ordered)
            if ordered_ids:
                synthesized_citations = [{"evidence_id": ev_id} for ev_id in ordered_ids]
    if not ordered_ids:
        return str(markdown_text or ""), []

    id_to_num: Dict[str, int] = {}
    for idx, ev_id in enumerate(ordered_ids, start=1):
        id_to_num[ev_id] = idx

    active_citations: Sequence[Dict[str, Any]] = citations or []
    if synthesized_citations is not None:
        active_citations = synthesized_citations
    else:
        ordered_set = set(ordered_ids)
        active_citations = [cit for cit in active_citations if isinstance(cit, dict) and str(cit.get("evidence_id") or "").strip() in ordered_set]

    text = str(markdown_text or "")
    appendix_marker = "## Appendix:"
    appendix_idx = text.find(appendix_marker)
    if appendix_idx >= 0:
        prefix = text[:appendix_idx].rstrip()
        suffix = text[appendix_idx:]
    else:
        prefix = text
        suffix = ""

    def _replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        tokens = [tok for tok in re.split(r"[,\s]+", inner.strip()) if tok]
        if not tokens:
            return match.group(0)
        resolved_nums: List[int] = []
        unresolved = False
        for token in tokens:
            ev_id = token.strip()
            if not ev_id:
                continue
            num = id_to_num.get(ev_id)
            if num is None:
                unresolved = True
                break
            resolved_nums.append(int(num))
        if unresolved or not resolved_nums:
            return match.group(0)
        seen_nums: set[int] = set()
        parts: List[str] = []
        for num in resolved_nums:
            if num in seen_nums:
                continue
            seen_nums.add(num)
            parts.append(f"E{num}")
        return "[" + ",".join(parts) + "]"

    converted_prefix = _BRACKET_RE.sub(_replace, prefix)
    converted_prefix = _CJK_BRACKET_RE.sub(_replace, converted_prefix)
    updated_text = converted_prefix.rstrip()
    if suffix:
        updated_text = updated_text + "\n\n" + suffix.lstrip()

    refs = build_reference_entries(
        ordered_ids=ordered_ids,
        citations=list(active_citations),
        evidence_lookup=evidence_lookup,
        id_to_num=id_to_num,
    )
    references_markdown = render_reference_list_markdown(refs)
    if references_markdown:
        updated_text = (updated_text.rstrip() + "\n\n" + references_markdown.strip() + "\n").rstrip() + "\n"

    return updated_text, refs


def build_reference_entries(
    *,
    ordered_ids: Sequence[str],
    citations: Sequence[Dict[str, Any]],
    evidence_lookup: Mapping[str, Dict[str, Any]],
    id_to_num: Mapping[str, int],
) -> List[Dict[str, Any]]:
    citation_by_id: Dict[str, Dict[str, Any]] = {}
    for citation in citations:
        if not isinstance(citation, dict):
            continue
        ev_id = str(citation.get("evidence_id") or "").strip()
        if not ev_id or ev_id in citation_by_id:
            continue
        citation_by_id[ev_id] = citation

    entries: List[Dict[str, Any]] = []
    for ev_id in ordered_ids:
        ev_token = str(ev_id or "").strip()
        if not ev_token:
            continue
        num = id_to_num.get(ev_token)
        if num is None:
            continue
        citation = citation_by_id.get(ev_token) or {}
        evidence = evidence_lookup.get(ev_token) or {}
        entries.append(
            {
                "n": int(num),
                "evidence_id": ev_token,
                "source": citation.get("source") or evidence.get("source"),
                "source_type": citation.get("source_type") or "chunk",
                "score": evidence.get("score"),
                "provenance": evidence.get("provenance") if isinstance(evidence.get("provenance"), dict) else None,
            }
        )
    entries.sort(key=lambda item: int(item.get("n") or 0))
    return entries


def render_reference_list_markdown(references: Sequence[Dict[str, Any]]) -> str:
    if not references:
        return ""
    lines: List[str] = ["## Evidence Index"]

    def _apply_chunk_meta(details: Dict[str, Any], meta: Mapping[str, Any]) -> None:
        # Do not include filenames in public evidence indices (privacy / prompt hygiene).
        source_file_id = meta.get("source_file_id")
        if isinstance(source_file_id, str) and source_file_id.strip():
            details.setdefault("source_file_id", source_file_id.strip())
        for key in ("chunk_index", "start_idx", "end_idx"):
            value = meta.get(key)
            if isinstance(value, int):
                details.setdefault(key, value)

    for ref in references:
        if not isinstance(ref, dict):
            continue
        num = ref.get("n")
        ev_id = str(ref.get("evidence_id") or "").strip()
        source = str(ref.get("source") or "").strip()
        # Keep this minimal: frontends can fetch full evidence/provenance from the payload,
        # while the answer should only show what's needed to verify the citation.
        details: Dict[str, Any] = {"evidence_id": ev_id}
        if source:
            details["source"] = source
        provenance = ref.get("provenance")
        if isinstance(provenance, Mapping):
            meta = provenance.get("metadata")
            if isinstance(meta, Mapping):
                _apply_chunk_meta(details, meta)

                chunk_meta = meta.get("chunk_metadata")
                if isinstance(chunk_meta, Mapping):
                    _apply_chunk_meta(details, chunk_meta)

            raw_chunk = provenance.get("raw_chunk")
            if isinstance(raw_chunk, Mapping):
                raw_meta = raw_chunk.get("metadata")
                if isinstance(raw_meta, Mapping):
                    _apply_chunk_meta(details, raw_meta)
        rendered = json.dumps(details, ensure_ascii=False, separators=(",", ":"), default=str)
        lines.append(f"E{int(num)}. {rendered}")
    return "\n".join(lines).strip()


__all__ = [
    "convert_bracket_citations_to_sup",
    "build_reference_entries",
    "render_reference_list_markdown",
]
