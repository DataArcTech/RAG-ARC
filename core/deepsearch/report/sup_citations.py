"""Legacy conversion helpers for bracket-style citations into HippoRAG-compatible <sup> anchors.

DeepSearch reports now emit <sup>k</sup> directly, but these helpers remain for
backward compatibility with older bracket-style outputs.

- The answer uses contiguous numeric superscripts starting from 1.
- ``sources[key=k]`` maps 1:1 to ``<sup>k</sup>``.
- ``citation_key_map`` provides ``evidence_id -> key`` for debugging/replay.
"""
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from config.output_limits import DEEPSEARCH_SOURCE_MAX_CHARS, DEEPSEARCH_SOURCE_TITLE_MAX_CHARS
from core.presentation.evidence import document_download_url, document_preview_url

_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_CJK_BRACKET_RE = re.compile(r"【([^【】]+)】")
_SUP_RUN_BEFORE_PUNCT_RE = re.compile(r"(?P<sups>(?:<sup>\s*\d{1,4}\s*</sup>\s*)+)(?P<punc>[。\.])")
_PUNCT_SPACE_SUP_RE = re.compile(r"(?P<punc>[。\.])\s+(?P<sup><sup>\s*\d{1,4}\s*</sup>)")
_SPACE_BEFORE_PUNCT_SUP_RE = re.compile(r"\s+(?P<punc>[。\.])(?P<sups>(?:<sup>\s*\d{1,4}\s*</sup>)+)")
_APPENDIX_MARKER = "## Appendix:"
_REFERENCES_HEADING = "## References"

def _infer_ordered_ids_from_text(
    markdown_text: str,
    *,
    known_ids: Iterable[str],
) -> List[str]:
    """Infer citation ordering directly from report text when structured citations are missing."""

    known = {str(token).strip() for token in known_ids if str(token).strip()}
    if not known:
        return []

    ordered: List[str] = []
    seen: set[str] = set()

    def _consider(candidate: str) -> None:
        token = (candidate or "").strip()
        if not token or token not in known:
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


def _apply_outside_fenced_code_blocks(text: str, fn) -> str:  # noqa: ANN001
    """Apply a text transform outside fenced code blocks (``` ... ```)."""
    if not text:
        return ""
    lines = str(text).splitlines(keepends=True)
    out: list[str] = []
    in_fence = False
    for line in lines:
        if line.lstrip().startswith("```"):
            in_fence = not in_fence
            out.append(line)
            continue
        out.append(line if in_fence else fn(line))
    return "".join(out)


def normalize_sup_punctuation(markdown_text: str) -> str:
    """Normalize citations so they appear after sentence-ending punctuation (。 or .)."""
    text = str(markdown_text or "")

    def _normalize(line: str) -> str:
        updated = _SUP_RUN_BEFORE_PUNCT_RE.sub(lambda m: f"{m.group('punc')}{''.join(m.group('sups').split())}", line)
        updated = _PUNCT_SPACE_SUP_RE.sub(lambda m: f"{m.group('punc')}{''.join(m.group('sup').split())}", updated)
        updated = _SPACE_BEFORE_PUNCT_SUP_RE.sub(lambda m: f"{m.group('punc')}{''.join(m.group('sups').split())}", updated)
        return updated

    return _apply_outside_fenced_code_blocks(text, _normalize)


def render_references_markdown(sources: Sequence[Mapping[str, Any]]) -> str:
    if not sources:
        return ""
    lines: list[str] = [_REFERENCES_HEADING]
    for entry in sources:
        if not isinstance(entry, Mapping):
            continue
        key = entry.get("key")
        title = str(entry.get("title") or "source").strip() or "source"
        url = entry.get("file")
        try:
            num = int(key)
        except Exception:  # noqa: BLE001
            continue
        if isinstance(url, str) and url.strip():
            file_id = entry.get("file_id")
            if isinstance(file_id, str) and file_id.strip() and url.strip().startswith("/knowledge/chunk/"):
                download_url = document_download_url(file_id.strip())
                lines.append(f"{num}. [{title}]({url.strip()}) ([file]({download_url}))")
            else:
                lines.append(f"{num}. [{title}]({url.strip()})")
        else:
            lines.append(f"{num}. {title}")
    if len(lines) == 1:
        return ""
    return "\n".join(lines).strip()


def append_references_before_appendix(markdown_text: str, references_markdown: str) -> str:
    if not references_markdown:
        return str(markdown_text or "")
    text = str(markdown_text or "")
    idx = text.find(_APPENDIX_MARKER)
    if idx >= 0:
        prefix = text[:idx].rstrip()
        suffix = text[idx:].lstrip()
        return f"{prefix}\n\n{references_markdown.strip()}\n\n{suffix}"
    return f"{text.rstrip()}\n\n{references_markdown.strip()}\n"


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
) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
    """Convert bracket citations in the markdown into HippoRAG-style <sup> anchors.

    Returns:
      - updated markdown text
      - sources entries (ordered, 1-based numbering; HippoRAG-compatible keys)
      - citation_key_map (evidence_id -> key)
    """

    evidence_lookup = _build_evidence_lookup(evidences)

    citations_ordered = _ordered_evidence_ids(citations)
    for ev_id in citations_ordered:
        evidence_lookup.setdefault(ev_id, {})

    # Prefer the ordering implied by the rendered report text. This prevents emitting sources that
    # are not actually cited (a frequent source of "sources != citations" inconsistencies).
    known_ids = set(evidence_lookup)
    text_ordered = _infer_ordered_ids_from_text(str(markdown_text or ""), known_ids=known_ids) if known_ids else []
    ordered_ids: List[str] = []
    if text_ordered:
        text_set = set(text_ordered)
        ordered_ids.extend([ev_id for ev_id in citations_ordered if ev_id in text_set])
        for ev_id in text_ordered:
            if ev_id not in ordered_ids:
                ordered_ids.append(ev_id)
    else:
        ordered_ids = list(citations_ordered)
    if not ordered_ids:
        return str(markdown_text or ""), [], {}

    id_to_num: Dict[str, int] = {}
    for idx, ev_id in enumerate(ordered_ids, start=1):
        id_to_num[ev_id] = idx

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
            parts.append(f"<sup>{num}</sup>")
        return "".join(parts)

    converted_prefix = _BRACKET_RE.sub(_replace, prefix)
    converted_prefix = _CJK_BRACKET_RE.sub(_replace, converted_prefix)
    updated_text = converted_prefix.rstrip()
    if suffix:
        updated_text = updated_text + "\n\n" + suffix.lstrip()

    sources = build_source_entries(
        ordered_ids=ordered_ids,
        evidence_lookup=evidence_lookup,
        id_to_num=id_to_num,
    )
    citation_key_map = {ev_id: int(id_to_num[ev_id]) for ev_id in ordered_ids if ev_id in id_to_num}
    return updated_text, sources, citation_key_map


def format_answer_with_references(
    markdown_text: str,
    *,
    citations: Sequence[Dict[str, Any]] | None,
    evidences: Sequence[Dict[str, Any]] | None = None,
) -> Tuple[str, List[Dict[str, Any]], Dict[str, int]]:
    converted, sources, citation_key_map = convert_bracket_citations_to_sup(
        markdown_text,
        citations=citations,
        evidences=evidences,
    )
    converted = normalize_sup_punctuation(converted)
    refs = render_references_markdown(sources)
    converted = append_references_before_appendix(converted, refs) if refs else converted
    return converted, sources, citation_key_map


def _truncate_text(text: str, *, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    raw = str(text or "")
    if len(raw) <= max_chars:
        return raw
    return f"{raw[:max_chars].rstrip()}..."


def _sanitize_title(text: str, *, max_chars: int) -> str:
    compact = " ".join(str(text or "").strip().split())
    if not compact:
        return "source"
    if max_chars > 0 and len(compact) > max_chars:
        compact = compact[:max_chars].rstrip()
    return compact


def _extract_url_from_provenance(provenance: Mapping[str, Any]) -> str | None:
    url = provenance.get("url")
    if isinstance(url, str):
        token = url.strip()
        if token.lower().startswith(("http://", "https://")):
            return token
    provider = provenance.get("provider")
    if isinstance(provider, str) and provider.strip() in {"web.tavily", "tavily"}:
        url = provenance.get("url")
        if isinstance(url, str):
            token = url.strip()
            if token.lower().startswith(("http://", "https://")):
                return token
    return None


def _extract_file_meta_from_provenance(provenance: Mapping[str, Any]) -> tuple[str | None, str | None]:
    def _coerce_str(value: Any) -> str | None:
        if isinstance(value, str):
            token = value.strip()
            return token or None
        return None

    def _maybe_apply(meta: Mapping[str, Any], current: tuple[str | None, str | None]) -> tuple[str | None, str | None]:
        file_id, filename = current
        chunk_meta = meta.get("chunk_metadata")
        candidates: List[Mapping[str, Any]] = []
        if isinstance(chunk_meta, Mapping):
            candidates.append(chunk_meta)
        candidates.append(meta)
        for cand in candidates:
            if file_id is None:
                file_id = _coerce_str(cand.get("source_file_id"))
            if filename is None:
                filename = _coerce_str(cand.get("filename"))
        return file_id, filename

    file_id: str | None = None
    filename: str | None = None

    metadata = provenance.get("metadata")
    if isinstance(metadata, Mapping):
        file_id, filename = _maybe_apply(metadata, (file_id, filename))

    filtered = provenance.get("filtered")
    if isinstance(filtered, Mapping):
        filtered_meta = filtered.get("metadata")
        if isinstance(filtered_meta, Mapping):
            file_id, filename = _maybe_apply(filtered_meta, (file_id, filename))

    raw_chunk = provenance.get("raw_chunk")
    if isinstance(raw_chunk, Mapping):
        raw_meta = raw_chunk.get("metadata")
        if isinstance(raw_meta, Mapping):
            file_id, filename = _maybe_apply(raw_meta, (file_id, filename))

    return file_id, filename


def build_source_entries(
    *,
    ordered_ids: Sequence[str],
    evidence_lookup: Mapping[str, Dict[str, Any]],
    id_to_num: Mapping[str, int],
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for ev_id in ordered_ids:
        ev_token = str(ev_id or "").strip()
        if not ev_token:
            continue
        key = id_to_num.get(ev_token)
        if key is None:
            continue
        evidence = evidence_lookup.get(ev_token) or {}
        provenance = evidence.get("provenance") if isinstance(evidence.get("provenance"), Mapping) else {}
        url = _extract_url_from_provenance(provenance) if provenance else None
        file_id, filename = _extract_file_meta_from_provenance(provenance) if provenance else (None, None)

        content = str(evidence.get("content") or "")
        description = _truncate_text(content, max_chars=DEEPSEARCH_SOURCE_MAX_CHARS)

        if url:
            title = _sanitize_title(content.splitlines()[0] if content else url, max_chars=DEEPSEARCH_SOURCE_TITLE_MAX_CHARS)
            file_url = url
        else:
            safe_name = Path(filename).name if filename else ""
            title = _sanitize_title(safe_name or "source", max_chars=DEEPSEARCH_SOURCE_TITLE_MAX_CHARS)
            file_url = document_preview_url(file_id, filename) if file_id else None

        entries.append(
            {
                "key": int(key),
                "chunk_id": ev_token,
                "file_id": file_id,
                "title": title,
                "file": file_url,
                "description": description,
            }
        )
    entries.sort(key=lambda item: int(item.get("key") or 0))
    return entries


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
    "format_answer_with_references",
    "build_source_entries",
    "build_reference_entries",
    "render_reference_list_markdown",
    "normalize_sup_punctuation",
    "render_references_markdown",
    "append_references_before_appendix",
]
