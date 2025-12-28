"""File-scope helpers for DeepSearch evidence alignment.

This module centralizes the "evidence must come from these files" contract so that
tools, adapters, traversal executors, and reporters can apply the same rules.
"""
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping


_QUOTED_PATTERNS: tuple[tuple[str, str], ...] = (
    ("《", "》"),
    ("“", "”"),
    ("‘", "’"),
    ("«", "»"),
    ("「", "」"),
    ("\"", "\""),
    ("'", "'"),
)

_FILE_HINT_EXT_RE = re.compile(r"\.[A-Za-z0-9]{2,8}$")
_PATH_PREFIX_RE = re.compile(r"^(?:~|/|\\./|\\.\\./|[A-Za-z]:\\\\)")


def _looks_like_file_hint(value: str) -> bool:
    token = str(value or "").strip()
    if not token:
        return False
    lowered = token.lower()
    if "not found in evidence" in lowered:
        return False
    if "chunk_id" in lowered or "chunkid" in lowered:
        return False

    has_sep = ("/" in token) or ("\\" in token)
    if has_sep:
        # Treat as a file hint only when it actually resembles a path (prefix) or contains an extension.
        if _PATH_PREFIX_RE.search(token):
            return True
        if _FILE_HINT_EXT_RE.search(token):
            return True
        # Avoid scoping on generic quoted alternatives like "A/B".
        return False
    if _FILE_HINT_EXT_RE.search(token):
        return True
    # Allow short "title-like" tokens without whitespace; avoid scoping on generic quoted phrases.
    if len(token) >= 3 and not re.search(r"\s", token):
        return True
    return False


@dataclass(frozen=True)
class FileScope:
    """Represents optional per-run/per-step file constraints."""

    file_ids: frozenset[str] = frozenset()
    filename_contains: tuple[str, ...] = ()
    source: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.file_ids or self.filename_contains)

    def as_dict(self) -> dict[str, Any]:
        return {
            "file_ids": sorted(self.file_ids),
            "filename_contains": list(self.filename_contains),
            "source": self.source,
        }


def extract_titles_from_question(question: str) -> list[str]:
    """Extract quoted spans from the user question as filename/title hints."""

    q = str(question or "")
    tokens: list[str] = []
    for left, right in _QUOTED_PATTERNS:
        if not left or not right:
            continue
        if left == right:
            pat = re.compile(re.escape(left) + r"([^" + re.escape(left) + r"]{1,120})" + re.escape(right))
        else:
            pat = re.compile(re.escape(left) + r"([^" + re.escape(right) + r"]{1,120})" + re.escape(right))
        for match in pat.finditer(q):
            value = (match.group(1) or "").strip()
            if value and _looks_like_file_hint(value) and value not in tokens:
                tokens.append(value)
    return tokens


def normalize_filename_token(value: Any) -> str:
    """Normalize a filename hint for substring matching."""

    token = str(value or "").strip()
    if not token:
        return ""
    for left, right in _QUOTED_PATTERNS:
        if token.startswith(left) and token.endswith(right) and len(token) >= len(left) + len(right) + 1:
            token = token[len(left) : -len(right)].strip()
            break
    token = token.strip().strip("\"'“”‘’`")
    return token


def _coerce_str_set(values: Any) -> frozenset[str]:
    if values is None:
        return frozenset()
    if isinstance(values, (set, frozenset, tuple, list)):
        items = values
    else:
        items = [values]
    out: set[str] = set()
    for item in items:
        token = str(item or "").strip()
        if token:
            out.add(token)
    return frozenset(out)


def _coerce_filename_contains(values: Any) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (list, tuple, set, frozenset)):
        raw = list(values)
    else:
        raw = [values]
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        token = normalize_filename_token(item)
        if not token:
            continue
        lowered = token.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        out.append(token)
    return tuple(out)


def resolve_file_scope(
    *,
    extra: Mapping[str, Any] | None = None,
    graph_context_metadata: Mapping[str, Any] | None = None,
    question: str | None = None,
) -> FileScope:
    """Resolve FileScope from explicit tool args, graph_context metadata, and question titles."""

    extra_payload = dict(extra or {})
    ctx_payload = dict(graph_context_metadata or {})

    # 1) explicit tool args (highest priority)
    file_ids = (
        _coerce_str_set(extra_payload.get("source_file_ids"))
        | _coerce_str_set(extra_payload.get("source_file_id"))
        | _coerce_str_set(extra_payload.get("file_ids"))
        | _coerce_str_set(extra_payload.get("file_id"))
    )
    filename_contains = _coerce_filename_contains(
        extra_payload.get("filename_contains")
        or extra_payload.get("source_file_names")
        or extra_payload.get("source_file_name")
        or extra_payload.get("filenames")
        or extra_payload.get("filename")
    )
    if file_ids or filename_contains:
        return FileScope(file_ids=file_ids, filename_contains=filename_contains, source="tool_args")

    # 2) graph_context metadata default
    raw_ctx_scope = ctx_payload.get("file_scope")
    if isinstance(raw_ctx_scope, Mapping):
        ctx_file_ids = _coerce_str_set(raw_ctx_scope.get("file_ids") or raw_ctx_scope.get("source_file_ids"))
        ctx_names = _coerce_filename_contains(raw_ctx_scope.get("filename_contains") or raw_ctx_scope.get("source_file_name"))
        if ctx_file_ids or ctx_names:
            return FileScope(file_ids=ctx_file_ids, filename_contains=ctx_names, source="graph_context")

    # 3) infer from question titles when present
    inferred_names: tuple[str, ...] = ()
    if question:
        inferred_names = _coerce_filename_contains(extract_titles_from_question(question))
    if inferred_names:
        return FileScope(file_ids=frozenset(), filename_contains=inferred_names, source="question_titles")

    return FileScope()


def _extract_chunk_file_meta(metadata: Any) -> tuple[str | None, str | None]:
    if not isinstance(metadata, Mapping):
        return None, None

    file_id = None
    for key in ("source_file_id", "file_id", "document_id", "doc_id"):
        value = metadata.get(key)
        if value is not None:
            token = str(value).strip()
            if token:
                file_id = token
                break

    filename = None
    for key in ("source_file_name", "filename", "source_file", "file_path", "path"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            filename = value.strip()
            break
    return file_id, filename


def _has_file_meta_fields(metadata: Any) -> bool:
    if not isinstance(metadata, Mapping):
        return False
    file_id, filename = _extract_chunk_file_meta(metadata)
    return bool(file_id or filename)


def chunk_in_scope(*, chunk_metadata: Any, scope: FileScope) -> bool:
    """Return True when the chunk metadata matches the file scope (or when scope is disabled)."""

    if not scope.enabled:
        return True
    file_id, filename = _extract_chunk_file_meta(chunk_metadata)
    if scope.file_ids:
        return bool(file_id and file_id in scope.file_ids)
    if scope.filename_contains:
        if not filename:
            return False
        lowered = filename.lower()
        return any(token.lower() in lowered for token in scope.filename_contains)
    return True


def evidence_in_scope(*, evidence: Any, scope: FileScope) -> bool:
    """Best-effort scope match for EvidenceChunk-like objects/dicts."""

    if not scope.enabled:
        return True

    provenance = None
    if isinstance(evidence, Mapping):
        provenance = evidence.get("provenance")
    else:
        provenance = getattr(evidence, "provenance", None)

    if not isinstance(provenance, Mapping):
        # Evidence without provenance is treated as scope-neutral (usually tool-generated).
        return True

    # Common: {"metadata": {...}}
    meta = provenance.get("metadata")
    if isinstance(meta, Mapping) and _has_file_meta_fields(meta):
        return chunk_in_scope(chunk_metadata=meta, scope=scope)

    # Common: {"raw_chunk": {"metadata": {...}}}
    raw_chunk = provenance.get("raw_chunk")
    if isinstance(raw_chunk, Mapping):
        raw_meta = raw_chunk.get("metadata")
        if isinstance(raw_meta, Mapping) and _has_file_meta_fields(raw_meta):
            return chunk_in_scope(chunk_metadata=raw_meta, scope=scope)

    # Traversal: provenance.metadata.chunk_metadata
    if isinstance(meta, Mapping):
        chunk_meta = meta.get("chunk_metadata")
        if isinstance(chunk_meta, Mapping) and _has_file_meta_fields(chunk_meta):
            return chunk_in_scope(chunk_metadata=chunk_meta, scope=scope)

    # If no file-identifiable metadata is present, treat as scope-neutral.
    return True


def filter_evidences(evidences: Iterable[Any], *, scope: FileScope) -> tuple[list[Any], dict[str, Any]]:
    """Filter evidence list by scope. Returns (kept, diagnostics)."""

    items = list(evidences or [])
    if not scope.enabled:
        return items, {"file_scope_applied": False}

    kept: list[Any] = []
    dropped = 0
    unknown = 0
    for ev in items:
        prov = ev.get("provenance") if isinstance(ev, Mapping) else getattr(ev, "provenance", None)
        has_file_meta = False
        if isinstance(prov, Mapping):
            meta = prov.get("metadata")
            if isinstance(meta, Mapping) and _has_file_meta_fields(meta):
                has_file_meta = True
            raw_chunk = prov.get("raw_chunk")
            if isinstance(raw_chunk, Mapping):
                raw_meta = raw_chunk.get("metadata")
                if isinstance(raw_meta, Mapping) and _has_file_meta_fields(raw_meta):
                    has_file_meta = True
            if isinstance(meta, Mapping):
                chunk_meta = meta.get("chunk_metadata")
                if isinstance(chunk_meta, Mapping) and _has_file_meta_fields(chunk_meta):
                    has_file_meta = True
        if not has_file_meta:
            unknown += 1
            kept.append(ev)
            continue
        if evidence_in_scope(evidence=ev, scope=scope):
            kept.append(ev)
        else:
            dropped += 1
    return kept, {
        "file_scope_applied": True,
        "file_scope": scope.as_dict(),
        "kept_in_scope": len(kept),
        "dropped_out_of_scope": dropped,
        "scope_neutral_evidence": unknown,
        "input_evidence_count": len(items),
    }
