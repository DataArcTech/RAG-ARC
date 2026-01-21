"""Evidence consistency filter for multi-turn retrieval (domain-agnostic).

Goal: reduce cross-entity / cross-product evidence mixing when the user asks a follow-up
question without repeating the company/product name.

Design constraints:
- No regex-based heuristics.
- Anchors come from upstream rewrite/intent (LLM or other router), not hardcoded lists.
- Conservative fallback: if filtering cannot keep a minimum number of chunks, return
  the original chunks and emit diagnostics so the caller can decide next action.
"""
from dataclasses import dataclass
from typing import Any, Iterable

from core.utils.text_contains import contains_any
from core.utils.anchor_variants import anchor_variants, contains_cjk, dedupe_preserve
from config.rag_intent_routing import (
    rag_anchor_filename_fallback_enabled,
    rag_anchor_filename_fallback_max_file_fraction,
    rag_anchor_filename_fallback_max_len,
    rag_anchor_filename_fallback_max_terms,
    rag_anchor_filename_fallback_min_len,
)


@dataclass(frozen=True, slots=True)
class EvidenceConsistencyInfo:
    enabled: bool
    anchors: tuple[str, ...]
    anchors_variants: tuple[str, ...]
    matched_by_filename: int
    matched_by_content: int
    kept: int
    min_keep: int
    passed: bool


def _chunk_filename(meta: Any) -> str:
    if not isinstance(meta, dict):
        return ""
    # Prefer explicit paths when present.
    for key in ("relative_path", "path", "filepath", "file_path"):
        token = str(meta.get(key) or "").strip()
        if token:
            return token
    return str(meta.get("filename") or "").strip()


def _chunk_file_key(meta: Any) -> str:
    if not isinstance(meta, dict):
        return ""
    for key in ("source_file_id", "sourceFileId", "file_id", "fileId", "document_id", "documentId"):
        token = str(meta.get(key) or "").strip()
        if token:
            return token
    return _chunk_filename(meta)


def _chunk_text(chunk: Any) -> str:
    meta = getattr(chunk, "metadata", None) or {}
    if isinstance(meta, dict):
        prompt_text = meta.get("prompt_text")
        if isinstance(prompt_text, str) and prompt_text.strip():
            return prompt_text
        index_text = meta.get("index_text")
        if isinstance(index_text, str) and index_text.strip():
            return index_text
    content = getattr(chunk, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "")
    return str(content or "")


def _dedupe_preserve(items: Iterable[str]) -> list[str]:
    # Backward-compatible alias for older local call sites; prefer the shared helper.
    return dedupe_preserve(items)


def _cjk_runs(text: str) -> list[str]:
    runs: list[str] = []
    buf: list[str] = []
    for ch in str(text or ""):
        if contains_cjk(ch):
            buf.append(ch)
        else:
            if buf:
                runs.append("".join(buf))
                buf = []
    if buf:
        runs.append("".join(buf))
    return runs


def _derive_filename_terms_from_anchors(anchors: list[str], filenames: list[str]) -> list[str]:
    """
    Derive short, filename-matchable terms from anchors based on the current retrieval window.

    This is a deterministic, domain-agnostic fallback:
    - Only used when we cannot match any filenames by the given anchors.
    - Extract CJK n-grams from anchor CJK runs and keep those that match a small fraction of filenames.
    """
    if not rag_anchor_filename_fallback_enabled():
        return []
    min_len = rag_anchor_filename_fallback_min_len()
    max_len = rag_anchor_filename_fallback_max_len()
    if min_len <= 0 or max_len <= 0 or max_len < min_len:
        return []
    max_terms = rag_anchor_filename_fallback_max_terms()
    if max_terms <= 0:
        return []
    distinct_files = [str(f or "").strip() for f in filenames if str(f or "").strip()]
    # De-dup preserving order.
    seen: set[str] = set()
    uniq_files: list[str] = []
    for f in distinct_files:
        if f in seen:
            continue
        seen.add(f)
        uniq_files.append(f)
    if not uniq_files:
        return []
    max_fraction = rag_anchor_filename_fallback_max_file_fraction()
    max_hits = max(int(len(uniq_files) * max_fraction), 1)

    # Candidate terms and their filename hit counts.
    hits: dict[str, int] = {}
    # Prefer generating terms from the longest CJK runs when available. This avoids picking overly generic
    # short runs (e.g. two-character marketing words) when the anchor also contains a more distinctive run.
    all_runs: list[str] = []
    runs_by_anchor: list[list[str]] = []
    for a in anchors:
        runs = _cjk_runs(a)
        runs_by_anchor.append(runs)
        all_runs.extend(runs)
    max_run_len = max((len(r) for r in all_runs), default=0)
    run_len_threshold = max(min_len, max_run_len - 1) if max_run_len > 0 else min_len

    for runs in runs_by_anchor:
        for run in runs:
            run = str(run or "").strip()
            if not run:
                continue
            if len(run) < run_len_threshold:
                continue
            n = len(run)
            for i in range(n):
                for j in range(i + min_len, min(i + max_len, n) + 1):
                    term = run[i:j]
                    if not term or term in hits:
                        continue
                    count = 0
                    for fn in uniq_files:
                        if term in fn:
                            count += 1
                    if count <= 0:
                        continue
                    hits[term] = count

    # Prefer more selective terms (lower hit count), and longer terms among ties.
    ranked = sorted(hits.items(), key=lambda kv: (kv[1], -len(kv[0]), kv[0]))
    out: list[str] = []
    for term, count in ranked:
        if count > max_hits:
            continue
        out.append(term)
        if len(out) >= max_terms:
            break
    # Expand selected terms with deterministic script variants to match mixed-script corpora.
    expanded: list[str] = list(out)
    try:
        from core.utils.query_variants import generate_query_variants

        for t in out:
            for v in generate_query_variants(t):
                vv = str(v or "").strip()
                if not vv:
                    continue
                if contains_cjk(t) and not contains_cjk(vv):
                    continue
                expanded.append(vv)
    except Exception:
        pass
    return dedupe_preserve(expanded)


def _anchor_variants(anchors: list[str]) -> list[str]:
    return anchor_variants(anchors)


def filter_chunks_by_anchors(
    *,
    chunks: list[Any],
    anchors: list[str] | None,
    min_keep: int,
    enable_filename_match: bool = True,
    enable_content_match: bool = True,
    content_scan_chars: int = 2000,
) -> tuple[list[Any], EvidenceConsistencyInfo]:
    anchors = dedupe_preserve(list(anchors or []))
    if not anchors:
        return (
            chunks,
            EvidenceConsistencyInfo(
                enabled=False,
                anchors=tuple(),
                anchors_variants=tuple(),
                matched_by_filename=0,
                matched_by_content=0,
                kept=len(chunks or []),
                min_keep=int(min_keep),
                passed=True,
            ),
        )

    variants = _anchor_variants(anchors)

    # First pass: identify which files are "on-topic" by anchor matches.
    filename_matched_file_keys: set[str] = set()
    content_matched_file_keys: set[str] = set()
    matched_filename = 0
    matched_content = 0

    filenames_in_window: list[str] = []
    for ch in chunks or []:
        meta = getattr(ch, "metadata", None) or {}
        file_key = _chunk_file_key(meta)
        filename = _chunk_filename(meta)
        if filename:
            filenames_in_window.append(filename)
        if enable_filename_match and filename and contains_any(filename, variants):
            matched_filename += 1
            if file_key:
                filename_matched_file_keys.add(file_key)
            continue
        if enable_content_match:
            text = _chunk_text(ch)
            if content_scan_chars > 0 and len(text) > content_scan_chars:
                text = text[:content_scan_chars]
            if text and contains_any(text, variants):
                matched_content += 1
                if file_key:
                    content_matched_file_keys.add(file_key)
                continue

    # If no filename match happened, try a deterministic fallback to derive shorter filename terms.
    if enable_filename_match and not filename_matched_file_keys:
        fallback_terms = _derive_filename_terms_from_anchors(anchors, filenames_in_window)
        if fallback_terms:
            for ch in chunks or []:
                meta = getattr(ch, "metadata", None) or {}
                file_key = _chunk_file_key(meta)
                filename = _chunk_filename(meta)
                if not file_key or not filename:
                    continue
                if contains_any(filename, fallback_terms):
                    filename_matched_file_keys.add(file_key)
                    matched_filename += 1

    # Second pass: keep ALL chunks from matched files. This avoids dropping relevant clauses that don't
    # repeat the product name inside every chunk.
    def _keep_by_file_keys(file_keys: set[str]) -> list[Any]:
        out: list[Any] = []
        if not file_keys:
            return out
        for ch in chunks or []:
            meta = getattr(ch, "metadata", None) or {}
            if _chunk_file_key(meta) in file_keys:
                out.append(ch)
        return out

    # Prefer filename matches when available (more precise than content matches).
    matched_file_keys = filename_matched_file_keys or content_matched_file_keys
    kept = _keep_by_file_keys(matched_file_keys)

    # If we matched too few chunks, try a deterministic filename fallback to derive shorter terms and expand matches.
    # This addresses cases where anchors are correct but overly descriptive compared to corpus filenames.
    if enable_filename_match and len(kept) < int(min_keep):
        fallback_terms = _derive_filename_terms_from_anchors(anchors, filenames_in_window)
        if fallback_terms:
            for ch in chunks or []:
                meta = getattr(ch, "metadata", None) or {}
                file_key = _chunk_file_key(meta)
                filename = _chunk_filename(meta)
                if not file_key or not filename:
                    continue
                if contains_any(filename, fallback_terms):
                    filename_matched_file_keys.add(file_key)
            matched_file_keys = filename_matched_file_keys or content_matched_file_keys
            kept = _keep_by_file_keys(matched_file_keys)

    # Conservative: if the filter would drop too much, keep the original list and mark failed.
    passed = len(kept) >= int(min_keep)
    if not passed:
        return (
            chunks,
            EvidenceConsistencyInfo(
                enabled=True,
                anchors=tuple(anchors),
                anchors_variants=tuple(variants),
                matched_by_filename=matched_filename,
                matched_by_content=matched_content,
                kept=len(chunks or []),
                min_keep=int(min_keep),
                passed=False,
            ),
        )

    return (
        kept,
        EvidenceConsistencyInfo(
            enabled=True,
            anchors=tuple(anchors),
            anchors_variants=tuple(variants),
            matched_by_filename=matched_filename,
            matched_by_content=matched_content,
            kept=len(kept),
            min_keep=int(min_keep),
            passed=True,
        ),
    )


__all__ = ["EvidenceConsistencyInfo", "filter_chunks_by_anchors"]
