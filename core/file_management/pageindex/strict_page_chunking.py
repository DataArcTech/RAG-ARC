import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from core.file_management.pageindex.utils import read_json, resolve_content_list_path

logger = logging.getLogger(__name__)


def _candidate_texts(item: Dict[str, Any]) -> List[str]:
    texts: List[str] = []

    def _push(value: Any) -> None:
        token = str(value or "").strip()
        if token:
            texts.append(token)

    raw_text = item.get("text")
    if isinstance(raw_text, str):
        text_level = item.get("text_level")
        if item.get("type") == "text" and isinstance(text_level, int) and text_level > 0:
            level = min(max(text_level, 1), 6)
            _push(("#" * level) + " " + raw_text)
        else:
            _push(raw_text)

    for key in ("image_caption", "image_footnote", "table_caption", "table_footnote"):
        value = item.get(key)
        if isinstance(value, list):
            for entry in value:
                _push(entry)
        elif isinstance(value, str):
            _push(value)

    table_body = item.get("table_body")
    if isinstance(table_body, str):
        _push(table_body)

    # Deduplicate while preserving relative order.
    deduped: List[str] = []
    seen: set[str] = set()
    for text in texts:
        if text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def build_page_markdown_from_content_list(content_list: Sequence[Dict[str, Any]]) -> Dict[int, str]:
    pages: Dict[int, List[str]] = {}
    for item in content_list:
        if not isinstance(item, dict):
            continue
        page_idx = item.get("page_idx")
        if not isinstance(page_idx, int):
            continue
        tokens = _candidate_texts(item)
        if not tokens:
            continue
        pages.setdefault(page_idx, []).extend(tokens)

    output: Dict[int, str] = {}
    for page_idx, lines in pages.items():
        if not lines:
            continue
        output[page_idx] = "\n\n".join(lines).strip()
    return output


def load_page_markdown(
    *,
    md_path: Optional[str],
    output_dir: Optional[str],
) -> Tuple[Optional[Dict[int, str]], Dict[str, Any]]:
    diagnostics: Dict[str, Any] = {}
    content_path = resolve_content_list_path(md_path, output_dir)
    if content_path is None:
        diagnostics["content_list_path"] = None
        diagnostics["reason"] = "content_list_not_found"
        return None, diagnostics

    diagnostics["content_list_path"] = str(content_path)
    try:
        loaded = read_json(Path(content_path))
    except Exception as exc:  # noqa: BLE001
        diagnostics["reason"] = "content_list_read_failed"
        diagnostics["error"] = str(exc)
        return None, diagnostics

    if not isinstance(loaded, list):
        diagnostics["reason"] = "content_list_invalid"
        diagnostics["detail"] = f"expected list, got {type(loaded).__name__}"
        return None, diagnostics

    page_texts = build_page_markdown_from_content_list(loaded)
    diagnostics["pages_found"] = len(page_texts)
    diagnostics["content_items"] = len(loaded)
    if not page_texts:
        diagnostics["reason"] = "content_list_no_page_text"
        return None, diagnostics
    diagnostics["reason"] = "ok"
    return page_texts, diagnostics


def _with_page_suffix(value: str, page_idx: int) -> str:
    suffix = f":p{page_idx}"
    token = str(value or "").strip()
    if not token:
        return token
    if token.endswith(suffix):
        return token
    return f"{token}{suffix}"


def strict_page_chunk(
    *,
    chunker: Any,
    chunk_metadata: Dict[str, Any],
    page_texts: Dict[int, str],
    **kwargs: Any,
) -> List[Dict[str, Any]]:
    chunks: List[Dict[str, Any]] = []
    next_chunk_index = 0

    for page_idx in sorted(page_texts.keys()):
        page_text = str(page_texts.get(page_idx) or "").strip()
        if not page_text:
            continue
        page_meta = dict(chunk_metadata)
        page_meta.update(
            {
                "page_start": int(page_idx),
                "page_end": int(page_idx),
                "page_idx": int(page_idx),
                "page_strict": True,
                "page_source": "content_list",
            }
        )
        produced = chunker.chunk_text(text=page_text, metadata=page_meta, **kwargs)
        if not produced:
            continue

        for raw_chunk in produced:
            if not isinstance(raw_chunk, dict):
                continue
            content = str(raw_chunk.get("content") or "").strip()
            if not content:
                continue
            meta = dict(raw_chunk.get("metadata") or {})
            # Force one chunk == one page.
            meta["page_start"] = int(page_idx)
            meta["page_end"] = int(page_idx)
            meta["page_idx"] = int(page_idx)
            meta["page_strict"] = True
            # Keep a stable, global chunk index across pages.
            meta["chunk_index"] = int(next_chunk_index)
            # Ensure semantic_unit identifiers stay unique across pages.
            unit_id = meta.get("semantic_unit_id")
            if unit_id:
                updated = _with_page_suffix(str(unit_id), page_idx)
                meta["semantic_unit_id"] = updated
                parent_id = meta.get("parent_unit_id")
                if parent_id:
                    meta["parent_unit_id"] = _with_page_suffix(str(parent_id), page_idx)
            raw_chunk["metadata"] = meta
            chunks.append(raw_chunk)
            next_chunk_index += 1

    return chunks


def attach_page_chunking_diagnostics(base: Optional[Dict[str, Any]], extra: Dict[str, Any]) -> Dict[str, Any]:
    payload = dict(base or {})
    bucket = dict(payload.get("strict_page_chunking") or {})
    bucket.update(extra)
    payload["strict_page_chunking"] = bucket
    return payload
