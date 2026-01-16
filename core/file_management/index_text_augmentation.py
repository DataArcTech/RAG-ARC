"""Chunk-level index_text augmentation for file management pipeline."""

from __future__ import annotations

from typing import Any, Dict, Optional

from core.utils.index_text_augmentation import prepend_title_prefix


def _extract_filename_from_chunk_dict(chunk: Dict[str, Any]) -> Optional[str]:
    meta = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    src_meta = chunk.get("source_metadata") if isinstance(chunk.get("source_metadata"), dict) else {}
    token = str(meta.get("filename") or src_meta.get("filename") or "").strip()
    return token or None


def augment_chunk_dict_index_text(chunk: Dict[str, Any]) -> None:
    """In-place: ensure `metadata.index_text` includes a filename-derived title prefix."""
    if not isinstance(chunk, dict):
        return
    meta = chunk.get("metadata")
    if not isinstance(meta, dict):
        meta = {}

    filename = _extract_filename_from_chunk_dict(chunk)
    base_text = meta.get("index_text")
    if not isinstance(base_text, str) or not base_text.strip():
        base_text = str(chunk.get("content") or "")

    meta["index_text"] = prepend_title_prefix(text=str(base_text or ""), filename=filename)
    # Expose the extracted title for downstream consumers/debug (does not affect retrieval directly).
    if filename and "document_title" not in meta:
        try:
            from core.utils.index_text_augmentation import extract_title_from_filename

            title = extract_title_from_filename(filename)
            if title:
                meta["document_title"] = title
        except Exception:
            pass

    chunk["metadata"] = meta

