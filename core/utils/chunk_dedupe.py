import hashlib
from typing import Any, Iterable, Mapping

from encapsulation.data_model.schema import Chunk


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    try:
        return int(str(value).strip())
    except Exception:  # noqa: BLE001
        return None


def _first_present(meta: Mapping[str, Any], *keys: str) -> Any:
    for k in keys:
        if k in meta:
            return meta.get(k)
    return None


def _extract_metadata(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        meta = obj.get("metadata")
        return dict(meta) if isinstance(meta, Mapping) else {}
    meta = getattr(obj, "metadata", None)
    return dict(meta) if isinstance(meta, Mapping) else {}


def _extract_id(obj: Any) -> str | None:
    if isinstance(obj, Mapping):
        token = str(obj.get("id") or "").strip()
        return token or None
    token = str(getattr(obj, "id", "") or "").strip()
    return token or None


def _extract_file_id(metadata: Mapping[str, Any]) -> str | None:
    for key in (
        "source_file_id",
        "sourceFileId",
        "file_id",
        "fileId",
        "document_id",
        "documentId",
        "doc_id",
        "docId",
    ):
        token = str(metadata.get(key) or "").strip()
        if token:
            return token

    nested = metadata.get("chunk_metadata") or metadata.get("chunkMetadata")
    if isinstance(nested, Mapping):
        for key in (
            "source_file_id",
            "sourceFileId",
            "file_id",
            "fileId",
            "document_id",
            "documentId",
            "doc_id",
            "docId",
        ):
            token = str(nested.get(key) or "").strip()
            if token:
                return token
    return None


def chunk_dedupe_key(chunk: Any) -> tuple[str, ...]:
    """
    Return a stable, conservative key to dedupe chunk references for UI / citations.

    Why conservative:
    - The retrieval layer can surface duplicate "logical" chunks with different runtime IDs
      (e.g. re-indexed documents producing multiple UUIDs for the same file+chunk_index).
    - We prefer *not* to dedupe purely by content (boilerplate may appear across files).
    """

    meta = _extract_metadata(chunk)
    file_id = _extract_file_id(meta)

    start = _coerce_int(_first_present(meta, "start_idx", "startIdx"))
    end = _coerce_int(_first_present(meta, "end_idx", "endIdx"))
    if file_id and start is not None and end is not None:
        return ("file_range", file_id, str(start), str(end))

    chunk_index = _coerce_int(_first_present(meta, "chunk_index", "chunkIndex"))
    if file_id and chunk_index is not None:
        return ("file_chunk_index", file_id, str(chunk_index))

    anchor_id = str(meta.get("anchor_chunk_id") or "").strip()
    if file_id and anchor_id:
        return ("anchor", file_id, anchor_id)

    meta_chunk_id = str(meta.get("chunk_id") or meta.get("chunkId") or "").strip()
    if meta_chunk_id:
        return ("meta_chunk_id", meta_chunk_id)

    provenance = meta.get("provenance")
    if isinstance(provenance, Mapping):
        url = str(provenance.get("url") or "").strip()
        if url:
            return ("url", url)

    url = str(meta.get("url") or "").strip()
    if url:
        return ("url", url)

    cid = _extract_id(chunk)
    if cid:
        return ("id", cid)

    # Last resort: hash prompt/index/content together with a file-ish label if any.
    label = str(meta.get("filename") or "").strip() or (file_id or "")
    prompt = meta.get("prompt_text")
    index_text = meta.get("index_text")
    content = chunk.get("content") if isinstance(chunk, Mapping) else getattr(chunk, "content", None)
    text = str(prompt or index_text or content or "").strip()
    digest = hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest() if text else ""
    if label and digest:
        return ("text", label, digest)
    if digest:
        return ("text", digest)
    # Fully empty chunk: avoid dropping; keep unique per object identity by not deduping.
    return ("opaque", str(id(chunk)))


def dedupe_chunks(chunks: Iterable[Chunk]) -> list[Chunk]:
    seen: set[tuple[str, ...]] = set()
    out: list[Chunk] = []
    for chunk in chunks or []:
        key = chunk_dedupe_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        out.append(chunk)
    return out
