import json
from typing import Any, Dict, List

from encapsulation.data_model.schema import Chunk

from config.core.deepsearch.multimodal_evidence_defaults import (
    DEEPSEARCH_VISUAL_MIN_IMAGES,
    DEEPSEARCH_VISUAL_FALLBACK_MAX_IMAGE_CANDIDATES,
    DEEPSEARCH_VISUAL_QUERY_MAX_TOKENS,
    DEEPSEARCH_VISUAL_QUERY_TOKEN_MIN_LEN,
)
from config.output_limits import DEEPSEARCH_MAX_IMAGE_INPUTS
from core.deepsearch.utils.file_scope import FileScope


def merge_unique_chunks(primary: List[Chunk], secondary: List[Chunk]) -> List[Chunk]:
    seen: set[str] = set()
    merged: List[Chunk] = []

    def _key(chunk: Chunk) -> str:
        token = str(getattr(chunk, "id", None) or "").strip()
        if token:
            return token
        return str(getattr(chunk, "content", "") or "").strip()[:240]

    for chunk in primary + secondary:
        key = _key(chunk)
        if not key or key in seen:
            continue
        seen.add(key)
        merged.append(chunk)
    return merged


def search_visual_image_chunks_sync(
    *,
    retriever: Any,
    owner_token: str,
    query: str,
    file_scope: FileScope,
    limit: int,
) -> List[Chunk]:
    """Fallback: use Neo4j to locate candidate image chunks by keyword match."""

    graph_store = getattr(retriever, "graph_store", None)
    if graph_store is None or not hasattr(graph_store, "_execute_query"):
        return []

    owner_key = owner_token
    try:
        owner_key = graph_store._owner_key(owner_token)
    except Exception:
        owner_key = owner_token

    tokens = extract_visual_query_tokens(query)
    if not tokens:
        return []

    max_limit = max(0, int(limit))
    if max_limit <= 0:
        return []

    # Note: HippoRAG stores `c.metadata` as a JSON string (for compatibility); avoid
    # property-path lookups in Cypher and filter metadata in Python instead.
    conditions = [
        "c.content CONTAINS 'images/'",
        "ANY(t IN $tokens WHERE c.content CONTAINS t)",
    ]
    params: Dict[str, Any] = {"owner_id": str(owner_key), "tokens": tokens, "limit": max_limit}

    where = " AND ".join(conditions)
    cypher = f"""
    MATCH (c:Chunk)
    WHERE c.owner_id = $owner_id AND {where}
    RETURN c.chunk_id AS chunk_id, c.content AS content, c.metadata AS metadata
    LIMIT $limit
    """
    try:
        rows = graph_store._execute_query(cypher, params)
    except Exception:
        return []

    results: List[Chunk] = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        chunk_id = str(row.get("chunk_id") or "").strip()
        content = row.get("content")
        metadata = row.get("metadata")
        if not chunk_id or not isinstance(content, str):
            continue
        meta_dict: Dict[str, Any] = {}
        if isinstance(metadata, dict):
            meta_dict = dict(metadata)
        elif isinstance(metadata, str) and metadata.strip():
            try:
                meta_dict = dict(json.loads(metadata))
            except Exception:
                meta_dict = {}

        if file_scope.enabled and file_scope.file_ids:
            src = str(meta_dict.get("source_file_id") or "").strip()
            if src and src not in file_scope.file_ids:
                continue
        # Only keep likely image chunks (MinerU-style image markdown).
        if "images/" not in content:
            continue
        results.append(Chunk(id=chunk_id, content=content, metadata=meta_dict))
    return results


def extract_visual_query_tokens(query: str) -> List[str]:
    import re

    text = str(query or "").strip()
    if not text:
        return []

    min_len = max(1, int(DEEPSEARCH_VISUAL_QUERY_TOKEN_MIN_LEN))
    cjk_re = re.compile(r"[\u4e00-\u9fff]{%d,}" % min_len)
    alnum_re = re.compile(r"[A-Za-z0-9]{%d,}" % min_len)

    candidates: List[str] = []
    candidates.extend(cjk_re.findall(text))
    candidates.extend(alnum_re.findall(text))
    try:
        import jieba  # type: ignore[import-not-found]

        for token in jieba.lcut(text):
            t = str(token or "").strip()
            if len(t) >= min_len:
                candidates.append(t)
    except Exception:
        pass

    seen: set[str] = set()
    unique: List[str] = []
    for token in sorted(candidates, key=len, reverse=True):
        t = str(token or "").strip()
        if not t or t in seen:
            continue
        seen.add(t)
        unique.append(t)
        if len(unique) >= int(DEEPSEARCH_VISUAL_QUERY_MAX_TOKENS):
            break
    return unique


def is_image_chunk(chunk: Chunk) -> bool:
    meta = getattr(chunk, "metadata", None) or {}
    if str(meta.get("semantic_unit_type") or "").strip().lower() == "image":
        return True
    image_urls = meta.get("image_urls")
    if isinstance(image_urls, list) and any(isinstance(u, str) and u.strip() for u in image_urls):
        return True
    content = str(getattr(chunk, "content", "") or "")
    if "![](" in content or "<img" in content.lower():
        return True
    return False


def preserve_visual_evidence(
    chunks: List[Chunk],
    *,
    requested_k: int,
    query: str,
) -> tuple[List[Chunk], Dict[str, Any]]:
    """Ensure image chunks survive top-k truncation when visual_evidence_hint is enabled."""

    if requested_k <= 0 or not chunks:
        return chunks, {"visual_evidence_enabled": True, "visual_evidence_selected": 0}

    max_images = DEEPSEARCH_MAX_IMAGE_INPUTS
    if max_images is None:
        max_images = max(0, requested_k)
    max_images = max(0, min(int(max_images), requested_k))

    min_images = max(0, int(DEEPSEARCH_VISUAL_MIN_IMAGES))
    min_images = min(min_images, max_images)
    if min_images <= 0 or max_images <= 0:
        return chunks, {"visual_evidence_enabled": True, "visual_evidence_selected": 0}

    q = str(query or "").strip()
    selected = list(chunks[:requested_k])
    selected_ids = {str(getattr(c, "id", None) or "") for c in selected if getattr(c, "id", None)}

    def _file_id(chunk: Chunk) -> str:
        meta = getattr(chunk, "metadata", None) or {}
        return str(meta.get("source_file_id") or "").strip()

    preferred_files = {_file_id(c) for c in selected if _file_id(c)}
    current_images = [c for c in selected if is_image_chunk(c)]
    if len(current_images) >= min_images:
        return chunks, {
            "visual_evidence_enabled": True,
            "visual_evidence_selected": len(current_images),
            "visual_evidence_in_topk": len(current_images),
        }

    candidates = [c for c in chunks if is_image_chunk(c)]
    if preferred_files:
        candidates = [c for c in candidates if _file_id(c) in preferred_files]
    candidates = [c for c in candidates if not (getattr(c, "id", None) and str(getattr(c, "id")) in selected_ids)]
    if not candidates:
        return chunks, {
            "visual_evidence_enabled": True,
            "visual_evidence_selected": 0,
            "visual_evidence_in_topk": len(current_images),
        }

    def _candidate_phrases(chunk: Chunk) -> List[str]:
        meta = getattr(chunk, "metadata", None) or {}
        phrases: List[str] = []
        token = str(meta.get("index_text") or "").strip()
        if token:
            phrases.append(token)
        alts = meta.get("image_alts")
        if isinstance(alts, list):
            for alt in alts:
                token = str(alt or "").strip()
                if token:
                    phrases.append(token)
        return phrases

    def _relevance_score(chunk: Chunk) -> int:
        if not q:
            return 0
        score = 0
        for phrase in _candidate_phrases(chunk):
            if len(phrase) < 2:
                continue
            if phrase in q:
                score = max(score, len(phrase))
            if q and q in phrase:
                score = max(score, len(q))
        return score

    need = min_images - len(current_images)
    scored: List[tuple[int, int, Chunk]] = []
    for idx, candidate in enumerate(candidates):
        scored.append((_relevance_score(candidate), idx, candidate))
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)
    picked: List[Chunk] = []
    for score, _, candidate in scored:
        if score <= 0:
            continue
        picked.append(candidate)
        if len(picked) >= need:
            break

    if not picked:
        return chunks, {
            "visual_evidence_enabled": True,
            "visual_evidence_selected": 0,
            "visual_evidence_in_topk": len(current_images),
        }

    for candidate in picked:
        drop_index = None
        for idx in range(len(selected) - 1, -1, -1):
            if not is_image_chunk(selected[idx]):
                drop_index = idx
                break
        if drop_index is None:
            break
        selected.pop(drop_index)
        selected.append(candidate)

    final = selected + list(chunks[requested_k:])
    topk_images = sum(1 for c in final[:requested_k] if is_image_chunk(c))
    return final, {
        "visual_evidence_enabled": True,
        "visual_evidence_selected": len(picked),
        "visual_evidence_in_topk": topk_images,
    }


__all__ = [
    "DEEPSEARCH_VISUAL_FALLBACK_MAX_IMAGE_CANDIDATES",
    "merge_unique_chunks",
    "search_visual_image_chunks_sync",
    "extract_visual_query_tokens",
    "is_image_chunk",
    "preserve_visual_evidence",
]

