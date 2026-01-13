"""Helpers to assemble chunk, triple, and seed-entity evidence payloads."""
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence

from encapsulation.data_model.schema import Chunk

from config.output_limits import (
    CHAT_TOP_CHUNKS,
    CHAT_TOP_SEED_ENTITIES,
    CHAT_TOP_TRIPLES,
    CHAT_GRAPH_EDGE_LIMIT,
    CHAT_GRAPH_NODE_LIMIT,
    DEEPSEARCH_GRAPH_EDGE_LIMIT,
    DEEPSEARCH_GRAPH_NODE_LIMIT,
    DEEPSEARCH_TOP_CHUNKS,
    DEEPSEARCH_TOP_SEED_ENTITIES,
    DEEPSEARCH_TOP_TRIPLES,
    ENABLE_ALL_EVIDENCE,
)
from core.presentation.graph_chain import build_graph_chain
from encapsulation.graph_export import export_subgraph_snapshot


ChunkEntry = Dict[str, Any]
TripleEntry = Dict[str, Any]

_MD_IMAGE_URL_RE = re.compile(r"(!\[[^\]]*]\()([^)\s]+)([^)]*\))")
_HTML_IMG_RE = re.compile(
    r"(<img\b[^>]*\bsrc=)(?:\"([^\"]+)\"|'([^']+)'|([^\s>]+))",
    re.IGNORECASE,
)


def _document_download_url(file_id: str) -> str:
    return f"/knowledge/{file_id}/download"


def document_download_url(file_id: str) -> str:
    """Public helper for building the knowledge download URL."""
    return _document_download_url(file_id)


def _mineru_asset_url(file_id: str, rel_path: str) -> str:
    token = str(rel_path or "").lstrip("/").lstrip("\\")
    return f"/knowledge/{file_id}/mineru-assets/{token}"


def _is_relative_local_path(url: str) -> bool:
    token = str(url or "").strip()
    if not token:
        return False
    lower = token.lower()
    if lower.startswith(("http://", "https://", "data:", "file:")):
        return False
    if token.startswith("/"):
        return False
    if ".." in token.split("/"):
        return False
    return True


def _rewrite_image_urls_in_markdown(markdown: str, *, file_id: str | None) -> str:
    if not file_id:
        return markdown
    text = str(markdown or "")

    def _md_repl(match: re.Match[str]) -> str:
        url = match.group(2)
        if _is_relative_local_path(url):
            resolved = _mineru_asset_url(file_id, url)
            return f"{match.group(1)}{resolved}{match.group(3)}"
        return match.group(0)

    def _html_repl(match: re.Match[str]) -> str:
        src = match.group(2) or match.group(3) or match.group(4) or ""
        if _is_relative_local_path(src):
            resolved = _mineru_asset_url(file_id, src)
            return f'{match.group(1)}"{resolved}"'
        return match.group(0)

    text = _MD_IMAGE_URL_RE.sub(_md_repl, text)
    text = _HTML_IMG_RE.sub(_html_repl, text)
    return text


def _rewrite_chunk_metadata_urls(metadata: Dict[str, Any], *, file_id: str | None) -> Dict[str, Any]:
    if not file_id:
        return metadata
    updated = dict(metadata)

    image_urls = updated.get("image_urls")
    if isinstance(image_urls, list):
        rel_urls = [str(u) for u in image_urls if isinstance(u, str)]
        updated["image_urls_rel"] = rel_urls
        updated["image_urls"] = [
            _mineru_asset_url(file_id, u) if _is_relative_local_path(u) else u
            for u in rel_urls
        ]
    return updated


def _serialize_chunks(chunks: Sequence[Chunk], limit: Optional[int]) -> List[ChunkEntry]:
    subset = list(chunks[:limit]) if limit else list(chunks)
    payload: List[ChunkEntry] = []
    for chunk in subset:
        metadata = getattr(chunk, "metadata", {}) or {}
        file_id = str(metadata.get("source_file_id") or "").strip() or None
        safe_meta = _rewrite_chunk_metadata_urls(dict(metadata), file_id=file_id)
        chunk_text = metadata.get("prompt_text") or metadata.get("index_text")
        if not isinstance(chunk_text, str) or not chunk_text.strip():
            chunk_text = chunk.content or ""
        content = _rewrite_image_urls_in_markdown(chunk_text, file_id=file_id)
        payload.append(
            {
                "id": getattr(chunk, "id", None),
                "owner_id": getattr(chunk, "owner_id", None),
                "file_id": file_id,
                "document_url": _document_download_url(file_id) if file_id else None,
                "content": content,
                "score": metadata.get("score"),
                "metadata": safe_meta,
            }
        )
    return payload


def serialize_chunks(chunks: Sequence[Chunk], limit: Optional[int] = None) -> List[ChunkEntry]:
    """Public helper to convert chunk objects into serializable dictionaries."""

    return _serialize_chunks(chunks, limit)


def _seed_entities_from_subgraph(subgraph: Optional[Dict[str, Any]], limit: Optional[int]) -> List[str]:
    if not subgraph:
        return []
    if limit is not None and limit <= 0:
        return []
    seen: List[str] = []
    for node in subgraph.get("nodes") or []:
        if node.get("type") != "entity":
            continue
        if not node.get("is_seed"):
            continue
        name = str(node.get("name") or node.get("id") or "").strip()
        if not name:
            continue
        if name not in seen:
            seen.append(name)
        if limit is not None and len(seen) >= limit:
            break
    return seen[:limit] if limit is not None else seen


def _triples_from_subgraph(subgraph: Optional[Dict[str, Any]], limit: Optional[int]) -> List[TripleEntry]:
    if not subgraph:
        return []
    if limit is not None and limit <= 0:
        return []
    triples: List[TripleEntry] = []
    for edge in subgraph.get("edges") or []:
        relation = edge.get("relation")
        if not relation or relation == "mentions":
            continue
        source = str(edge.get("source") or "").strip()
        target = str(edge.get("target") or "").strip()
        if not source or not target:
            continue
        triples.append(
            {
                "head": source,
                "relation": relation,
                "tail": target,
                "weight": edge.get("weight"),
            }
        )
        if limit is not None and len(triples) >= limit:
            break
    return triples[:limit] if limit is not None else triples


def _graph_stats(subgraph: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not subgraph:
        return {}
    entity_nodes = subgraph.get("nodes") or []
    chunk_nodes = subgraph.get("chunks") or []
    edges = subgraph.get("edges") or []
    metadata = subgraph.get("metadata") or {}
    return {
        "nodes": len(entity_nodes) + len(chunk_nodes),
        "edges": len(edges),
        "categories": metadata.get("categories") or [],
    }


def _trim_graph_snapshot(
    snapshot: Optional[Dict[str, Any]],
    node_limit: Optional[int],
    edge_limit: Optional[int],
) -> Optional[Dict[str, Any]]:
    if not snapshot:
        return None
    if ENABLE_ALL_EVIDENCE:
        return snapshot

    chunk_nodes = list(snapshot.get("chunks") or [])
    entity_nodes = list(snapshot.get("nodes") or [])
    edges = list(snapshot.get("edges") or [])
    metadata = snapshot.get("metadata") or {}

    if node_limit is not None and node_limit <= 0:
        chunk_nodes = []
        entity_nodes = []
    elif node_limit is not None:

        def _rank(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return sorted(items, key=lambda entry: entry.get("ppr_score") or 0.0, reverse=True)

        chunk_ranked = _rank(chunk_nodes)
        chunk_nodes = chunk_ranked[: min(len(chunk_ranked), node_limit)]
        remaining = max(node_limit - len(chunk_nodes), 0)
        entity_ranked = _rank(entity_nodes)
        entity_nodes = entity_ranked[: min(len(entity_ranked), remaining)] if remaining else []

    def _id_tokens(entries: List[Dict[str, Any]]) -> set[str]:
        tokens: set[str] = set()
        for entry in entries:
            identifier = str(entry.get("id") or "").strip()
            if identifier:
                tokens.add(identifier)
            name = str(entry.get("name") or "").strip()
            if name:
                tokens.add(name)
        return tokens

    allowed_tokens = _id_tokens(chunk_nodes) | _id_tokens(entity_nodes)

    if edge_limit is not None and edge_limit <= 0:
        edges = []
    elif edge_limit is not None and allowed_tokens:
        filtered_edges = [
            edge
            for edge in edges
            if str(edge.get("source") or "").strip() in allowed_tokens
            and str(edge.get("target") or "").strip() in allowed_tokens
        ]
        filtered_edges.sort(key=lambda entry: entry.get("weight") or 0.0, reverse=True)
        edges = filtered_edges[:edge_limit]

    return {
        "chunks": chunk_nodes,
        "nodes": entity_nodes,
        "edges": edges,
        "metadata": metadata,
    }


def _ensure_subgraph(
    subgraph_data: Optional[Dict[str, Any]],
    subgraph_info: Optional[Dict[str, Any]],
    *,
    node_limit: Optional[int],
    edge_limit: Optional[int],
    graph_store: Any | None = None,
) -> Optional[Dict[str, Any]]:
    if subgraph_data:
        return _trim_graph_snapshot(subgraph_data, node_limit, edge_limit)
    if subgraph_info:
        exported = export_subgraph_snapshot(subgraph_info, graph_store=graph_store)
        return _trim_graph_snapshot(exported, node_limit, edge_limit)
    return None


def build_chat_evidence(
    chunks: Sequence[Chunk],
    *,
    subgraph_data: Optional[Dict[str, Any]] = None,
    subgraph_info: Optional[Dict[str, Any]] = None,
    max_chunks: Optional[int] = CHAT_TOP_CHUNKS,
    graph_store: Any | None = None,
) -> Dict[str, Any]:
    """Assemble chunk/seed/triple information for chat responses."""

    graph_snapshot = _ensure_subgraph(
        subgraph_data,
        subgraph_info,
        node_limit=CHAT_GRAPH_NODE_LIMIT,
        edge_limit=CHAT_GRAPH_EDGE_LIMIT,
        graph_store=graph_store,
    )
    evidence = {
        "chunks": _serialize_chunks(chunks, max_chunks),
        "seed_entities": _seed_entities_from_subgraph(graph_snapshot, CHAT_TOP_SEED_ENTITIES),
        "triples": _triples_from_subgraph(graph_snapshot, CHAT_TOP_TRIPLES),
        "graph": graph_snapshot or {},
        "graph_stats": _graph_stats(graph_snapshot),
    }
    return evidence


def _first_subgraph_info_from_reasoning(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    reasoning = payload.get("reasoning") or {}
    for evidence in reasoning.get("evidences") or []:
        provenance = evidence.get("provenance") or {}
        metadata = provenance.get("metadata") or {}
        if isinstance(metadata, dict):
            subgraph = metadata.get("_subgraph_info") or metadata.get("subgraph_info")
            if isinstance(subgraph, dict) and subgraph:
                return subgraph
        subgraph = provenance.get("_subgraph_info") or provenance.get("subgraph_info")
        if isinstance(subgraph, dict) and subgraph:
            return subgraph
        filtered = provenance.get("filtered")
        if isinstance(filtered, dict):
            filtered_meta = filtered.get("metadata") or {}
            if isinstance(filtered_meta, dict):
                subgraph = filtered_meta.get("_subgraph_info") or filtered_meta.get("subgraph_info")
                if isinstance(subgraph, dict) and subgraph:
                    return subgraph
    return None


def _serialize_deepsearch_chunks(evidences: Iterable[Dict[str, Any]], limit: Optional[int]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for idx, evidence in enumerate(evidences):
        if limit is not None and idx >= limit:
            break
        provenance = evidence.get("provenance") or {}
        metadata = provenance.get("metadata") or {}
        file_id = None
        if isinstance(metadata, dict):
            chunk_meta = metadata.get("chunk_metadata")
            if isinstance(chunk_meta, dict):
                token = str(chunk_meta.get("source_file_id") or "").strip()
                file_id = token or None

        safe_meta = dict(metadata) if isinstance(metadata, dict) else {}
        if file_id and isinstance(safe_meta.get("chunk_metadata"), dict):
            safe_meta["chunk_metadata"] = _rewrite_chunk_metadata_urls(dict(safe_meta["chunk_metadata"]), file_id=file_id)

        content = _rewrite_image_urls_in_markdown(str(evidence.get("content") or ""), file_id=file_id)
        entries.append(
            {
                "chunk_id": evidence.get("chunk_id"),
                "source": evidence.get("source"),
                "file_id": file_id,
                "document_url": _document_download_url(file_id) if file_id else None,
                "content": content,
                "score": evidence.get("score"),
                "metadata": safe_meta,
            }
        )
    return entries


def build_deepsearch_evidence(
    payload: Dict[str, Any],
    *,
    chunk_limit: Optional[int] = DEEPSEARCH_TOP_CHUNKS,
    graph_store: Any | None = None,
) -> Dict[str, Any]:
    """Construct evidence payload from DeepSearch service output."""

    report_block = payload.get("report") or {}
    evidences = report_block.get("evidences") or []
    chunk_entries = _serialize_deepsearch_chunks(evidences, chunk_limit)

    subgraph_info = _first_subgraph_info_from_reasoning(payload)
    graph_snapshot = _ensure_subgraph(
        None,
        subgraph_info,
        node_limit=DEEPSEARCH_GRAPH_NODE_LIMIT,
        edge_limit=DEEPSEARCH_GRAPH_EDGE_LIMIT,
        graph_store=graph_store,
    )
    graph_chain = build_graph_chain(graph_snapshot)

    evidence = {
        "chunks": chunk_entries,
        "seed_entities": _seed_entities_from_subgraph(graph_snapshot, DEEPSEARCH_TOP_SEED_ENTITIES),
        "triples": _triples_from_subgraph(graph_snapshot, DEEPSEARCH_TOP_TRIPLES),
        "graph": graph_snapshot or {},
        "graph_stats": _graph_stats(graph_snapshot),
        "graph_chain": graph_chain,
    }
    return evidence
