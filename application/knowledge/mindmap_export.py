import hashlib
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable

from fastapi import HTTPException, status

from encapsulation.data_model.orm_models import FileMindmapCache
from core.prompts import MINDMAP_MERGE_SYSTEM_PROMPT_EN, build_mindmap_merge_user_prompt
from config.output_limits import KNOWLEDGE_MINDMAP_EXPORT_SEGMENT_SNIPPET_CHARS
from core.mindmap.utils import add_chunks_to_nodes

ProgressCallback = Callable[[str, str, int | None, dict[str, Any] | None], None]


def build_mindmap_merge_prompt(filename: str, chunks: List[Dict[str, Any]]) -> str:
    sections = []
    for idx, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id", "")
        chunk_index = chunk.get("chunk_index", "")
        content = chunk.get("content", "") or ""
        max_chars = int(KNOWLEDGE_MINDMAP_EXPORT_SEGMENT_SNIPPET_CHARS or 0) or 600
        snippet = (content[:max_chars] + "...") if len(content) > max_chars else content
        mindmap = chunk.get("mindmap", {}) or {}
        mindmap_tsv = mindmap_dict_to_tsv(mindmap)

        block = (
            f"### Segment {idx} (Chunk ID: {chunk_id}, Chunk Index: {chunk_index})\n"
            f"Content summary:\n{snippet}\n"
        )
        if mindmap_tsv.strip():
            block += f"\nLocal mind map (TSV):\n{mindmap_tsv}\n"
        sections.append(block)

    sections_text = "\n".join(sections)
    return build_mindmap_merge_user_prompt(filename=filename, sections_text=sections_text)


def mindmap_dict_to_tsv(mindmap: Dict[str, Any]) -> str:
    nodes = mindmap.get("nodes", []) if isinstance(mindmap, dict) else []
    lines = []
    for node in nodes:
        level = node.get("level") if isinstance(node, dict) else None
        content = node.get("content") if isinstance(node, dict) else None
        if level and content:
            lines.append(f"{level}\t{content}")
    return "\n".join(lines) if lines else ""


def extract_tsv_from_response(response: str) -> str:
    if not response:
        return ""

    if "```" in response:
        start = None
        end = None
        for marker in ("```tsv", "```txt", "```text", "```"):
            if marker in response:
                start = response.find(marker) + len(marker)
                end = response.find("```", start)
                if end != -1:
                    break
        if start is not None and end != -1:
            return response[start:end].strip()

    return response.strip()


def convert_tsv_to_graph(tsv_text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    entries: List[Tuple[str, str]] = []
    for line in tsv_text.strip().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "\t" not in stripped:
            continue
        level, content = stripped.split("\t", 1)
        level = level.strip()
        content = content.strip()
        if not level or not content:
            continue
        entries.append((level, content))

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    node_lookup: Dict[str, Dict[str, Any]] = {}

    for level, content in entries:
        depth = len(level.split(".")) if level else 1
        node_id = f"{level} {content}"
        parent_level = ".".join(level.split(".")[:-1]) if depth > 1 else None
        parent_info = node_lookup.get(parent_level) if parent_level else None

        if depth <= 2:
            category = content
        else:
            level_parts = level.split(".")
            second_level = ".".join(level_parts[:2]) if len(level_parts) >= 2 else None
            second_level_info = node_lookup.get(second_level) if second_level else None
            category = second_level_info["name"] if second_level_info else content

        node_data = {
            "id": node_id,
            "name": content,
            "category": category,
            "weight": depth,
        }
        nodes.append(node_data)
        node_lookup[level] = {"id": node_id, "name": content}

        if parent_info:
            parent_id = parent_info["id"]
            if depth == 2:
                edge_weight = 0.85
            elif depth == 3:
                edge_weight = 0.8
            else:
                edge_weight = 0.75

            edge_data = {
                "id": f"edge-{len(edges) + 1:03d}",
                "source": parent_id,
                "target": node_id,
                "relation": "contains",
                "weight": edge_weight,
            }
            edges.append(edge_data)

    return nodes, edges


async def export_file_mindmap_payload(
    *,
    knowledge: Any,
    rag_inference: Any,
    file_id: str,
    owner_id: Any,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if progress:
        progress("mindmap_export", "start", 1, {"file_id": file_id})

    try:
        file_mindmaps = await knowledge.get_file_chunk_mindmaps(file_id, owner_id)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to gather chunk mind maps: {exc}",
        ) from exc

    chunks = file_mindmaps.get("chunks", []) if isinstance(file_mindmaps, dict) else []
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No mind map data found for this file",
        )

    filename = file_mindmaps.get("filename") or file_id

    metadata_store = getattr(getattr(knowledge, "file_storage", None), "metadata_store", None)
    if metadata_store is not None and hasattr(metadata_store, "SessionMaker"):
        try:
            with metadata_store.SessionMaker() as session:
                cache = session.query(FileMindmapCache).filter_by(file_id=file_id).first()
                if cache:
                    if progress:
                        progress("mindmap_export", "cache_hit", 100, {"file_id": file_id})
                    # Add chunks to all nodes for cached data
                    cached_nodes = list(cache.nodes)
                    cached_nodes = add_chunks_to_nodes(cached_nodes, chunks, filename, file_id)
                    return {"tsv": cache.tsv, "nodes": cached_nodes, "edges": list(cache.edges)}
        except Exception:
            pass
    prompt = build_mindmap_merge_prompt(filename, chunks)

    llm = getattr(rag_inference, "llm", None)
    if llm is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM service is not configured",
        )

    if progress:
        progress("mindmap_export", "llm_merge", 50, {"file_id": file_id})

    messages = [
        {
            "role": "system",
            "content": MINDMAP_MERGE_SYSTEM_PROMPT_EN,
        },
        {"role": "user", "content": prompt},
    ]

    try:
        llm_response = llm.chat(messages)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate merged mind map: {exc}",
        ) from exc

    merged_tsv = extract_tsv_from_response(llm_response)
    if not merged_tsv.strip():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM did not return valid TSV content",
        )

    nodes, edges = convert_tsv_to_graph(merged_tsv)

    if metadata_store is not None and hasattr(metadata_store, "SessionMaker"):
        try:
            with metadata_store.SessionMaker() as session:
                now = datetime.now()
                nodes_data = [
                    {"id": n["id"], "name": n["name"], "category": n["category"], "weight": n.get("weight", 1)}
                    for n in nodes
                ]
                edges_data = [
                    {"id": e["id"], "source": e["source"], "target": e["target"], "relation": e.get("relation", "contains"), "weight": e.get("weight", 1.0)}
                    for e in edges
                ]

                chunk_ids = sorted([chunk.get("chunk_id", "") for chunk in chunks])
                chunk_hash = hashlib.sha256("|".join(chunk_ids).encode()).hexdigest()

                cache = session.query(FileMindmapCache).filter_by(file_id=file_id).first()
                if cache:
                    cache.tsv = merged_tsv
                    cache.nodes = nodes_data
                    cache.edges = edges_data
                    cache.chunk_hash = chunk_hash
                    cache.updated_at = now
                else:
                    cache = FileMindmapCache(
                        file_id=file_id,
                        tsv=merged_tsv,
                        nodes=nodes_data,
                        edges=edges_data,
                        chunk_hash=chunk_hash,
                        created_at=now,
                        updated_at=now,
                    )
                    session.add(cache)
                session.commit()
        except Exception:
            pass

    if progress:
        progress("mindmap_export", "end", 100, {"file_id": file_id})

    # Add chunks to all nodes as source evidence
    nodes = add_chunks_to_nodes(nodes, chunks, filename, file_id)

    return {"tsv": merged_tsv, "nodes": nodes, "edges": edges}
