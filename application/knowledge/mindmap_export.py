import hashlib
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

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


def extract_relations_from_chunks(
    chunks: List[Dict[str, Any]], 
    nodes: List[Dict[str, Any]],
    knowledge: Any
) -> List[Dict[str, Any]]:
    """
    从 chunks 的 graph 数据中提取关系，并转换为 edges。
    
    Args:
        chunks: Chunk 字典列表，包含 chunk_id
        nodes: 已创建的节点列表，用于匹配实体名称
        knowledge: Knowledge 模块实例，用于获取 graph_store
    
    Returns:
        关系 edges 列表
    """
    relations_edges: List[Dict[str, Any]] = []
    
    # 获取 graph_store
    graph_store = None
    if hasattr(knowledge, "file_index"):
        for indexer in getattr(knowledge.file_index, "indexers", []):
            if hasattr(indexer, "graph_store"):
                graph_store = indexer.graph_store
                break
    
    if graph_store is None:
        return relations_edges
    
    # 构建节点名称到节点ID的映射（用于匹配实体）
    # 支持精确匹配和部分匹配
    node_name_to_id: Dict[str, str] = {}
    node_name_parts: List[Tuple[str, str]] = []  # (节点名称片段, 节点ID)
    for node in nodes:
        node_name = node.get("name", "").strip()
        if node_name:
            node_name_to_id[node_name] = node.get("id", "")
            # 提取节点名称的关键部分（去除冒号后的描述等）
            # 例如："公司名称: 数创弧光智能科技有限公司" -> "数创弧光智能科技有限公司"
            name_parts = node_name.split(":")
            if len(name_parts) > 1:
                key_part = name_parts[-1].strip()
                if key_part:
                    node_name_parts.append((key_part, node.get("id", "")))
            # 提取括号中的内容
            if "(" in node_name and ")" in node_name:
                bracket_content = re.findall(r'\(([^)]+)\)', node_name)
                for content in bracket_content:
                    node_name_parts.append((content.strip(), node.get("id", "")))
    
    # 从每个 chunk 的 graph 数据中提取关系
    chunk_ids = [chunk.get("chunk_id", "") for chunk in chunks if chunk.get("chunk_id")]
    if not chunk_ids:
        return relations_edges
    
    try:
        # 获取 chunks 的 graph 数据
        chunk_objects = graph_store.get_by_ids(chunk_ids)
        
        edge_counter = 0
        for chunk in chunk_objects:
            if not chunk or not hasattr(chunk, "graph") or not chunk.graph:
                continue
            
            graph_data = chunk.graph
            if not hasattr(graph_data, "relations"):
                continue
            
            # 处理每个关系
            for relation in graph_data.relations:
                if not isinstance(relation, list) or len(relation) < 3:
                    continue
                
                head_name = str(relation[0]).strip()
                relation_type = str(relation[1]).strip() if len(relation) > 1 else ""
                tail_name = str(relation[2]).strip() if len(relation) > 2 else ""
                
                if not relation_type or not head_name or not tail_name:
                    continue
                
                # 跳过 contains 关系（已由 TSV 层级结构处理）
                if relation_type.lower() in ("contains", "包含", "none", ""):
                    continue
                
                # 查找对应的节点ID（支持精确匹配和部分匹配）
                head_node_id = node_name_to_id.get(head_name)
                tail_node_id = node_name_to_id.get(tail_name)
                
                # 如果精确匹配失败，尝试部分匹配
                if not head_node_id:
                    for part, nid in node_name_parts:
                        if head_name in part or part in head_name:
                            head_node_id = nid
                            break
                    # 如果还是找不到，尝试在完整节点名称中查找
                    if not head_node_id:
                        for node_name, nid in node_name_to_id.items():
                            if head_name in node_name or node_name in head_name:
                                head_node_id = nid
                                break
                
                if not tail_node_id:
                    for part, nid in node_name_parts:
                        if tail_name in part or part in tail_name:
                            tail_node_id = nid
                            break
                    if not tail_node_id:
                        for node_name, nid in node_name_to_id.items():
                            if tail_name in node_name or node_name in tail_name:
                                tail_node_id = nid
                                break
                
                # 如果实体名称在节点中存在，创建关系 edge
                if head_node_id and tail_node_id:
                    edge_counter += 1
                    relations_edges.append({
                        "id": f"relation-{edge_counter:03d}",
                        "source": head_node_id,
                        "target": tail_node_id,
                        "relation": relation_type,
                        "weight": 0.7,  # 关系边的默认权重
                    })
                # 如果实体不在节点中，但关系类型重要，可以考虑创建新节点
                # 这里为了最小代价，暂时跳过
                
    except Exception as e:
        # 静默失败，不影响主流程
        logger.warning(f"Failed to extract relations from chunks: {e}")
    
    return relations_edges


def convert_tsv_to_graph(tsv_text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    entries: List[Tuple[str, str, Optional[str], Optional[str]]] = []  # (level, head, relation, tail)
    for line in tsv_text.strip().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "\t" not in stripped:
            continue
        parts = [p.strip() for p in stripped.split("\t")]
        level = parts[0]
        if not level:
            continue
        
        # 解析不同格式：
        # 格式1: 编号\t内容 (2列)
        # 格式2: 编号\t内容\t关系 (3列，关系可能是 contains)
        # 格式3: 编号\t实体1\t关系\t实体2 (4列，三元组)
        # 格式4: 编号\t实体1\t关系1\t实体2\t关系2\t实体3... (多列)
        
        if len(parts) == 2:
            # 格式1: 只有编号和内容
            entries.append((level, parts[1], None, None))
        elif len(parts) == 3:
            # 格式2: 编号\t内容\t关系
            entries.append((level, parts[1], parts[2], None))
        elif len(parts) >= 4:
            # 格式3/4: 三元组或多列关系
            # 取前两个实体和第一个关系作为主要关系
            head = parts[1]
            relation = parts[2]
            tail = "\t".join(parts[3:])  # 合并后面的所有列作为尾实体
            entries.append((level, head, relation, tail))
        else:
            continue

    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    node_lookup: Dict[str, Dict[str, Any]] = {}
    entity_node_map: Dict[str, str] = {}  # 实体名称 -> 节点ID映射

    for entry in entries:
        level, head, relation, tail = entry
        # 构建节点显示内容
        if tail:
            content = f"{head}\t{relation}\t{tail}"
        else:
            content = head
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

        # 提取节点名称（使用头实体作为主要名称）
        node_name = head.strip() if head else content.split("\t")[0].strip()
        if not node_name:
            node_name = content
        
        node_data = {
            "id": node_id,
            "name": node_name,
            "category": category,
            "weight": depth,
        }
        nodes.append(node_data)
        node_lookup[level] = {"id": node_id, "name": node_name}
        
        # 记录实体到节点的映射（用于关系匹配）
        if head:
            entity_node_map[head.strip()] = node_id
        if tail:
            entity_node_map[tail.strip()] = node_id

        if parent_info:
            parent_id = parent_info["id"]
            if depth == 2:
                edge_weight = 0.85
            elif depth == 3:
                edge_weight = 0.8
            else:
                edge_weight = 0.75

            # 如果有明确的关系类型且不是 contains，使用该关系；否则使用 contains
            edge_relation = relation if relation and relation.lower() not in ("contains", "包含") else "contains"
            
            edge_data = {
                "id": f"edge-{len(edges) + 1:03d}",
                "source": parent_id,
                "target": node_id,
                "relation": edge_relation,
                "weight": edge_weight,
            }
            edges.append(edge_data)
        
        # 如果是三元组格式（有 tail），创建额外的关系 edge
        if tail and relation and relation.lower() not in ("contains", "包含"):
            head_entity = head.strip()
            tail_entity = tail.strip()
            rel_type = relation.strip()
            
            # 查找头实体和尾实体对应的节点
            head_node_id = entity_node_map.get(head_entity)
            tail_node_id = entity_node_map.get(tail_entity)
            
            # 如果找不到，尝试在当前节点中查找（部分匹配）
            if not head_node_id:
                for n in nodes:
                    n_name = n.get("name", "")
                    if head_entity in n_name or n_name in head_entity or head_entity == n_name:
                        head_node_id = n.get("id")
                        break
            if not tail_node_id:
                for n in nodes:
                    n_name = n.get("name", "")
                    if tail_entity in n_name or n_name in tail_entity or tail_entity == n_name:
                        tail_node_id = n.get("id")
                        break
            
            # 如果找到两个不同的节点，创建关系 edge
            if head_node_id and tail_node_id and head_node_id != tail_node_id and rel_type:
                edges.append({
                    "id": f"relation-{len(edges) + 1:03d}",
                    "source": head_node_id,
                    "target": tail_node_id,
                    "relation": rel_type,
                    "weight": 0.7,
                })

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
    
    # 从 chunks 的 graph 数据中提取关系并添加到 edges
    relation_edges = extract_relations_from_chunks(chunks, nodes, knowledge)
    edges.extend(relation_edges)

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
