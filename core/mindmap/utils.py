"""Utility functions for mindmap data formatting."""
from pathlib import Path
from typing import Any, Dict, List


def convert_chunks_to_frontend_format(
    chunks: List[Dict[str, Any]], 
    filename: str | None = None, 
    file_id: str | None = None
) -> List[Dict[str, Any]]:
    """
    Convert chunks to frontend format with id, title, content, documentName, fileId.
    
    Args:
        chunks: List of chunk dictionaries. Each chunk should have:
            - "content": chunk content (required)
            - "metadata": dict containing "filename" and/or "source_file_id" (optional, for multi-file mode)
        filename: Single filename for all chunks (used in single-file mode).
                  If None, will extract from chunk.metadata["filename"] for each chunk.
        file_id: Single file_id for all chunks (used in single-file mode).
                 If None, will extract from chunk.metadata["source_file_id"] for each chunk.
    
    Returns:
        List of frontend-formatted chunk dictionaries with:
            - id: "c1", "c2", ...
            - title: "片段1", "片段2", ...
            - content: chunk content
            - documentName: filename (only name part, no path prefix)
            - fileId: file_id (if available)
    """
    frontend_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        metadata = chunk.get("metadata") or {}
        
        # Determine filename and file_id:
        # - If both provided (single-file mode), use them for all chunks
        # - Otherwise (multi-file mode), extract from each chunk's metadata
        if filename is not None and file_id is not None:
            # Single-file mode: use provided filename and file_id for all chunks
            chunk_filename = filename
            chunk_file_id = file_id
        else:
            # Multi-file mode: extract from each chunk's metadata
            chunk_filename = str(metadata.get("filename") or "").strip() or filename or "source"
            chunk_file_id = str(metadata.get("source_file_id") or "").strip() or file_id
        
        # Extract only the filename without path prefix (e.g., "分析.png" from "RAG-ARC/分析.png")
        document_name = Path(chunk_filename).name
        
        frontend_chunks.append(
            {
                "id": f"c{idx}",
                "title": f"片段{idx}",
                "content": chunk.get("content", ""),
                "documentName": document_name,
                "fileId": chunk_file_id if chunk_file_id else None,
            }
        )
    return frontend_chunks


def add_chunks_to_nodes(
    nodes: List[Dict[str, Any]], 
    chunks: List[Dict[str, Any]], 
    filename: str | None = None, 
    file_id: str | None = None
) -> List[Dict[str, Any]]:
    """
    Add chunks field to all nodes as source evidence for each node.
    
    Args:
        nodes: List of node dictionaries to add chunks to
        chunks: List of chunk dictionaries (see convert_chunks_to_frontend_format)
        filename: Single filename for all chunks (optional, for single-file mode)
        file_id: Single file_id for all chunks (optional, for single-file mode)
    
    Returns:
        Modified nodes list with chunks field added to each node
    """
    if not nodes:
        return nodes

    # Convert chunks to frontend format once
    frontend_chunks = convert_chunks_to_frontend_format(chunks, filename, file_id)

    # Add chunks to all nodes (each weight level should have chunks as source evidence)
    for node in nodes:
        node["chunks"] = frontend_chunks

    return nodes


def ensure_mindmap_edge_relation(edges: List[Dict[str, Any]], relation: str = "contains") -> List[Dict[str, Any]]:
    """
    Ensure all edges have the specified relation type.
    
    Args:
        edges: List of edge dictionaries
        relation: Desired relation type (default: "contains" for mindmaps)
    
    Returns:
        Modified edges list with relation field set
    """
    for edge in edges:
        if "relation" not in edge or edge.get("relation") != relation:
            edge["relation"] = relation
    return edges