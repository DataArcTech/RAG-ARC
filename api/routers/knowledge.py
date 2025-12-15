from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    status,
    Query,
)
from typing import Annotated, Optional, List, Dict, Any, Tuple
from pydantic import BaseModel
from api.routers.auth import get_current_user
from encapsulation.data_model.orm_models import User
from framework.register import Register

router = APIRouter(prefix="/knowledge", tags=["files"])

registrator = Register()
account_handler = registrator.get_object("account")
knowledge_handler = registrator.get_object("knowledge")


# Response models
class FileInfo(BaseModel):
    """Response model for file information"""
    file_id: str
    filename: str
    status: str
    created_at: str
    updated_at: str
    file_size: int
    content_type: str

    model_config = {"from_attributes": True}


class FileListResponse(BaseModel):
    """Response model for file list"""
    files: List[FileInfo]
    total: int

@router.post(
    "",
    response_model=str,
    status_code=status.HTTP_201_CREATED,
)
async def upload_file(
    file: UploadFile,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """
    Upload a file to the knowledge base

    Args:
        file: File to upload
        owner_id: User ID string. Defaults to a placeholder string.
                  After adding JWT authentication, this will be extracted from the token.

    Returns:
        Document ID
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    try:
        print(f"Uploading file: {file.filename} for owner_id: {user.id}")
        # Convert string UUID to UUID object
        doc_id = await knowledge_handler.upload_file(file, user.id)
        return doc_id
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid user.id format: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}",
        )


@router.get("/{file_id}/download")
async def download_file(file_id: str, user: Annotated[User | None, Depends(get_current_user)]):
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    try:
        return knowledge_handler.get_file(file_id, user.id)
    except HTTPException:
        # re-raise 404s from underlying module
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download file: {str(e)}",
        )


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(file_id: str, user: Annotated[User | None, Depends(get_current_user)]):
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    try:
        knowledge_handler.delete_file(file_id, user.id)
        return None
    except HTTPException:
        # surface 404s and 403s if thrown by storage layer
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid user.id format: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {str(e)}",
        )

@router.get(
    "/list_files",
    response_model=FileListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_files(
    user: Annotated[User | None, Depends(get_current_user)],
    limit: Optional[int] = Query(default=100, ge=1, le=1000, description="Maximum number of files to return"),
    offset: Optional[int] = Query(default=0, ge=0, description="Number of files to skip"),
):
    """
    Get all files for the current user.
    
    Returns a list of files with their metadata including:
    - file_id: Unique identifier for the file
    - filename: Original filename
    - status: Current processing status (STORED, PARSED, CHUNKED, INDEXED, FAILED, DELETED)
    - created_at: Timestamp when file was uploaded
    - updated_at: Timestamp when file was last updated
    - file_size: Size of the file in bytes
    - content_type: MIME type of the file
    
    Args:
        user: Current authenticated user (automatically injected)
        limit: Maximum number of files to return (default: 100, max: 1000)
        offset: Number of files to skip for pagination (default: 0)
        
    Returns:
        FileListResponse with list of files and total count
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    try:
        # Get files for current page
        files = knowledge_handler.list_user_files(
            user_id=user.id,
            limit=limit,
            offset=offset
        )
        
        # Get total count of files for the user
        total_count = knowledge_handler.count_user_files(user.id)
        
        # Convert FileMetadata objects to FileInfo response models
        file_infos = [
            FileInfo(
                file_id=file.file_id,
                filename=file.filename,
                status=file.status.value,  # Convert enum to string
                created_at=file.created_at.isoformat(),
                updated_at=file.updated_at.isoformat(),
                file_size=file.file_size,
                content_type=file.content_type
            )
            for file in files
        ]
        
        return FileListResponse(
            files=file_infos,
            total=total_count
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve files: {str(e)}",
        )


class IndexTriggerRequest(BaseModel):
    """Request model for triggering indexing"""
    file_ids: List[str]

class IndexTriggerResponse(BaseModel):
    """Response model for index triggering results"""
    message: str

class GraphExportRequest(BaseModel):
    """Request model for graph export"""
    max_nodes: int = 500
    max_edges: int = 2000
    include_node_types: Optional[List[str]] = None  # e.g., ['chunk', 'entity', 'fact']


class MindmapNode(BaseModel):
    """Mind map node structure."""
    id: str
    name: str
    category: str
    weight: int


class MindmapEdge(BaseModel):
    """Mind map edge structure."""
    id: str
    source: str
    target: str
    relation: str
    weight: float


class MindmapExportRequest(BaseModel):
    """Request model for exporting merged mind map."""
    file_id: str


class MindmapExportResponse(BaseModel):
    """Response model for exported mind map."""
    tsv: str
    nodes: List[MindmapNode]
    edges: List[MindmapEdge]


@router.post(
    "/trigger_indexing",
    response_model=IndexTriggerResponse,
    status_code=status.HTTP_200_OK,
)
async def trigger_indexing(
    request: IndexTriggerRequest,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """
    Trigger indexing for multiple files.

    Args:
        request: IndexTriggerRequest containing list of file IDs
        user: Current authenticated user

    Returns:
        IndexTriggerResponse with indexing results
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    if not request.file_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="file_ids list cannot be empty"
        )

    try:
        result = await knowledge_handler.trigger_indexing(request.file_ids, user.id)
        
        return IndexTriggerResponse(
            message=result
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (404, 403)
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger indexing: {str(e)}",
        )


@router.post("/graph/export", status_code=status.HTTP_200_OK)
def export_knowledge_graph(
    request: GraphExportRequest,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """
    Export the complete knowledge graph for the current user

    Args:
        request: GraphExportRequest with export parameters
        user: Current authenticated user

    Returns:
        Graph data in Cytoscape.js format with nodes and edges
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    try:
        # Get the RAG inference handler to access the retriever
        rag_inference = registrator.get_object("rag_inference")

        # Find graph_store from retriever (support both direct and multipath retrievers)
        graph_store = None

        # Check if retriever has graph_store directly
        if hasattr(rag_inference.retriever, 'graph_store'):
            graph_store = rag_inference.retriever.graph_store
        # Check if it's a multipath retriever with sub-retrievers
        elif hasattr(rag_inference.retriever, 'retrievers'):
            # Find the first retriever with graph_store
            for sub_retriever in rag_inference.retriever.retrievers:
                if hasattr(sub_retriever, 'graph_store'):
                    graph_store = sub_retriever.graph_store
                    break

        if graph_store is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current retriever does not support graph visualization"
            )

        # Import appropriate GraphExporter based on graph_store type
        # Check by class name to avoid import issues
        graph_store_class_name = graph_store.__class__.__name__

        if graph_store_class_name == 'PrunedHippoRAGNeo4jStore':
            from encapsulation.database.utils.graph_export_utils_neo4j import GraphExporterNeo4j as GraphExporter
        else:
            from encapsulation.database.utils.graph_export_utils import GraphExporter

        # Export full graph
        graph_data = GraphExporter.export_full_graph(
            graph_store=graph_store,
            max_nodes=request.max_nodes,
            max_edges=request.max_edges,
            include_node_types=request.include_node_types
        )

        return graph_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export knowledge graph: {str(e)}",
        )


@router.post(
    "/mindmap/export",
    response_model=MindmapExportResponse,
    status_code=status.HTTP_200_OK,
)
async def export_file_mindmap(
    request: MindmapExportRequest,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """Export merged mind map for a specific file."""
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    if not request.file_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="file_id is required"
        )

    try:
        file_mindmaps = await knowledge_handler.get_file_chunk_mindmaps(request.file_id, user.id)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to gather chunk mind maps: {str(e)}",
        )

    chunks = file_mindmaps.get("chunks", []) if isinstance(file_mindmaps, dict) else []
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No mind map data found for this file"
        )

    filename = file_mindmaps.get("filename") or request.file_id
    prompt = _build_mindmap_merge_prompt(filename, chunks)

    rag_inference = registrator.get_object("rag_inference")
    llm = getattr(rag_inference, "llm", None)
    if llm is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM service is not configured"
        )

    messages = [
        {
            "role": "system",
            "content": "你是一位资深的知识工程专家，擅长将多个思维导图整合为结构化的全局思维导图。"
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        llm_response = llm.chat(messages)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate merged mind map: {str(e)}"
        )

    merged_tsv = _extract_tsv_from_response(llm_response)
    if not merged_tsv.strip():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LLM did not return valid TSV content"
        )

    nodes, edges = _convert_tsv_to_graph(merged_tsv)

    return MindmapExportResponse(
        tsv=merged_tsv,
        nodes=[MindmapNode(**node) for node in nodes],
        edges=[MindmapEdge(**edge) for edge in edges],
    )


def _build_mindmap_merge_prompt(filename: str, chunks: List[Dict[str, Any]]) -> str:
    sections = []
    for idx, chunk in enumerate(chunks, start=1):
        chunk_id = chunk.get("chunk_id", "")
        chunk_index = chunk.get("chunk_index")
        content = chunk.get("content", "") or ""
        snippet = content.strip().replace("\t", " ")
        if len(snippet) > 400:
            snippet = snippet[:400] + "..."

        mindmap = chunk.get("mindmap", {}) or {}
        mindmap_tsv = _mindmap_dict_to_tsv(mindmap)

        sections.append(
            f"### 片段 {idx} (Chunk ID: {chunk_id}, Chunk Index: {chunk_index})\n"
            f"内容摘要:\n{snippet}\n\n"
            f"局部思维导图 (TSV):\n{mindmap_tsv}\n"
        )

    sections_text = "\n".join(sections)

    prompt = (
        f"我们从文档《{filename}》的多个片段中提取了思维导图片段 (TSV 格式)。"
        "请综合这些片段，生成一个完整的全局思维导图，仍然使用 TSV 层级编号。\n\n"
        "输出要求:\n"
        "1. 使用 1, 1.1, 1.1.1 等编号表达层级，编号需连续、严谨。\n"
        "2. 仅输出 TSV 内容，每一行采用“编号\t内容”的形式，不要添加额外说明、标题或前缀。\n"
        "3. 根节点 (编号 1) 应对整篇文档进行高度概括。\n"
        "4. 二级及以下节点应覆盖所有重要信息，保持表达简洁准确。\n"
        "5. 若某些信息重复或冲突，请自行消解并保持结构一致性。\n\n"
        "以下是分片的思维导图信息：\n"
        f"{sections_text}\n"
        "请现在输出汇总后的 TSV 思维导图："
    )

    return prompt


def _mindmap_dict_to_tsv(mindmap: Dict[str, Any]) -> str:
    nodes = mindmap.get("nodes", []) if isinstance(mindmap, dict) else []
    lines = []
    for node in nodes:
        level = node.get("level") if isinstance(node, dict) else None
        content = node.get("content") if isinstance(node, dict) else None
        if level and content:
            lines.append(f"{level}\t{content}")
    return "\n".join(lines) if lines else "(空)"


def _extract_tsv_from_response(response: str) -> str:
    if not response:
        return ""

    if "```" in response:
        # Try to capture code block (prefer ```tsv`` or ```)
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


def _convert_tsv_to_graph(tsv_text: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
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
        depth = len(level.split('.')) if level else 1
        node_id = f"{level} {content}"
        parent_level = '.'.join(level.split('.')[:-1]) if depth > 1 else None
        parent_info = node_lookup.get(parent_level) if parent_level else None


        if depth <= 2:
            category = content
        else:
            level_parts = level.split('.')
            second_level = '.'.join(level_parts[:2]) if len(level_parts) >= 2 else None
            second_level_info = node_lookup.get(second_level) if second_level else None
            category = second_level_info["name"] if second_level_info else content

        node_data = {
            "id": node_id,
            "name": content,
            "category": category,
            "weight": depth
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
                "relation": "包含",
                "weight": edge_weight
            }
            edges.append(edge_data)

    return nodes, edges
