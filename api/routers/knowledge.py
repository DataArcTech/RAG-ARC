from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    Form,
    status,
    Query,
    Body,
)
import asyncio
import logging
import os
import re
import time
import uuid
from typing import Annotated, Optional, List, Dict, Any, Tuple, Literal
from datetime import datetime
from api.routers.auth import get_current_user
from api.sse import sse_done, sse_json
from encapsulation.data_model.orm_models import (
    User,
    Department,
    FilePermission,
    PermissionReceiverType,
    PermissionType,
)
from framework.register import Register
from application.knowledge.module import Knowledge
from application.account.user import Account
from core.file_management.storage.file import FileValidationError
from core.utils.owner_guard import is_admin_owner
from core.utils.path_guard import safe_leaf_name
from encapsulation.message_queue.redis_task_queue import RedisTaskQueue, TaskState
from fastapi.responses import StreamingResponse, FileResponse
from pathlib import Path

from application.knowledge.graph_export import export_full_graph_payload
from application.knowledge.mindmap_export import export_file_mindmap_payload
from api.routers.knowledge_models import (
    CheckAccessResponse,
    DepartmentInfo,
    FileInfo,
    FileListResponse,
    FileTaskStatusResponse,
    GrantPermissionRequest,
    GrantPermissionResponse,
    GraphExportRequest,
    IndexTriggerRequest,
    IndexTriggerResponse,
    MindmapEdge,
    MindmapExportRequest,
    MindmapExportResponse,
    MindmapNode,
    PermissionInfo,
    PermissionListResponse,
    RevokePermissionRequest,
    TaskRunStatusResponse,
    UserInfo,
)
from api.routers.knowledge_streaming import stream_events_redis

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["files"])

registrator = Register()

def get_account_handler() -> Account:
    """Lazy loading function to get account handler after initialization."""
    return registrator.get_object("account")

def get_knowledge_handler() -> Knowledge:
    """Lazy loading function to get knowledge handler after initialization."""
    return registrator.get_object("knowledge")


def _get_task_queue() -> RedisTaskQueue:
    return RedisTaskQueue.from_env()


def _use_celery() -> bool:
    return os.getenv("TASK_QUEUE_MODE", "inprocess").strip().lower() == "celery"


def _assert_task_owner(task_run: Dict[str, Any], *, user_id: uuid.UUID) -> None:
    if is_admin_owner(user_id):
        return
    raw = task_run.get("owner_id")
    if not raw:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed to access this run_id")
    try:
        owner_uuid = uuid.UUID(str(raw))
    except Exception:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed to access this run_id") from None
    if owner_uuid != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not allowed to access this run_id")


def _format_sse(*, event: str, data: Dict[str, Any], event_id: int | None = None) -> str:
    payload: Dict[str, Any] = {"event": event, "data": data}
    if event_id is not None:
        payload["id"] = event_id
    return sse_json(payload)


def _normalize_file_id(file_id: str) -> str:
    """Normalize file_id by extracting the last valid UUID if it appears to be duplicated."""
    # Standard UUID format: xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx (36 chars)
    if len(file_id) <= 36:
        return file_id
    # If length > 36, try to extract valid UUID from the end
    # Try last 36 characters first (most common case)
    if len(file_id) >= 36:
        candidate = file_id[-36:]
        parts = candidate.split('-')
        if len(parts) == 5 and [len(p) for p in parts] == [8, 4, 4, 4, 12]:
            return candidate
    # Fallback: try from position 29 (for 72-char duplicated UUIDs)
    if len(file_id) >= 65:
        candidate = file_id[29:29+36]
        parts = candidate.split('-')
        if len(parts) == 5 and [len(p) for p in parts] == [8, 4, 4, 4, 12]:
            return candidate
    return file_id
@router.post(
    "",
    status_code=status.HTTP_200_OK,
)
async def upload_file(
    file: UploadFile,
    user: Annotated[User | None, Depends(get_current_user)],
    relative_path: Optional[str] = Form(
        default=None,
        description="Optional repo-relative path (e.g. RAG-ARC/docs/a.pdf).",
    ),
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
        # Convert string UUID to UUID object
        doc_id = await get_knowledge_handler().upload_file(file, user.id, relative_path=relative_path)
        return doc_id
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid user.id format: {str(e)}",
        )
    except FileValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}",
        )


@router.get("/{file_id}/task", response_model=FileTaskStatusResponse, status_code=status.HTTP_200_OK)
async def get_file_task_status(
    file_id: str,
    user: Annotated[User | None, Depends(get_current_user)],
):
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )
    try:
        status_payload = await get_knowledge_handler().get_file_task_status(file_id, user.id)
        return FileTaskStatusResponse(**status_payload)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}",
        )


@router.get("/task_run/{run_id}", response_model=TaskRunStatusResponse, status_code=status.HTTP_200_OK)
async def get_task_run(
    run_id: str,
    user: Annotated[User | None, Depends(get_current_user)],
):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    if not _use_celery():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Task queue mode is not enabled")
    task_queue = _get_task_queue()
    task_run = await asyncio.to_thread(task_queue.get_task_run, run_id)
    if not task_run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id not found")
    _assert_task_owner(task_run, user_id=user.id)
    return TaskRunStatusResponse(
        run_id=run_id,
        task_type=task_run.get("task_type"),
        state=task_run.get("state"),
        progress_percent=task_run.get("progress_percent"),
        error_message=task_run.get("error_message"),
        resource_id=task_run.get("resource_id"),
        updated_at_ms=task_run.get("updated_at_ms"),
    )


@router.get("/result/{run_id}", status_code=status.HTTP_200_OK)
async def get_task_result(
    run_id: str,
    user: Annotated[User | None, Depends(get_current_user)],
):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    if not _use_celery():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Task queue mode is not enabled")
    task_queue = _get_task_queue()
    task_run = await asyncio.to_thread(task_queue.get_task_run, run_id)
    if not task_run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id not found")
    _assert_task_owner(task_run, user_id=user.id)
    state = str(task_run.get("state") or "")
    if state not in {TaskState.SUCCESS.value, TaskState.FAILURE.value, TaskState.CANCELED.value}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="run not finished")
    if state != TaskState.SUCCESS.value:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(task_run.get("error_message") or "task failed"))
    result = await asyncio.to_thread(task_queue.get_task_result, run_id)
    return result or {"run_id": run_id, "done": True, "result": None}


@router.get("/stream/{run_id}")
async def stream_task_progress(
    run_id: str,
    user: Annotated[User | None, Depends(get_current_user)],
    last_event_id: int = -1,
):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    if not _use_celery():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Task queue mode is not enabled")
    task_queue = _get_task_queue()
    task_run = task_queue.get_task_run(run_id)
    if not task_run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id not found")
    _assert_task_owner(task_run, user_id=user.id)
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        stream_events_redis(task_queue, format_sse=_format_sse, run_id=run_id, last_event_id=last_event_id),
        media_type="text/event-stream",
        headers=headers,
    )


@router.get("/{file_id}/download")
async def download_file(file_id: str, user: Annotated[User | None, Depends(get_current_user)]):
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    try:
        # Log original file_id to debug URL construction issues
        if len(file_id) > 36:
            logger.warning(f"Received malformed file_id (length={len(file_id)}): {file_id[:50]}...")
        # Normalize file_id to handle duplicated UUIDs in URL
        normalized_file_id = _normalize_file_id(file_id)
        if normalized_file_id != file_id:
            logger.info(f"Normalized file_id from {file_id[:50]}... to {normalized_file_id}")
        return await get_knowledge_handler().get_file(normalized_file_id, user.id)
    except HTTPException:
        # re-raise 404s from underlying module
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download file: {str(e)}",
        )


@router.get("/{file_id}/mineru-assets/{rel_path:path}")
async def get_mineru_asset(
    file_id: str,
    rel_path: str,
):
    """Serve MinerU local image assets for a given knowledge file.

    Only allows paths under: `${PARSER_OUTPUT_DIR}/mineru/<file_id>/images/...`

    Backwards compatibility: older runs may have stored artifacts under
    `${PARSER_OUTPUT_DIR}/mineru/<doc_stem>/images/...`.
    """

    metadata = await asyncio.to_thread(get_knowledge_handler().file_storage.get_file_metadata, file_id)
    if metadata is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="File not found")

    token = str(rel_path or "").lstrip("/").lstrip("\\")
    if not token.startswith("images/"):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")

    base_output = str(os.getenv("PARSER_OUTPUT_DIR", "./data/parsed_files") or "./data/parsed_files").strip()
    base_dir = Path(base_output).expanduser().resolve()
    file_dir = (base_dir / "mineru" / safe_leaf_name(file_id, default="document")).resolve()
    stem = Path(str(getattr(metadata, "filename", "") or "document")).stem or "document"
    legacy_dir = (base_dir / "mineru" / safe_leaf_name(stem, default="document")).resolve()

    for doc_dir in [file_dir, legacy_dir]:
        candidate = (doc_dir / token).resolve()
        try:
            candidate.relative_to(doc_dir)
        except Exception:
            continue
        if candidate.exists() and candidate.is_file():
            return FileResponse(path=str(candidate))

    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Asset not found")


@router.delete("/{file_id}", status_code=status.HTTP_200_OK)
async def delete_file(file_id: str, user: Annotated[User | None, Depends(get_current_user)]):
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    try:
        result = await get_knowledge_handler().delete_file(file_id, user.id)
        return result
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
    Get all files accessible to the current user (files with permissions only).
    
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
        # Get files for current page (async, non-blocking)
        files = await get_knowledge_handler().list_user_files_async(
            user_id=user.id,
            limit=limit,
            offset=offset
        )
        
        # Get total count of files for the user (async, non-blocking)
        total_count = await get_knowledge_handler().count_user_files_async(user.id)
        
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
        result = await get_knowledge_handler().trigger_indexing(request.file_ids, user.id)
        
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


@router.post("/graph/export_async", status_code=status.HTTP_202_ACCEPTED)
async def export_knowledge_graph_async(
    request: GraphExportRequest,
    user: Annotated[User | None, Depends(get_current_user)],
):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    if not _use_celery():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Task queue mode is not enabled")

    run_id = uuid.uuid4().hex
    task_queue = _get_task_queue()
    task_queue.create_task_run(
        task_run_id=run_id,
        task_type="graph_export",
        owner_id=user.id,
        resource_id=str(user.id),
        metadata={
            "executor": "api",
            "max_nodes": int(request.max_nodes),
            "max_edges": int(request.max_edges),
            "include_node_types": request.include_node_types or [],
            "directed_edges": bool(getattr(request, "directed_edges", False)),
            "preserve_multi_edges": bool(getattr(request, "preserve_multi_edges", False)),
        },
    )

    from application.knowledge.celery_tasks import export_graph as export_graph_task

    queue = os.getenv("CELERY_QUEUE_EXPORT") or os.getenv("CELERY_QUEUE_INDEXING", "indexing")
    export_graph_task.apply_async(
        kwargs={
            "owner_id": str(user.id),
            "max_nodes": int(request.max_nodes),
            "max_edges": int(request.max_edges),
            "include_node_types": request.include_node_types,
            "directed_edges": bool(getattr(request, "directed_edges", False)),
            "preserve_multi_edges": bool(getattr(request, "preserve_multi_edges", False)),
        },
        task_id=run_id,
        queue=queue,
    )

    return {
        "run_id": run_id,
        "status": "scheduled",
        "task_run_url": f"/knowledge/task_run/{run_id}",
        "stream_url": f"/knowledge/stream/{run_id}",
        "result_url": f"/knowledge/result/{run_id}",
    }


@router.post("/graph/export", status_code=status.HTTP_200_OK)
async def export_knowledge_graph(
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
        rag_inference = registrator.get_object("rag_inference")
        knowledge_handler = get_knowledge_handler()
        owner_scope = str(user.id)

        if hasattr(knowledge_handler, "_run_blocking"):
            return await knowledge_handler._run_blocking(  # noqa: SLF001
                export_full_graph_payload,
                rag_inference=rag_inference,
                owner_scope=owner_scope,
                max_nodes=int(request.max_nodes),
                max_edges=int(request.max_edges),
                include_node_types=request.include_node_types,
                directed_edges=bool(getattr(request, "directed_edges", False)),
                preserve_multi_edges=bool(getattr(request, "preserve_multi_edges", False)),
            )

        from framework.thread_pool import get_thread_pool

        return await get_thread_pool().run_blocking(
            export_full_graph_payload,
            rag_inference=rag_inference,
            owner_scope=owner_scope,
            max_nodes=int(request.max_nodes),
            max_edges=int(request.max_edges),
            include_node_types=request.include_node_types,
            directed_edges=bool(getattr(request, "directed_edges", False)),
            preserve_multi_edges=bool(getattr(request, "preserve_multi_edges", False)),
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export knowledge graph: {str(e)}",
        )


# ==================== FILE PERMISSION MANAGEMENT ====================

@router.post(
    "/mindmap/export_async",
    status_code=status.HTTP_202_ACCEPTED,
)
async def export_file_mindmap_async(
    request: MindmapExportRequest,
    user: Annotated[User | None, Depends(get_current_user)],
):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    if not request.file_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="file_id is required")
    if not _use_celery():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Task queue mode is not enabled")

    run_id = uuid.uuid4().hex
    task_queue = _get_task_queue()
    task_queue.create_task_run(
        task_run_id=run_id,
        task_type="mindmap_export",
        owner_id=user.id,
        resource_id=request.file_id,
        metadata={"executor": "api"},
    )

    from application.knowledge.celery_tasks import export_mindmap as export_mindmap_task

    queue = os.getenv("CELERY_QUEUE_EXPORT") or os.getenv("CELERY_QUEUE_INDEXING", "indexing")
    export_mindmap_task.apply_async(
        kwargs={"file_id": request.file_id, "owner_id": str(user.id)},
        task_id=run_id,
        queue=queue,
    )

    return {
        "run_id": run_id,
        "status": "scheduled",
        "task_run_url": f"/knowledge/task_run/{run_id}",
        "stream_url": f"/knowledge/stream/{run_id}",
        "result_url": f"/knowledge/result/{run_id}",
    }


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

    knowledge_handler = get_knowledge_handler()
    rag_inference = registrator.get_object("rag_inference")
    payload = await export_file_mindmap_payload(
        knowledge=knowledge_handler,
        rag_inference=rag_inference,
        file_id=request.file_id,
        owner_id=user.id,
    )
    return MindmapExportResponse(
        tsv=str(payload.get("tsv") or ""),
        nodes=[MindmapNode(**node) for node in (payload.get("nodes") or [])],
        edges=[MindmapEdge(**edge) for edge in (payload.get("edges") or [])],
    )


@router.post(
    "/{file_id}/permissions/grant",
    response_model=GrantPermissionResponse,
    status_code=status.HTTP_200_OK,
)
async def grant_file_permission(
    file_id: str,
    request: GrantPermissionRequest,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """
    Grant file permission to a user, department, or all users.
    
    Only users with EDIT permission can grant permissions.
    
    Args:
        file_id: File ID to grant permission for
        request: GrantPermissionRequest with permission details
        user: Current authenticated user (must have EDIT permission)
    
    Returns:
        GrantPermissionResponse with permission ID and message
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    if get_knowledge_handler().check_file_access(file_id, user.id) != PermissionType.EDIT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to grant permissions for this file"
        )

    # Pydantic automatically validates and converts enum types
    receiver_type: PermissionReceiverType = request.receiver_type
    permission_type: PermissionType = request.permission_type

    # Parse and validate required fields based on receiver_type
    receiver_user_id = None
    receiver_department_id = None
    
    if receiver_type == PermissionReceiverType.USER:
        if not request.user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id is required when receiver_type is 'user'"
            )
        try:
            receiver_user_id = uuid.UUID(request.user_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid user_id format: {request.user_id}"
            )
    elif receiver_type == PermissionReceiverType.DEPARTMENT:
        if not request.department_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="department_id is required when receiver_type is 'department'"
            )
        try:
            receiver_department_id = uuid.UUID(request.department_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid department_id format: {request.department_id}"
            )
    # For ALL receiver_type, both user_id and department_id should be None

    try:
        permission_id = get_knowledge_handler().grant_file_permission(
            file_id=file_id,
            receiver_type=receiver_type,
            permission_type=permission_type,
            granted_by=user.id,
            user_id=receiver_user_id,
            department_id=receiver_department_id
        )
        if permission_id:
            return GrantPermissionResponse(
                permission_id=str(permission_id),
                message=f"Permission granted successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to grant permission"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to grant permission: {str(e)}",
        )


@router.delete(
    "/permissions/{permission_id}",
    status_code=status.HTTP_200_OK,
)
async def revoke_file_permission(
    permission_id: str,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """
    Revoke a file permission by permission ID.
    
    Only users with EDIT permission can revoke permissions.
    
    Args:
        permission_id: Permission ID to revoke
        user: Current authenticated user (must have EDIT permission)
    
    Returns:
        Success message
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    try:
        perm_id = uuid.UUID(permission_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid permission_id format: {permission_id}"
        )

    file_id = get_knowledge_handler().get_file_id_by_permission_id(perm_id)
    if not file_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Permission not found"
        )

    # Only users with EDIT permission can revoke permissions
    permission_type = get_knowledge_handler().check_file_access(file_id, user.id)
    if permission_type != PermissionType.EDIT:
        raise HTTPException(status_code=403, detail="Only users with EDIT permission can revoke permissions")

    try:
        get_knowledge_handler().revoke_file_permission(perm_id, user.id)
        return {"message": "Permission revoked successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke permission: {str(e)}",
        )

@router.get(
    "/{file_id}/permissions",
    response_model=PermissionListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_file_permissions(
    file_id: str,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """
    List all permissions for a specific file.
    
    Users with VIEW or EDIT permission can view permissions.
    
    Args:
        file_id: File ID to list permissions for
        user: Current authenticated user (must have VIEW or EDIT permission)
    
    Returns:
        PermissionListResponse with list of permissions
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    # Check if user has VIEW permission to list permissions
    permission_type = get_knowledge_handler().check_file_access(file_id, user.id)
    if permission_type is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to list permissions for this file"
        )
    try:
        permissions = get_knowledge_handler().list_file_permissions(file_id, user.id)
        
        permission_infos = []
        for perm in permissions:
            # Build UserInfo if user relationship is loaded and receiver type is USER
            user_info = None
            if perm.permission_receiver_type == PermissionReceiverType.USER and perm.user:
                # Build DepartmentInfo if user has a department
                user_department_info = None
                if perm.user.department:
                    user_department_info = DepartmentInfo(
                        id=str(perm.user.department.id),
                        name=perm.user.department.name,
                        description=perm.user.department.description,
                        path=perm.user.department.path
                    )
                
                user_info = UserInfo(
                    id=str(perm.user.id),
                    user_name=perm.user.user_name,
                    department=user_department_info,
                    status=perm.user.status.value
                )
            
            # Build DepartmentInfo if department relationship is loaded and receiver type is DEPARTMENT
            department_info = None
            if perm.permission_receiver_type == PermissionReceiverType.DEPARTMENT and perm.department:
                department_info = DepartmentInfo(
                    id=str(perm.department.id),
                    name=perm.department.name,
                    description=perm.department.description,
                    path=perm.department.path
                )
            
            permission_infos.append(
                PermissionInfo(
                    permission_id=str(perm.id),
                    file_id=perm.file_id,
                    receiver_type=perm.permission_receiver_type.value,
                    permission_type=perm.permission_type.value,
                    user=user_info,
                    department=department_info,
                    granted_by=str(perm.granted_by),
                    granted_at=perm.granted_at.isoformat()
                )
            )
        
        return PermissionListResponse(
            permissions=permission_infos,
            total=len(permission_infos)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list permissions: {str(e)}",
        )


@router.get(
    "/{file_id}/permissions/check",
    response_model=CheckAccessResponse,
    status_code=status.HTTP_200_OK,
)
async def check_file_access(
    file_id: str,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """
    Check if the current user has access to a file and return the permission type.
    
    Args:
        file_id: File ID to check
        user: Current authenticated user
    
    Returns:
        CheckAccessResponse with access status and permission type
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    try:
        permission_type = get_knowledge_handler().check_file_access(file_id, user.id)
        
        return CheckAccessResponse(
            has_access=permission_type is not None,
            permission_type=permission_type.value if permission_type else None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check access: {str(e)}",
        )


@router.put(
    "/permissions/{permission_id}",
    status_code=status.HTTP_200_OK,
)
async def update_file_permission(
    permission_id: str,
    user: Annotated[User | None, Depends(get_current_user)],
    permission_type: PermissionType = Body(..., embed=True, description="New permission type: 'view' or 'edit'"),
):
    """
    Update an existing file permission.
    
    Only users with EDIT permission can update permissions.
    
    Args:
        permission_id: Permission ID to update
        permission_type: New permission type ('view' or 'edit')
        user: Current authenticated user (must have EDIT permission)
    
    Returns:
        Success message
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    try:
        perm_id = uuid.UUID(permission_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid permission_id format: {permission_id}"
        )

    # Get file_id first to check ownership
    file_id = get_knowledge_handler().get_file_id_by_permission_id(perm_id)
    if not file_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Permission not found"
        )

    # Only file editor can update permissions
    if get_knowledge_handler().check_file_access(file_id, user.id) != PermissionType.EDIT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to update permissions for this file"
        )

    # Pydantic automatically validates and converts enum types
    try:
        result = get_knowledge_handler().update_file_permission(
            permission_id=perm_id,
            permission_type=permission_type,
            user_id=user.id
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Permission not found"
            )
        
        return {"message": "Permission updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update permission: {str(e)}",
        )
