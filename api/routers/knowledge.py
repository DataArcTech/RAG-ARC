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
from datetime import datetime, timezone, timedelta
from api.routers.auth import get_current_user
from api.sse import sse_done, sse_json
from encapsulation.data_model.orm_models import (
    User,
    Department,
    FilePermission,
    PermissionReceiverType,
    PermissionType,
    FileStatus,
)
from framework.register import Register
from application.knowledge.module import Knowledge
from application.account.user import Account
from core.file_management.storage.file import FileValidationError
from core.utils.owner_guard import is_admin_owner
from core.utils.path_guard import safe_leaf_name
from encapsulation.message_queue.redis_task_queue import TaskState
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
    ParseTriggerRequest,
    ParseTriggerResponse,
    KnowledgeChunkResponse,
    MindmapEdge,
    MindmapExportRequest,
    MindmapExportResponse,
    MindmapNode,
    KGMaintenanceL2Request,
    KGMaintenanceL2Response,
    PermissionInfo,
    PermissionListResponse,
    QueueStatusResponse,
    RevokePermissionRequest,
    TaskRunStatusResponse,
    UserInfo,
)
from api.routers.knowledge_streaming import stream_events_redis
from api.routers.knowledge_utils import (
    assert_task_owner,
    format_sse,
    get_task_queue,
    normalize_file_id,
    use_celery,
)
from core.presentation.evidence import document_download_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/knowledge", tags=["files"])

registrator = Register()

def get_account_handler() -> Account:
    """Lazy loading function to get account handler after initialization."""
    return registrator.get_object("account")

def get_knowledge_handler() -> Knowledge:
    """Lazy loading function to get knowledge handler after initialization."""
    return registrator.get_object("knowledge")
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
    ingest_mode: Optional[str] = Form(
        default="index",
        description="Upload mode: index (default), parse (parse-only), store (store-only).",
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
    
    # Validate file type
    allowed_extensions = {'.docx', '.xlsx', '.pptx', '.pdf', '.jpg', '.jpeg', '.png', '.txt', '.html', '.md'}
    file_ext = Path(file.filename).suffix.lower() if file.filename else ''
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"不支持的文件类型。支持的文件类型: {', '.join(sorted(allowed_extensions))}"
        )
    
    # Validate file size (10MB limit)
    max_size = 10 * 1024 * 1024  # 10MB
    file_content = await file.read()
    if len(file_content) > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="该文件超过10MB，请检查后重新上传！"
        )
    
    # Reset file pointer for actual upload
    await file.seek(0)
    
    try:
        # Convert string UUID to UUID object
        doc_id = await get_knowledge_handler().upload_file(
            file,
            user.id,
            relative_path=relative_path,
            ingest_mode=str(ingest_mode or "index"),
        )
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
    if not use_celery():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Task queue mode is not enabled")
    task_queue = get_task_queue()
    task_run = await asyncio.to_thread(task_queue.get_task_run, run_id)
    if not task_run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id not found")
    assert_task_owner(task_run, user_id=user.id)
    return TaskRunStatusResponse(
        run_id=run_id,
        task_type=task_run.get("task_type"),
        state=task_run.get("state"),
        progress_percent=task_run.get("progress_percent"),
        error_message=task_run.get("error_message"),
        resource_id=task_run.get("resource_id"),
        updated_at_ms=task_run.get("updated_at_ms"),
    )


@router.get("/queue_status", response_model=QueueStatusResponse, status_code=status.HTTP_200_OK)
async def get_queue_status(
    user: Annotated[User | None, Depends(get_current_user)],
    limit: int = Query(default=30, ge=1, le=200, description="RedisTaskQueue 最近任务条数"),
):
    """调试用：查询 Celery 与 RedisTaskQueue 的队列/任务状态。"""
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    task_queue = get_task_queue()

    celery_data: Dict[str, Any] | None = None
    if use_celery():
        try:
            from encapsulation.message_queue.celery_app import app as celery_app

            def _inspect() -> Dict[str, Any]:
                out: Dict[str, Any] = {}
                insp = celery_app.control.inspect()
                out["active"] = insp.active() or {}
                out["reserved"] = insp.reserved() or {}
                out["scheduled"] = insp.scheduled() or {}
                out["queue_lengths"] = task_queue.get_broker_queue_lengths()
                return out

            celery_data = await asyncio.to_thread(_inspect)
        except Exception as exc:
            celery_data = {"error": str(exc)}

    redis_data: Dict[str, Any] = {}
    try:
        recent = await asyncio.to_thread(task_queue.list_recent_task_runs, limit=limit)
        redis_data["recent_task_runs"] = recent
        # 按 state 聚合计数
        by_state: Dict[str, int] = {}
        for r in recent:
            s = str(r.get("state") or "unknown")
            by_state[s] = by_state.get(s, 0) + 1
        redis_data["by_state"] = by_state
    except Exception as exc:
        redis_data["error"] = str(exc)

    return QueueStatusResponse(celery=celery_data, redis_task_queue=redis_data)


@router.get("/result/{run_id}", status_code=status.HTTP_200_OK)
async def get_task_result(
    run_id: str,
    user: Annotated[User | None, Depends(get_current_user)],
):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    if not use_celery():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Task queue mode is not enabled")
    task_queue = get_task_queue()
    task_run = await asyncio.to_thread(task_queue.get_task_run, run_id)
    if not task_run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id not found")
    assert_task_owner(task_run, user_id=user.id)
    state = str(task_run.get("state") or "")
    if state not in {TaskState.SUCCESS.value, TaskState.FAILURE.value, TaskState.CANCELED.value}:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="run not finished")
    if state != TaskState.SUCCESS.value:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(task_run.get("error_message") or "task failed"),
        )
    result = await asyncio.to_thread(task_queue.get_task_result, run_id)
    return result or {"run_id": run_id, "done": True, "result": None}


@router.post("/kg/maintenance/l2", response_model=KGMaintenanceL2Response, status_code=status.HTTP_200_OK)
async def kg_maintenance_l2(
    req: KGMaintenanceL2Request = Body(...),
    user: Annotated[User | None, Depends(get_current_user)] = None,
):
    """Run L2 KG maintenance (repair) for the current owner scope.

    This endpoint is intended for debugging/admin maintenance and is owner-scoped by default.
    """
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")

    # For now, only allow users to run maintenance on their own scope (admin can impersonate by calling with admin token).
    owner_id = user.id
    try:
        stats = await get_knowledge_handler().run_kg_maintenance_l2(
            owner_id=owner_id,
            file_ids=req.file_ids,
            rebuild_same_as=req.rebuild_same_as,
            backfill_missing_mentions=req.backfill_missing_mentions,
        )
        return KGMaintenanceL2Response(owner_id=str(owner_id), stats=stats)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@router.get("/stream/{run_id}")
async def stream_task_progress(
    run_id: str,
    user: Annotated[User | None, Depends(get_current_user)],
    last_event_id: int = -1,
):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    if not use_celery():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Task queue mode is not enabled")
    task_queue = get_task_queue()
    task_run = task_queue.get_task_run(run_id)
    if not task_run:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run_id not found")
    assert_task_owner(task_run, user_id=user.id)
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        stream_events_redis(task_queue, format_sse=format_sse, run_id=run_id, last_event_id=last_event_id),
        media_type="text/event-stream",
        headers=headers,
    )


@router.get("/{file_id}/download")
async def download_file(file_id: str):
    """Download a file by file_id.
    
    Note: Authentication and permission checks are disabled for this endpoint.
    """
    try:
        # Log original file_id to debug URL construction issues
        if len(file_id) > 36:
            logger.warning(f"Received malformed file_id (length={len(file_id)}): {file_id[:50]}...")
        # Normalize file_id to handle duplicated UUIDs in URL
        normalized_file_id = normalize_file_id(file_id)
        if normalized_file_id != file_id:
            logger.info(f"Normalized file_id from {file_id[:50]}... to {normalized_file_id}")
        return await get_knowledge_handler().get_file(normalized_file_id, None)
    except HTTPException:
        # re-raise 404s from underlying module
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download file: {str(e)}",
        )


@router.get("/chunk/{chunk_id}", response_model=KnowledgeChunkResponse, status_code=status.HTTP_200_OK)
async def get_chunk(
    chunk_id: str,
):
    """Fetch a single indexed chunk (evidence) by chunk_id for citation inspection.
    
    Note: Authentication and permission checks are disabled for this endpoint.
    """
    try:
        rag_inference = registrator.get_object("rag_inference")
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="rag_inference not initialized") from exc

    graph_store = getattr(rag_inference, "get_graph_store", None)
    graph_store = graph_store() if callable(graph_store) else None
    if graph_store is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="graph store not available")

    from framework.thread_pool import get_thread_pool

    try:
        chunks = await get_thread_pool().run_blocking(graph_store.get_by_ids, [str(chunk_id)])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to fetch chunk: {exc}") from exc

    chunk_obj = None
    if isinstance(chunks, list) and chunks:
        chunk_obj = chunks[0]
    if chunk_obj is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="chunk not found")

    metadata = dict(getattr(chunk_obj, "metadata", None) or {})
    file_id = str(metadata.get("source_file_id") or "").strip() or None
    if not file_id:
        nested = metadata.get("chunk_metadata") if isinstance(metadata.get("chunk_metadata"), dict) else {}
        file_id = str(nested.get("source_file_id") or "").strip() or None

    # Permission check disabled - all chunks are accessible without authentication

    filename = None
    if file_id:
        try:
            meta = await asyncio.to_thread(get_knowledge_handler().file_storage.get_file_metadata, file_id)
            filename = getattr(meta, "filename", None) if meta is not None else None
        except Exception:  # noqa: BLE001
            filename = None

    content = metadata.get("prompt_text") or metadata.get("index_text") or getattr(chunk_obj, "content", "") or ""
    return KnowledgeChunkResponse(
        chunk_id=str(getattr(chunk_obj, "id", None) or chunk_id),
        content=str(content),
        file_id=file_id,
        filename=str(filename) if filename else None,
        document_url=document_download_url(file_id) if file_id else None,
        metadata=metadata,
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
async def delete_file(
    file_id: str,
    user: Annotated[User | None, Depends(get_current_user)],
    purge: bool = Query(
        default=False,
        description=(
            "When true, purge underlying file bytes + parsed content. "
            "Default (false) only removes derived artifacts (chunks/indexes) in the owner scope and keeps parsed artifacts for reuse."
        ),
    ),
):
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    try:
        result = await get_knowledge_handler().delete_file(file_id, user.id, purge=purge)
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
    page: Optional[int] = Query(default=1, ge=1, description="Page number (starts from 1)"),
    pagesize: Optional[int] = Query(default=100, ge=1, le=1000, description="Number of files per page"),
    search: Optional[str] = Query(default=None, description="Search keyword for filename (fuzzy match)"),
    status: Optional[str] = Query(default=None, description="Filter by file status (STORED, PARSED, CHUNKED, PARTIAL_INDEXED, INDEXED, FAILED, DELETED)"),
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
        page: Page number (starts from 1, default: 1)
        pagesize: Number of files per page (default: 100, max: 1000)
        search: Optional search keyword for filename fuzzy matching
        status: Optional filter by file status (STORED, PARSED, CHUNKED, PARTIAL_INDEXED, INDEXED, FAILED, DELETED)
        
    Returns:
        FileListResponse with list of files and total count
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    try:
        # Parse status parameter if provided
        status_enum = None
        if status:
            try:
                status_enum = FileStatus[status.upper()]
            except KeyError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status value: {status}. Valid values are: STORED, PARSED, CHUNKED, PARTIAL_INDEXED, INDEXED, FAILED, DELETED"
                )
        
        # Calculate offset from page number
        # page starts from 1, so:
        # page=1 → offset=0 (first page)
        # page=2 → offset=pagesize (second page)
        # page=3 → offset=2*pagesize (third page)
        offset = (page - 1) * pagesize
        
        # Log pagination parameters for debugging
        logger.debug(f"List files pagination: page={page}, pagesize={pagesize}, offset={offset}, status={status}")
        
        # Get files for current page (async, non-blocking)
        files = await get_knowledge_handler().list_user_files_async(
            user_id=user.id,
            limit=pagesize,
            offset=offset,
            search=search,
            status=status_enum
        )
        
        # Get total count of files for the user (async, non-blocking)
        total_count = await get_knowledge_handler().count_user_files_async(user.id, search=search, status=status_enum)
        
        # Log results for debugging
        logger.debug(f"List files result: returned {len(files)} files, total_count={total_count}, page={page}, pagesize={pagesize}")
        
        # Convert FileMetadata objects to FileInfo response models
        # Database stores naive datetime in Beijing time (since we store with Beijing timezone)
        # Just add timezone info for ISO format, no conversion needed
        beijing_tz = timezone(timedelta(hours=8))
        file_infos = []
        for file in files:
            # Database stores naive datetime in Beijing time, just add timezone info
            if file.created_at.tzinfo is None:
                # Naive datetime, assume it's already Beijing time, just add timezone info
                created_at_beijing = file.created_at.replace(tzinfo=beijing_tz)
            else:
                # Timezone-aware datetime, convert to Beijing time
                created_at_beijing = file.created_at.astimezone(beijing_tz)
            
            if file.updated_at.tzinfo is None:
                # Naive datetime, assume it's already Beijing time, just add timezone info
                updated_at_beijing = file.updated_at.replace(tzinfo=beijing_tz)
            else:
                # Timezone-aware datetime, convert to Beijing time
                updated_at_beijing = file.updated_at.astimezone(beijing_tz)
            
            # 对前端统一将“部分索引”视为失败，返回 FAILED
            status_for_frontend = "FAILED" if file.status == FileStatus.PARTIAL_INDEXED else file.status.value
            file_infos.append(
                FileInfo(
                    file_id=file.file_id,
                    filename=file.filename,
                    status=status_for_frontend,  # Convert enum to string; PARTIAL_INDEXED → FAILED
                    created_at=created_at_beijing.isoformat(),
                    updated_at=updated_at_beijing.isoformat(),
                    file_size=file.file_size,
                    content_type=file.content_type
                )
            )
        
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


@router.post(
    "/trigger_parsing",
    response_model=ParseTriggerResponse,
    status_code=status.HTTP_200_OK,
)
async def trigger_parsing(
    request: ParseTriggerRequest,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """Trigger parse-only for multiple files (no chunk/index)."""
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    if not request.file_ids:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="file_ids list cannot be empty")
    try:
        result = await get_knowledge_handler().trigger_parsing(
            request.file_ids,
            user.id,
            force_reparse=bool(getattr(request, "force_reparse", False)),
        )
        return ParseTriggerResponse(message=result)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to trigger parsing: {e}")


@router.post(
    "/{file_id}/parse",
    response_model=ParseTriggerResponse,
    status_code=status.HTTP_200_OK,
)
async def parse_single_file(
    file_id: str,
    user: Annotated[User | None, Depends(get_current_user)],
    force_reparse: bool = False,
):
    """Trigger parse-only for a single file."""
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    try:
        result = await get_knowledge_handler().trigger_parsing([file_id], user.id, force_reparse=bool(force_reparse))
        return ParseTriggerResponse(message=result)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to trigger parsing: {e}")


@router.post(
    "/retry_indexing",
    response_model=IndexTriggerResponse,
    status_code=status.HTTP_200_OK,
)
async def retry_indexing(
    request: IndexTriggerRequest,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """
    Retry indexing for failed files.

    This endpoint is specifically designed for retrying failed file indexing.
    It accepts a single file_id or a list of file_ids and triggers re-indexing.

    Args:
        request: IndexTriggerRequest containing list of file IDs to retry
        user: Current authenticated user

    Returns:
        IndexTriggerResponse with retry results
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
        # Use the existing trigger_indexing method which already supports FAILED files
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
            detail=f"Failed to retry indexing: {str(e)}",
        )


@router.post(
    "/{file_id}/retry_indexing",
    response_model=IndexTriggerResponse,
    status_code=status.HTTP_200_OK,
)
async def retry_indexing_single(
    file_id: str,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """
    Retry indexing for a single failed file.

    This is a convenience endpoint for retrying a single file by file_id.
    It's equivalent to POST /retry_indexing with a single file_id in the list.

    Args:
        file_id: The file ID to retry indexing
        user: Current authenticated user

    Returns:
        IndexTriggerResponse with retry results
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    try:
        # Use the existing trigger_indexing method which already supports FAILED files
        result = await get_knowledge_handler().trigger_indexing([file_id], user.id)
        
        return IndexTriggerResponse(
            message=result
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (404, 403)
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retry indexing: {str(e)}",
        )


@router.post("/graph/export_async", status_code=status.HTTP_202_ACCEPTED)
async def export_knowledge_graph_async(
    request: GraphExportRequest,
    user: Annotated[User | None, Depends(get_current_user)],
):
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")
    if not use_celery():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Task queue mode is not enabled")

    run_id = uuid.uuid4().hex
    task_queue = get_task_queue()
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
    if not use_celery():
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Task queue mode is not enabled")

    run_id = uuid.uuid4().hex
    task_queue = get_task_queue()
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
