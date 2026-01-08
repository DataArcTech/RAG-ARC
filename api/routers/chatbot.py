import asyncio
import json
import logging
import os
import time
import uuid
import threading
import mimetypes
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import quote

import anyio
from fastapi import APIRouter, HTTPException, Request, Response, status, Depends
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import Annotated

from framework.register import Register
from api.routers.auth import get_current_user, get_current_user_optional
from encapsulation.data_model.orm_models import User
from core.presentation.evidence import build_chat_evidence
from config.output_limits import CHAT_TOP_CHUNKS
from core.utils.path_guard import ensure_writable_dir
from api.utils.owner_scope import resolve_default_owner_id

logger = logging.getLogger(__name__)

router = APIRouter(tags=["chatbot"])

_COOKIE_NAME = "rag_arc_uid"
_registrator = Register()

_conversation_locks: Dict[str, asyncio.Lock] = {}
_conversation_last_used: Dict[str, float] = {}
_locks_guard = asyncio.Lock()
_global_semaphore = asyncio.Semaphore(int(os.getenv("CHATBOT_MAX_CONCURRENCY", "8")))

_CHATBOT_SYSTEM_PROMPT_V2 = (
    "You are a helpful RAG assistant.\n"
    "You may be given a list of numbered Sources (key=1..N).\n"
    "Rules:\n"
    "1) If the user message is just a greeting / test / acknowledgement (e.g. '测试', 'test', 'hello', 'hi', '你好'),\n"
    "   answer briefly and DO NOT use any Sources and DO NOT include any <sup> tags.\n"
    "2) Otherwise, if Sources are provided, ground your answer in Sources and add inline citations using HTML <sup> tags.\n"
    "   - Every sentence that contains factual information supported by Sources MUST end with one or more <sup>key</sup>.\n"
    "   - Cite only the minimal number of sources needed; do NOT cite all sources by default.\n"
    "   - Do NOT output a bare block/list of citations (e.g. '<sup>1</sup><sup>2</sup>...') without nearby supporting text.\n"
    "   - Do NOT cite a source you did not use.\n"
    "3) If Sources are provided but none are relevant, say you don't know based on the provided Sources and ask a clarifying question.\n"
    "4) Do NOT use bracket citations like [1] and do NOT add a trailing 'Sources:' section.\n"
    "5) Output in Markdown. The only HTML allowed is <sup>...</sup>.\n"
)


class ChatbotBootstrapCapabilities(BaseModel):
    streaming: bool = True
    evidence: bool = True


class ChatbotBootstrapResponse(BaseModel):
    browser_user_id: str
    cookie_name: str
    server_time: str
    capabilities: ChatbotBootstrapCapabilities


class ChatbotApiMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)


class ChatbotApiMessagesRequest(BaseModel):
    id: str = Field(min_length=1, description="conversation_id (uuid)")
    content: str = Field(min_length=1, description="current user message")
    messages: List[ChatbotApiMessage] = Field(default_factory=list, description="full conversation history")
    stream: bool = True


class ChatbotSourceItem(BaseModel):
    key: int
    chunk_id: Optional[str] = None
    file_id: Optional[str] = None
    title: str
    file: Optional[str] = None
    description: str


def _parse_uuid(value: str, field_name: str) -> uuid.UUID:
    try:
        return uuid.UUID(str(value))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"{field_name} must be a valid UUID") from exc


def _ensure_browser_user_id(request: Request) -> tuple[uuid.UUID, bool]:
    raw = (
        request.headers.get("x-owner-id")
        or request.headers.get("x-browser-user-id")
        or request.query_params.get("owner_id")
        or request.cookies.get(_COOKIE_NAME)
    )
    if raw:
        return _parse_uuid(raw, _COOKIE_NAME), request.cookies.get(_COOKIE_NAME) is None

    new_id = uuid.uuid4()
    return new_id, True


def _require_browser_user_id(request: Request) -> uuid.UUID:
    raw = (
        request.headers.get("x-owner-id")
        or request.headers.get("x-browser-user-id")
        or request.query_params.get("owner_id")
        or request.cookies.get(_COOKIE_NAME)
    )
    if not raw:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing browser identity")
    return _parse_uuid(raw, _COOKIE_NAME)


def _apply_browser_cookie(response: Response, browser_user_id: uuid.UUID) -> None:
    response.set_cookie(
        key=_COOKIE_NAME,
        value=str(browser_user_id),
        path="/",
        httponly=os.getenv("CHATBOT_COOKIE_HTTPONLY", "0") == "1",
        samesite="lax",
    )


def _resolve_config_path(path_value: str) -> str:
    path = Path(path_value)
    if path.is_absolute():
        return str(path)
    return str((Path(__file__).resolve().parents[2] / path_value).resolve())


def _ensure_registered(app_name: str, config_env: str, default_app_name: str, config_type: Any) -> Any:
    config_path = os.getenv(config_env)
    if not config_path:
        return _registrator.get_object(default_app_name)

    if app_name in _registrator.registrations:
        return _registrator.get_object(app_name)

    resolved = _resolve_config_path(config_path)
    _registrator.register(config_path=resolved, app_name=app_name, config_type=config_type)
    return _registrator.get_object(app_name)


def _get_chatbot_rag_inference():
    from config.application.rag_inference_config import RAGInferenceConfig

    return _ensure_registered(
        "chatbot_rag_inference",
        "CHATBOT_RAG_INFERENCE_CONFIG_PATH",
        "rag_inference",
        RAGInferenceConfig,
    )


def _get_chatbot_knowledge():
    from config.application.knowledge_config import KnowledgeConfig

    return _ensure_registered(
        "chatbot_knowledge",
        "CHATBOT_KNOWLEDGE_CONFIG_PATH",
        "knowledge",
        KnowledgeConfig,
    )


def _reset_chatbot_modules_for_tests() -> None:
    _registrator.registrations.pop("chatbot_rag_inference", None)
    _registrator.registrations.pop("chatbot_knowledge", None)


async def _get_conversation_lock(conversation_id: str) -> asyncio.Lock:
    now = time.time()
    async with _locks_guard:
        lock = _conversation_locks.get(conversation_id)
        if lock is None:
            lock = asyncio.Lock()
            _conversation_locks[conversation_id] = lock
        _conversation_last_used[conversation_id] = now

        ttl_s = int(os.getenv("CHATBOT_LOCK_TTL_S", "1800"))
        if ttl_s > 0 and len(_conversation_last_used) > 256:
            cutoff = now - ttl_s
            expired = [key for key, last in _conversation_last_used.items() if last < cutoff]
            for key in expired:
                _conversation_last_used.pop(key, None)
                _conversation_locks.pop(key, None)
        return lock


def _conversation_key(browser_user_id: uuid.UUID, conversation_id: uuid.UUID) -> str:
    return f"{browser_user_id}:{conversation_id}"


def _estimate_tokens(text: str) -> tuple[int, str]:
    if not text:
        return 0, "heuristic"
    ascii_count = sum(1 for ch in text if ord(ch) < 128)
    ratio = ascii_count / max(len(text), 1)
    if ratio > 0.8:
        return max(int(len(text) / 4), 1), "heuristic(ascii/4)"
    return len(text), "heuristic(chars)"

def _estimate_tokens_for_messages(messages: List[Dict[str, str]]) -> tuple[int, str]:
    texts: List[str] = []
    for msg in messages or []:
        content = (msg.get("content") or "").strip()
        if content:
            texts.append(content)
    combined = "\n".join(texts).strip()
    return _estimate_tokens(combined)


def _build_llm_messages(
    system_message: Dict[str, str],
    sources: List[Dict[str, str]],
    history: List[Dict[str, str]],
    user_message: str,
) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [system_message]
    msgs.extend(history or [])
    msgs.extend(sources or [])
    msgs.append({"role": "user", "content": user_message})
    return msgs


def _ensure_context_within_limit(
    messages: List[Dict[str, str]],
    *,
    max_context_tokens: int,
    threshold_fraction: float,
    token_estimator_suffix: str = "window_disabled",
) -> Dict[str, Any]:
    estimated, estimator = _estimate_tokens_for_messages(messages)
    allowed = int(max_context_tokens * threshold_fraction)
    if estimated > allowed:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail={
                "code": "context_too_long",
                "max_context_tokens": max_context_tokens,
                "threshold_fraction": threshold_fraction,
                "estimated_context_tokens": estimated,
                "suggestion": "start_new_conversation",
            },
        )
    return {
        "max_context_tokens": max_context_tokens,
        "threshold_fraction": threshold_fraction,
        "estimated_context_tokens": estimated,
        "token_estimator": f"{estimator}|{token_estimator_suffix}",
    }


def _sanitize_title(value: str) -> str:
    title = (value or "").strip()
    title = title.splitlines()[0].strip() if title else ""
    title = title.strip("“”\"'` ")
    title = " ".join(title.split())
    if len(title) > 80:
        title = title[:80].rstrip()
    return title


def _fallback_title(user_text: str) -> str:
    fallback = " ".join((user_text or "").strip().split())
    fallback = fallback[:40].rstrip()
    return fallback or "New conversation"


def _generate_title_messages(user_text: str, assistant_text: str) -> List[Dict[str, str]]:
    prompt_user = (user_text or "").strip()[:800]
    prompt_assistant = (assistant_text or "").strip()[:800]
    content = f"User: {prompt_user}\nAssistant: {prompt_assistant}\n\nReturn title only."
    return [
        {
            "role": "system",
            "content": (
                "You generate short conversation titles.\n"
                "- Output only the title.\n"
                "- Max 8 words or 20 Chinese characters.\n"
                "- No quotes, no punctuation at the end."
            ),
        },
        {"role": "user", "content": content},
    ]


async def _generate_title_via_llm(
    user_text: str,
    assistant_text: str,
) -> str:
    rag_inference_handler = _get_chatbot_rag_inference()
    llm = getattr(rag_inference_handler, "llm", None)
    if llm is None:
        return _fallback_title(user_text)

    messages = _generate_title_messages(user_text, assistant_text)

    def _run():
        return llm.chat(messages)

    try:
        raw = await anyio.to_thread.run_sync(_run)
    except Exception:  # noqa: BLE001
        return _fallback_title(user_text)

    title = _sanitize_title(str(raw or ""))
    return title or _fallback_title(user_text)


def _normalize_history(messages: List[ChatbotApiMessage], *, turns: int) -> List[Dict[str, str]]:
    turns = max(int(turns or 0), 0)
    if turns <= 0:
        return []
    max_messages = turns * 2
    tail = list(messages or [])[-max_messages:] if len(messages or []) > max_messages else list(messages or [])
    return [{"role": m.role, "content": m.content} for m in tail]


def _build_source_messages_v2(sources: List[ChatbotSourceItem]) -> List[Dict[str, str]]:
    if not sources:
        return []
    messages: List[Dict[str, str]] = []
    for item in sources:
        header = f"Source key={item.key} title={item.title}"
        if item.file:
            header += f" file={item.file}"
        messages.append({"role": "user", "content": f"{header}\n{item.description}"})
    return messages


def _sse_json(data: Dict[str, Any]) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


def _sse_done() -> str:
    return "data: [DONE]\n\n"


_SUP_KEY_RE = re.compile(r"<sup>\s*(?P<key>\d{1,4})\s*</sup>")


def _extract_sup_keys(text: str) -> List[int]:
    """Extract referenced source keys in the order they appear."""
    if not text:
        return []
    seen: set[int] = set()
    keys: List[int] = []
    for match in _SUP_KEY_RE.finditer(text):
        try:
            key = int(match.group("key"))
        except Exception:  # noqa: BLE001
            continue
        if key <= 0:
            continue
        if key not in seen:
            seen.add(key)
            keys.append(key)
    return keys


def _filter_sources_by_sup_keys(sources: List[ChatbotSourceItem], answer_text: str) -> List[ChatbotSourceItem]:
    """
    Source payload should be citation-driven:
    - If LLM didn't output any <sup>key</sup>, return no sources.
    - Otherwise return only the referenced sources (by key).
    """
    if not sources:
        return []
    used_keys = set(_extract_sup_keys(answer_text))
    if not used_keys:
        return []
    return [s for s in sources if s.key in used_keys]

def _guess_media_type(filename: str | None, fallback: str = "application/octet-stream") -> str:
    if filename:
        guessed, _ = mimetypes.guess_type(filename)
        if guessed:
            return guessed
    return fallback


def _content_disposition_inline(filename: str) -> str:
    """Generate Content-Disposition header with proper encoding for non-ASCII filenames.
    
    Uses RFC 2231 format (filename*=UTF-8''encoded) for non-ASCII characters,
    with a fallback for ASCII-only filenames.
    """
    from urllib.parse import quote
    
    safe = (Path(filename).name or "file").replace('"', "")
    
    # Check if filename contains non-ASCII characters
    try:
        safe.encode('latin-1')
        # ASCII-only filename, use simple format
        return f'inline; filename="{safe}"'
    except UnicodeEncodeError:
        # Contains non-ASCII characters, use RFC 2231 format
        encoded = quote(safe, safe='')
        return f"inline; filename*=UTF-8''{encoded}"


def _resolve_local_file_path(blob_key: str) -> Path:
    preferred = os.getenv("LOCAL_FILE_STORAGE_PATH", "./data/files")
    runtime_root = os.getenv("RAGARC_RUNTIME_DIR", "./local/runtime")
    fallback = os.path.join(runtime_root, "files")
    base_dir = Path(ensure_writable_dir(preferred, fallback))
    safe_key = (blob_key or "").replace("..", "").lstrip("/")
    return base_dir / safe_key


def _localdb_path_from_instance(local_db: Any, blob_key: str) -> Path | None:
    get_full_path = getattr(local_db, "_get_full_path", None)
    if callable(get_full_path):
        try:
            return Path(get_full_path(blob_key))
        except Exception:  # noqa: BLE001
            return None
    return None


async def _stream_minio_object(minio_db: Any, blob_key: str, chunk_size: int = 1024 * 256):
    response = None
    try:
        client = getattr(minio_db, "client", None)
        config = getattr(minio_db, "config", None)
        bucket = getattr(config, "bucket_name", None) if config is not None else None
        if client is None or not bucket:
            raise RuntimeError("minio client or bucket not configured")
        response = client.get_object(bucket, blob_key)
        while True:
            data = response.read(chunk_size)
            if not data:
                break
            yield data
    finally:
        if response is not None:
            try:
                response.close()
                response.release_conn()
            except Exception:  # noqa: BLE001
                pass


def _build_sources_for_frontend(entries: List[Dict[str, Any]], max_sources: int) -> List[ChatbotSourceItem]:
    max_chars = int(os.getenv("CHATBOT_SOURCE_MAX_CHARS", "1600"))
    sources: List[ChatbotSourceItem] = []
    seen_chunk_ids: set[str] = set()

    knowledge = None
    file_storage = None
    file_storage_resolved = False

    for entry in entries or []:
        if len(sources) >= max_sources:
            break
        chunk_id = str(entry.get("id") or "").strip()
        if not chunk_id or chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)

        metadata = dict(entry.get("metadata") or {})
        file_id = str(metadata.get("source_file_id") or "").strip() or None
        filename = str(metadata.get("filename") or "").strip() or "source"

        content = str(entry.get("content") or "").strip()
        if max_chars > 0 and len(content) > max_chars:
            content = f"{content[:max_chars].rstrip()}..."

        file_url = None
        if file_id and not file_storage_resolved:
            file_storage_resolved = True
            try:
                knowledge = _get_chatbot_knowledge()
                file_storage = getattr(knowledge, "file_storage", None)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to resolve knowledge file_storage; skipping source file URLs")
                file_storage = None

        if file_id and file_storage is not None:
            try:
                meta = file_storage.get_file_metadata(file_id)
                meta_filename = getattr(meta, "filename", None) if meta is not None else None
                safe_name = Path((meta_filename or filename or "source")).name
                file_url = f"/static/files/{file_id}/{quote(safe_name)}"
            except Exception:  # noqa: BLE001
                file_url = None

        sources.append(
            ChatbotSourceItem(
                key=len(sources) + 1,
                chunk_id=chunk_id,
                file_id=file_id,
                title=filename,
                file=file_url,
                description=content,
            )
        )
    return sources


@router.get("/static/files/{file_id}")
async def get_static_file_redirect(
    file_id: str,
    current_user: Annotated[User | None, Depends(get_current_user_optional)] = None,
):
    logger.info(f"[FILE_ACCESS_REDIRECT] Starting file redirect: file_id={file_id}, user={current_user.id if current_user else None}, user_type={getattr(current_user, 'type', None) if current_user else None}")
    
    knowledge = _get_chatbot_knowledge()
    file_storage = getattr(knowledge, "file_storage", None)
    if file_storage is None:
        logger.error(f"[FILE_ACCESS_REDIRECT] File storage not configured: file_id={file_id}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="file storage not configured")

    logger.debug(f"[FILE_ACCESS_REDIRECT] Querying file metadata: file_id={file_id}")
    meta = await anyio.to_thread.run_sync(file_storage.get_file_metadata, file_id)
    if meta is None:
        logger.warning(f"[FILE_ACCESS_REDIRECT] File metadata not found: file_id={file_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="file not found")
    
    logger.debug(f"[FILE_ACCESS_REDIRECT] File metadata found: file_id={file_id}, owner_id={getattr(meta, 'owner_id', None)}, filename={getattr(meta, 'filename', None)}")
    
    # 临时方案：暂时跳过所有权限检查，允许所有用户（包括未认证用户）访问文件
    # TODO: 后续添加权限管理
    logger.info(f"[FILE_ACCESS_REDIRECT] File access allowed (no permission check): file_id={file_id}, user={current_user.id if current_user else None}, user_type={getattr(current_user, 'type', None) if current_user else None}")

    filename = getattr(meta, "filename", None) or "file"
    safe_name = Path(filename).name
    redirect_url = f"/static/files/{file_id}/{quote(safe_name)}"
    logger.info(f"[FILE_ACCESS_REDIRECT] Redirecting: file_id={file_id}, filename={filename}, safe_name={safe_name}, redirect_url={redirect_url}")
    return RedirectResponse(url=redirect_url, status_code=status.HTTP_307_TEMPORARY_REDIRECT)


@router.get("/static/files/{file_id}/{_filename:path}")
async def get_static_file(
    file_id: str,
    _filename: str,
    current_user: Annotated[User | None, Depends(get_current_user_optional)] = None,
):
    logger.info(f"[FILE_ACCESS] Starting file access: file_id={file_id}, filename={_filename}, user={current_user.id if current_user else None}, user_type={getattr(current_user, 'type', None) if current_user else None}")
    
    knowledge = _get_chatbot_knowledge()
    file_storage = getattr(knowledge, "file_storage", None)
    if file_storage is None:
        logger.error(f"[FILE_ACCESS] File storage not configured: file_id={file_id}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="file storage not configured")

    logger.debug(f"[FILE_ACCESS] Querying file metadata: file_id={file_id}")
    meta = await anyio.to_thread.run_sync(file_storage.get_file_metadata, file_id)
    if meta is None:
        logger.warning(f"[FILE_ACCESS] File metadata not found: file_id={file_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="file not found")
    
    logger.debug(f"[FILE_ACCESS] File metadata found: file_id={file_id}, owner_id={getattr(meta, 'owner_id', None)}, filename={getattr(meta, 'filename', None)}, blob_key={getattr(meta, 'blob_key', None)}")
    
    # 临时方案：暂时跳过所有权限检查，允许所有用户（包括未认证用户）访问文件
    # TODO: 后续添加权限管理
    logger.info(f"[FILE_ACCESS] File access allowed (no permission check): file_id={file_id}, user={current_user.id if current_user else None}, user_type={getattr(current_user, 'type', None) if current_user else None}")

    blob_key = getattr(meta, "blob_key", None)
    if not blob_key:
        logger.warning(f"[FILE_ACCESS] File access failed: blob_key is None for file_id={file_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="file not found")
    
    logger.debug(f"[FILE_ACCESS] Blob key retrieved: file_id={file_id}, blob_key={blob_key}")

    filename = getattr(meta, "filename", None) or "file"
    content_type = getattr(meta, "content_type", None) or _guess_media_type(filename)
    headers = {"Content-Disposition": _content_disposition_inline(filename)}
    logger.debug(f"[FILE_ACCESS] File info: file_id={file_id}, filename={filename}, content_type={content_type}")

    blob_store = getattr(file_storage, "blob_store", None)
    if blob_store is None:
        logger.error(f"[FILE_ACCESS] File access failed: blob_store not configured for file_id={file_id}")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="blob store not configured")
    
    blob_store_type = blob_store.__class__.__name__
    logger.debug(f"[FILE_ACCESS] Blob store type: file_id={file_id}, blob_store_type={blob_store_type}")

    try:
        if blob_store_type == "LocalDB":
            logger.debug(f"[FILE_ACCESS] Using LocalDB for file access: file_id={file_id}, blob_key={blob_key}")
            # 优先使用 LocalDB 的 _get_full_path 方法，确保与存储时使用相同的路径解析逻辑
            # 这样可以保证存储和访问使用相同的 base_path（考虑配置中的 base_path）
            path = await anyio.to_thread.run_sync(blob_store._get_full_path, blob_key)
            logger.debug(f"[FILE_ACCESS] Primary path resolved: file_id={file_id}, blob_key={blob_key}, path={path}, exists={path.exists()}")
            
            if path.exists():
                logger.info(f"[FILE_ACCESS] File access success (primary path): file_id={file_id}, blob_key={blob_key}, path={path}")
                return FileResponse(path, media_type=content_type, headers=headers)
            else:
                logger.debug(f"[FILE_ACCESS] Primary path not found, trying fallback: file_id={file_id}, path={path}")
                # 兼容性处理1：如果路径不存在，尝试使用环境变量直接拼接（兼容历史数据）
                # 某些历史文件可能是在不同的 base_path 下存储的
                base_path = os.getenv("LOCAL_FILE_STORAGE_PATH") or os.getenv("LOCAL_BLOB_STORE_BASE_PATH") or "./data/files"
                base_dir = Path(base_path).expanduser().resolve()
                safe_key = str(blob_key).replace("..", "").lstrip("/")
                fallback_path = base_dir / safe_key
                logger.debug(f"[FILE_ACCESS] Fallback path: file_id={file_id}, base_path={base_path}, base_dir={base_dir}, safe_key={safe_key}, fallback_path={fallback_path}, exists={fallback_path.exists()}")
                
                if fallback_path.exists():
                    logger.info(f"[FILE_ACCESS] File access success (fallback path): file_id={file_id}, blob_key={blob_key}, path={fallback_path}")
                    return FileResponse(fallback_path, media_type=content_type, headers=headers)
                
                # 兼容性处理2：尝试修复 blob_key 中可能包含的重复路径
                # blob_key 可能包含类似 "RAG-ARC/local/files_chatKB_test/文件名.pdf" 的路径
                # 我们需要提取出实际的文件名部分
                logger.debug(f"[FILE_ACCESS] Fallback path not found, trying path fix: file_id={file_id}, safe_key={safe_key}")
                key_parts = safe_key.split("/")
                logger.debug(f"[FILE_ACCESS] Key parts: file_id={file_id}, key_parts={key_parts}, len={len(key_parts)}")
                if len(key_parts) >= 3:  # files/{prefix}/{file_id}/...
                    # 尝试只使用最后一部分（文件名）
                    file_id_part = key_parts[2] if len(key_parts) > 2 else ""
                    filename_part = key_parts[-1] if key_parts else ""
                    logger.debug(f"[FILE_ACCESS] Extracted parts: file_id={file_id}, file_id_part={file_id_part}, filename_part={filename_part}")
                    # 如果文件名部分包含路径分隔符，只取 basename
                    if "/" in filename_part or "\\" in filename_part:
                        original_filename_part = filename_part
                        filename_part = Path(filename_part).name
                        logger.debug(f"[FILE_ACCESS] Filename contains path, extracted basename: file_id={file_id}, original={original_filename_part}, basename={filename_part}")
                    # 重新构建 blob_key: files/{prefix}/{file_id}/{basename}
                    if file_id_part and filename_part:
                        fixed_key = f"files/{key_parts[1]}/{file_id_part}/{filename_part}"
                        logger.debug(f"[FILE_ACCESS] Fixed key generated: file_id={file_id}, original_blob_key={blob_key}, fixed_key={fixed_key}")
                        # 先尝试使用 LocalDB 的方法
                        fixed_path = await anyio.to_thread.run_sync(blob_store._get_full_path, fixed_key)
                        logger.debug(f"[FILE_ACCESS] Fixed path via LocalDB: file_id={file_id}, fixed_key={fixed_key}, fixed_path={fixed_path}, exists={fixed_path.exists()}")
                        if fixed_path.exists():
                            logger.info(f"[FILE_ACCESS] File access success (fixed path via LocalDB): file_id={file_id}, original_blob_key={blob_key}, fixed_blob_key={fixed_key}, path={fixed_path}")
                            return FileResponse(fixed_path, media_type=content_type, headers=headers)
                        # 再尝试使用环境变量直接拼接
                        fixed_fallback_path = base_dir / fixed_key
                        logger.debug(f"[FILE_ACCESS] Fixed path via env: file_id={file_id}, fixed_key={fixed_key}, fixed_fallback_path={fixed_fallback_path}, exists={fixed_fallback_path.exists()}")
                        if fixed_fallback_path.exists():
                            logger.info(f"[FILE_ACCESS] File access success (fixed path via env): file_id={file_id}, original_blob_key={blob_key}, fixed_blob_key={fixed_key}, path={fixed_fallback_path}")
                            return FileResponse(fixed_fallback_path, media_type=content_type, headers=headers)
                    else:
                        logger.debug(f"[FILE_ACCESS] Cannot fix key: file_id={file_id}, file_id_part={file_id_part}, filename_part={filename_part}")
                else:
                    logger.debug(f"[FILE_ACCESS] Key parts insufficient for fixing: file_id={file_id}, key_parts={key_parts}, len={len(key_parts)}")
                
                logger.warning(f"[FILE_ACCESS] File access failed: all paths not found for file_id={file_id}, blob_key={blob_key}, primary_path={path}, fallback_path={fallback_path}")
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="file not found")

        if blob_store_type == "MinIODB":
            logger.debug(f"[FILE_ACCESS] Using MinIODB for file access: file_id={file_id}, blob_key={blob_key}")
            return StreamingResponse(_stream_minio_object(blob_store, str(blob_key)), media_type=content_type, headers=headers)

        logger.debug(f"[FILE_ACCESS] Using generic blob store retrieve: file_id={file_id}, blob_key={blob_key}, blob_store_type={blob_store_type}")
        data = await anyio.to_thread.run_sync(blob_store.retrieve, str(blob_key))
        logger.info(f"[FILE_ACCESS] File access success (generic blob store): file_id={file_id}, blob_key={blob_key}, data_size={len(data) if data else 0}")
        return Response(content=data, media_type=content_type, headers=headers)
    except HTTPException as exc:
        logger.debug(f"[FILE_ACCESS] HTTPException raised: file_id={file_id}, status_code={exc.status_code}, detail={exc.detail}")
        raise
    except KeyError as exc:
        logger.warning(f"[FILE_ACCESS] File access failed: KeyError for file_id={file_id}, blob_key={blob_key}, error={exc}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="file not found") from exc
    except Exception as exc:
        logger.error(f"[FILE_ACCESS] File access failed: unexpected error for file_id={file_id}, blob_key={blob_key}, error={exc}, error_type={type(exc).__name__}", exc_info=True)
        raise


@router.get("/chatbot/bootstrap", response_model=ChatbotBootstrapResponse)
async def bootstrap(request: Request):
    user_id, should_set = _ensure_browser_user_id(request)
    payload = ChatbotBootstrapResponse(
        browser_user_id=str(user_id),
        cookie_name=_COOKIE_NAME,
        server_time=datetime.now(timezone.utc).isoformat(),
        capabilities=ChatbotBootstrapCapabilities(),
    )
    response = JSONResponse(content=payload.model_dump())
    if should_set:
        _apply_browser_cookie(response, user_id)
    return response


@router.post("/api/messages")
async def messages(
    request: Request,
    payload: ChatbotApiMessagesRequest,
    current_user: Annotated[User, Depends(get_current_user)],
):
    started = time.monotonic()
    request_id = str(uuid.uuid4())

    browser_user_id, should_set = _ensure_browser_user_id(request)
    conversation_uuid = _parse_uuid(payload.id, "id")
    
    logger.info(
        "chatbot.sse_start request_id=%s browser_user_id=%s conversation_id=%s message_length=%d user_id=%s",
        request_id,
        str(browser_user_id),
        str(conversation_uuid),
        len(payload.content) if payload.content else 0,
        str(current_user.id),
    )

    lock_timeout_s = float(os.getenv("CHATBOT_CONVERSATION_LOCK_TIMEOUT_S", "30"))
    semaphore_timeout_s = float(os.getenv("CHATBOT_GLOBAL_SEMAPHORE_TIMEOUT_S", "30"))
    max_context_tokens = int(os.getenv("CHATBOT_MAX_CONTEXT_TOKENS", "8192"))
    threshold_fraction = float(os.getenv("CHATBOT_MAX_CONTEXT_FRACTION", "0.9"))
    context_turns = int(os.getenv("CHATBOT_CONTEXT_TURNS", "5"))
    max_sources = int(os.getenv("CHATBOT_TOP_SOURCES", "5"))

    owner_id = resolve_default_owner_id(current_user)
    lock = await _get_conversation_lock(_conversation_key(browser_user_id, conversation_uuid))

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    async def _event_stream():
        acquired_lock = False
        acquired_global = False
        stop_event = threading.Event()
        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        parts: List[str] = []
        sources_for_frontend: List[ChatbotSourceItem] = []
        first_turn = len(payload.messages or []) == 0

        def _emit_error(code: int, message: str):
            # 使用统一的response格式
            error_response = {
                "code": code,
                "message": message,
                "data": None,
                "request_id": request_id
            }
            return [
                _sse_json(error_response),
                _sse_json({"type": "done", "status": "error", "id": payload.id}),
                _sse_done(),
            ]

        try:
            try:
                await asyncio.wait_for(lock.acquire(), timeout=lock_timeout_s)
                acquired_lock = True
            except TimeoutError:
                logger.warning(
                    "chatbot.lock_timeout request_id=%s conversation_id=%s",
                    request_id,
                    str(conversation_uuid),
                )
                for line in _emit_error(429, "conversation_busy"):
                    yield line
                return

            try:
                await asyncio.wait_for(_global_semaphore.acquire(), timeout=semaphore_timeout_s)
                acquired_global = True
            except TimeoutError:
                logger.warning(
                    "chatbot.semaphore_timeout request_id=%s conversation_id=%s",
                    request_id,
                    str(conversation_uuid),
                )
                for line in _emit_error(429, "server_busy"):
                    yield line
                return

            rag_inference_handler = _get_chatbot_rag_inference()
            needs_subgraph = True
            retrieval_query = payload.content.strip()

            def _prepare():
                return rag_inference_handler._build_messages_and_context(
                    query=retrieval_query,
                    owner_id=str(owner_id),
                    return_subgraph=needs_subgraph,
                )

            _, chunks, subgraph_data, subgraph_info = await anyio.to_thread.run_sync(_prepare)
            
            logger.info(
                "chatbot.retrieval_done request_id=%s conversation_id=%s chunks_count=%d",
                request_id,
                str(conversation_uuid),
                len(chunks) if chunks else 0,
            )

            evidence = build_chat_evidence(
                chunks or [],
                subgraph_data=subgraph_data,
                subgraph_info=subgraph_info,
                max_chunks=min(max_sources, CHAT_TOP_CHUNKS),
                graph_store=None,
            )
            evidence_chunks = evidence.get("chunks") or []

            sources_for_frontend = await anyio.to_thread.run_sync(
                _build_sources_for_frontend,
                evidence_chunks,
                min(max_sources, CHAT_TOP_CHUNKS),
            )
            source_messages = _build_source_messages_v2(sources_for_frontend)
            history = _normalize_history(payload.messages, turns=context_turns)

            llm_messages = _build_llm_messages(
                {"role": "system", "content": _CHATBOT_SYSTEM_PROMPT_V2},
                source_messages,
                history,
                payload.content.strip(),
            )

            try:
                _ensure_context_within_limit(
                    llm_messages,
                    max_context_tokens=max_context_tokens,
                    threshold_fraction=threshold_fraction,
                    token_estimator_suffix=f"window(last_turns={context_turns})",
                )
            except HTTPException as exc:
                detail = exc.detail if isinstance(exc.detail, dict) else {"code": "error", "message": str(exc.detail)}
                msg = detail.get("code") or "context_too_long"
                logger.warning(
                    "chatbot.context_too_long request_id=%s conversation_id=%s error=%s",
                    request_id,
                    str(conversation_uuid),
                    msg,
                )
                for line in _emit_error(413, str(msg)):
                    yield line
                return

            loop = asyncio.get_running_loop()

            def _producer():
                try:
                    for piece in rag_inference_handler.llm.stream_chat(llm_messages):
                        if stop_event.is_set():
                            break
                        loop.call_soon_threadsafe(queue.put_nowait, str(piece))
                finally:
                    loop.call_soon_threadsafe(queue.put_nowait, None)

            asyncio.create_task(asyncio.to_thread(_producer))

            while True:
                item = await queue.get()
                if item is None:
                    break
                parts.append(item)
                yield _sse_json({"type": "chunk", "content": item, "id": payload.id})

            full = "".join(parts).strip()
            sources_for_frontend = _filter_sources_by_sup_keys(sources_for_frontend, full)
            yield _sse_json(
                {"type": "sources", "sources": [s.model_dump() for s in sources_for_frontend], "id": payload.id}
            )

            if first_turn:
                title = await _generate_title_via_llm(payload.content.strip(), full)
                yield _sse_json({"type": "title", "title": title, "id": payload.id})

            yield _sse_json({"type": "done", "status": "success", "id": payload.id})
            yield _sse_done()

            elapsed_ms = int((time.monotonic() - started) * 1000)
            logger.info(
                "chatbot.sse_done request_id=%s browser_user_id=%s conversation_id=%s total_ms=%d sources=%d",
                request_id,
                str(browser_user_id),
                str(conversation_uuid),
                elapsed_ms,
                len(sources_for_frontend),
            )
        except asyncio.CancelledError:
            stop_event.set()
            raise
        except Exception as exc:  # noqa: BLE001
            stop_event.set()
            for line in _emit_error(500, "internal_error"):
                yield line
            logger.exception(
                "chatbot.sse_error request_id=%s browser_user_id=%s conversation_id=%s error=%s",
                request_id,
                str(browser_user_id),
                str(conversation_uuid),
                repr(exc),
            )
        finally:
            stop_event.set()
            if acquired_global:
                _global_semaphore.release()
            if acquired_lock:
                lock.release()

    response = StreamingResponse(_event_stream(), media_type="text/event-stream", headers=headers)
    if should_set:
        _apply_browser_cookie(response, browser_user_id)
    return response
