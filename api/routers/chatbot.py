import asyncio
import inspect
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
from core.prompts.chatbot import CHATBOT_SYSTEM_PROMPT_V2
from config.output_limits import (
    CHAT_TOP_CHUNKS,
    CHATBOT_LLM_TOP_SOURCES,
    RAG_RETRIEVAL_OBSERVABILITY,
    RAG_RETRIEVAL_LOG_TOP_FILES,
    RAG_RETRIEVAL_LOG_TOP_CHUNKS,
)
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
    # Optional page index info (recommended for UI display/persistence; populated by RAG SSE sources event).
    page_start: Optional[int] = None
    page_end: Optional[int] = None


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


# Match single key: <sup>1</sup> or multiple keys: <sup>1, 2</sup> or <sup>1,2,3</sup>
_SUP_KEY_RE = re.compile(r"<sup>\s*(?P<keys>[\d\s,]+?)\s*</sup>")
_SINGLE_KEY_RE = re.compile(r"\d{1,4}")


def _extract_sup_keys(text: str) -> List[int]:
    """Extract referenced source keys in the order they appear.
    Supports both single key format (<sup>1</sup>) and multiple keys format (<sup>1, 2</sup>).
    """
    if not text:
        return []
    seen: set[int] = set()
    keys: List[int] = []
    for match in _SUP_KEY_RE.finditer(text):
        keys_str = match.group("keys")
        if not keys_str:
            continue
        # Extract all numbers from the keys string (handles "1, 2" or "1,2,3" etc.)
        for key_match in _SINGLE_KEY_RE.finditer(keys_str):
            try:
                key = int(key_match.group(0))
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


def _build_sup_key_map_sorted(
    answer_text: str,
    *,
    sources: List[ChatbotSourceItem] | None = None,
) -> Dict[int, int]:
    """Build a stable citation key remap (sorted by original key, contiguous from 1)."""
    used_keys = set(_extract_sup_keys(answer_text))
    if sources is not None:
        available = {int(s.key) for s in sources if isinstance(getattr(s, "key", None), int)}
        used_keys &= available
    ordered = sorted(used_keys)
    return {old: idx + 1 for idx, old in enumerate(ordered)}


def _renumber_sup_tags(answer_text: str, key_map: Dict[int, int], *, sort_keys: bool = True) -> str:
    """Renumber <sup> tags using key_map. Supports both single and multiple key formats.
    
    Examples:
    - <sup>1</sup> -> <sup>1</sup> (if 1 maps to 1)
    - <sup>1, 2</sup> -> <sup>1, 2</sup> (if both keys exist in key_map)
    - <sup>1, 2</sup> -> <sup>1</sup> (if only 1 exists in key_map)
    - <sup>1, 2</sup> -> "" (if neither exists in key_map)
    """
    if not answer_text or not key_map:
        return answer_text or ""

    def _replace(match: re.Match[str]) -> str:
        keys_str = match.group("keys")
        if not keys_str:
            return ""
        
        # Extract all keys from the string
        original_keys: List[int] = []
        for key_match in _SINGLE_KEY_RE.finditer(keys_str):
            try:
                key = int(key_match.group(0))
                if key > 0:
                    original_keys.append(key)
            except Exception:  # noqa: BLE001
                continue
        
        if not original_keys:
            return ""
        
        # Map each key to new key, filtering out keys not in key_map
        new_keys: List[int] = []
        for orig_key in original_keys:
            new_key = key_map.get(orig_key)
            if new_key is not None:
                new_keys.append(new_key)
        
        # If no keys remain, remove the entire tag
        if not new_keys:
            return ""
        
        # Remove duplicates while preserving order
        seen = set()
        unique_new_keys = []
        for k in new_keys:
            if k not in seen:
                seen.add(k)
                unique_new_keys.append(k)
        
        # Sort for consistent output unless we want first-appearance order.
        if sort_keys:
            unique_new_keys.sort()
        
        # Format as comma-separated list
        keys_formatted = ", ".join(str(k) for k in unique_new_keys)
        return f"<sup>{keys_formatted}</sup>"

    return _SUP_KEY_RE.sub(_replace, answer_text)


def _filter_and_renumber_sources_by_sup_keys_sorted(
    sources: List[ChatbotSourceItem],
    answer_text: str,
) -> tuple[str, List[ChatbotSourceItem], Dict[int, int]]:
    """
    Keep only cited sources but ensure contiguous keys starting at 1.
    Remap strategy: sort by original key ascending.
    """
    if not sources:
        return (answer_text or "", [], {})

    key_map = _build_sup_key_map_sorted(answer_text, sources=sources)
    if not key_map:
        return (answer_text or "", [], {})

    filtered = [s for s in sources if s.key in key_map]
    filtered.sort(key=lambda item: item.key)
    renumbered_sources = [s.model_copy(update={"key": key_map[s.key]}) for s in filtered]
    renumbered_answer = _renumber_sup_tags(answer_text, key_map)
    return (renumbered_answer, renumbered_sources, key_map)


def _build_sup_key_map_by_appearance(
    answer_text: str,
    *,
    sources: List[ChatbotSourceItem] | None = None,
) -> Dict[int, int]:
    """Build a citation key remap based on first-appearance order."""
    ordered_keys = _extract_sup_keys(answer_text)
    if sources is not None:
        available = {int(s.key) for s in sources if isinstance(getattr(s, "key", None), int)}
        ordered_keys = [key for key in ordered_keys if key in available]
    key_map: Dict[int, int] = {}
    for key in ordered_keys:
        if key not in key_map:
            key_map[key] = len(key_map) + 1
    return key_map


def _renumber_sources_by_key_map(
    sources: List[ChatbotSourceItem],
    key_map: Dict[int, int],
) -> List[ChatbotSourceItem]:
    """Renumber sources using a precomputed key map (old->new)."""
    if not sources or not key_map:
        return []
    filtered = [s for s in sources if s.key in key_map]
    filtered.sort(key=lambda item: key_map[item.key])
    return [s.model_copy(update={"key": key_map[s.key]}) for s in filtered]


def _filter_and_renumber_sources_by_sup_keys_appearance(
    sources: List[ChatbotSourceItem],
    answer_text: str,
) -> tuple[str, List[ChatbotSourceItem], Dict[int, int]]:
    """
    Keep only cited sources but ensure contiguous keys starting at 1.
    Remap strategy: first-appearance order in the answer.
    """
    if not sources:
        return (answer_text or "", [], {})

    key_map = _build_sup_key_map_by_appearance(answer_text, sources=sources)
    if not key_map:
        return (answer_text or "", [], {})

    renumbered_sources = _renumber_sources_by_key_map(sources, key_map)
    renumbered_answer = _renumber_sup_tags(answer_text, key_map, sort_keys=False)
    return (renumbered_answer, renumbered_sources, key_map)


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


def _build_sources_for_frontend_with_llm_keys(
    entries: List[Dict[str, Any]], 
    max_sources: int,
    llm_chunk_id_to_key: dict[str, int]
) -> List[ChatbotSourceItem]:
    """
    Build sources for frontend using the same key assignment as LLM received.
    This ensures that if LLM cites key=4, the frontend will have a source with key=4.
    """
    max_chars = int(os.getenv("CHATBOT_SOURCE_MAX_CHARS", "1600"))
    sources: List[ChatbotSourceItem] = []
    seen_chunk_ids: set[str] = set()

    knowledge = None
    file_storage = None
    file_storage_resolved = False

    for entry in entries or []:
        chunk_id = str(entry.get("id") or "").strip()
        if not chunk_id:
            continue
        
        # Use LLM's key if available, otherwise skip (to maintain consistency)
        llm_key = llm_chunk_id_to_key.get(chunk_id)
        if llm_key is None:
            # This chunk was not passed to LLM, skip it
            continue
        
        # Skip if we've already seen this chunk_id (deduplication)
        if chunk_id in seen_chunk_ids:
            continue
        seen_chunk_ids.add(chunk_id)
        
        # Limit by max_sources
        if len(sources) >= max_sources:
            break

        metadata = dict(entry.get("metadata") or {})
        file_id = str(metadata.get("source_file_id") or "").strip() or None
        filename = str(metadata.get("filename") or "").strip() or "source"
        page_start = metadata.get("page_start")
        page_end = metadata.get("page_end")
        try:
            page_start = int(page_start) if page_start is not None else None
        except Exception:  # noqa: BLE001
            page_start = None
        try:
            page_end = int(page_end) if page_end is not None else None
        except Exception:  # noqa: BLE001
            page_end = None

        content = str(entry.get("content") or "").strip()
        if max_chars > 0 and len(content) > max_chars:
            content = f"{content[:max_chars].rstrip()}..."

        # Detect whether this chunk comes from Tavily web search.
        is_tavily_chunk = (
            chunk_id.startswith("tavily-")
            or metadata.get("source") == "web.tavily"
            or (not file_id and chunk_id and "tavily" in chunk_id.lower())
        )

        file_url = None
        source_title = filename

        if is_tavily_chunk:
            provenance = metadata.get("provenance") if isinstance(metadata.get("provenance"), dict) else {}
            url = provenance.get("url")
            if isinstance(url, str) and url.strip().lower().startswith(("http://", "https://")):
                file_url = url.strip()

            if content:
                content_lines = content.split("\n", 1)
                if content_lines and content_lines[0].strip():
                    source_title = content_lines[0].strip()
        else:
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

        if not file_url:
            provenance = metadata.get("provenance") if isinstance(metadata.get("provenance"), dict) else {}
            url = provenance.get("url")
            if isinstance(url, str) and url.strip().lower().startswith(("http://", "https://")):
                file_url = url.strip()

        sources.append(
            ChatbotSourceItem(
                key=llm_key,  # Use LLM's key instead of renumbering
                chunk_id=chunk_id,
                file_id=file_id,
                title=source_title,
                file=file_url,
                description=content,
                page_start=page_start,
                page_end=page_end,
            )
        )
    
    # Sort by LLM key to maintain order
    sources.sort(key=lambda s: s.key)
    return sources


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
    
    # Temporary: skip permission checks and allow all users (including unauthenticated users) to access files.
    # TODO: add proper authorization checks.
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
    
    # Temporary: skip permission checks and allow all users (including unauthenticated users) to access files.
    # TODO: add proper authorization checks.
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
            # Prefer LocalDB._get_full_path to keep path resolution consistent with storage time.
            # This ensures storage and access share the same base_path (respecting configured base_path).
            path = await anyio.to_thread.run_sync(blob_store._get_full_path, blob_key)
            logger.debug(f"[FILE_ACCESS] Primary path resolved: file_id={file_id}, blob_key={blob_key}, path={path}, exists={path.exists()}")
            
            if path.exists():
                logger.info(f"[FILE_ACCESS] File access success (primary path): file_id={file_id}, blob_key={blob_key}, path={path}")
                return FileResponse(path, media_type=content_type, headers=headers)
            else:
                logger.debug(f"[FILE_ACCESS] Primary path not found, trying fallback: file_id={file_id}, path={path}")
                # Compatibility 1: if path doesn't exist, try constructing from env base path (legacy data).
                # Some historical files may have been stored under a different base_path.
                base_path = os.getenv("LOCAL_FILE_STORAGE_PATH") or os.getenv("LOCAL_BLOB_STORE_BASE_PATH") or "./data/files"
                base_dir = Path(base_path).expanduser().resolve()
                safe_key = str(blob_key).replace("..", "").lstrip("/")
                fallback_path = base_dir / safe_key
                logger.debug(f"[FILE_ACCESS] Fallback path: file_id={file_id}, base_path={base_path}, base_dir={base_dir}, safe_key={safe_key}, fallback_path={fallback_path}, exists={fallback_path.exists()}")
                
                if fallback_path.exists():
                    logger.info(f"[FILE_ACCESS] File access success (fallback path): file_id={file_id}, blob_key={blob_key}, path={fallback_path}")
                    return FileResponse(fallback_path, media_type=content_type, headers=headers)
                
                # Compatibility 2: try repairing blob_key if it contains duplicated path segments.
                # blob_key may include paths like "RAG-ARC/local/files_chatKB_test/<filename>.pdf".
                # We need to extract the actual filename segment.
                logger.debug(f"[FILE_ACCESS] Fallback path not found, trying path fix: file_id={file_id}, safe_key={safe_key}")
                key_parts = safe_key.split("/")
                logger.debug(f"[FILE_ACCESS] Key parts: file_id={file_id}, key_parts={key_parts}, len={len(key_parts)}")
                if len(key_parts) >= 3:  # files/{prefix}/{file_id}/...
                    # Try using only the last segment (filename).
                    file_id_part = key_parts[2] if len(key_parts) > 2 else ""
                    filename_part = key_parts[-1] if key_parts else ""
                    logger.debug(f"[FILE_ACCESS] Extracted parts: file_id={file_id}, file_id_part={file_id_part}, filename_part={filename_part}")
                    # If filename contains separators, keep basename only.
                    if "/" in filename_part or "\\" in filename_part:
                        original_filename_part = filename_part
                        filename_part = Path(filename_part).name
                        logger.debug(f"[FILE_ACCESS] Filename contains path, extracted basename: file_id={file_id}, original={original_filename_part}, basename={filename_part}")
                    # Rebuild blob_key: files/{prefix}/{file_id}/{basename}
                    if file_id_part and filename_part:
                        fixed_key = f"files/{key_parts[1]}/{file_id_part}/{filename_part}"
                        logger.debug(f"[FILE_ACCESS] Fixed key generated: file_id={file_id}, original_blob_key={blob_key}, fixed_key={fixed_key}")
                        # First try LocalDB resolution.
                        fixed_path = await anyio.to_thread.run_sync(blob_store._get_full_path, fixed_key)
                        logger.debug(f"[FILE_ACCESS] Fixed path via LocalDB: file_id={file_id}, fixed_key={fixed_key}, fixed_path={fixed_path}, exists={fixed_path.exists()}")
                        if fixed_path.exists():
                            logger.info(f"[FILE_ACCESS] File access success (fixed path via LocalDB): file_id={file_id}, original_blob_key={blob_key}, fixed_blob_key={fixed_key}, path={fixed_path}")
                            return FileResponse(fixed_path, media_type=content_type, headers=headers)
                        # Then try env base path concatenation.
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
    max_sources = int(os.getenv("CHATBOT_TOP_SOURCES", "5"))  # UI sources
    llm_max_sources = int(CHATBOT_LLM_TOP_SOURCES or max_sources)

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
            # Use the standard response format.
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

            history = _normalize_history(payload.messages, turns=context_turns)
            # Build a compact "role: content" history string for history-aware query rewrite.
            history_text_lines: list[str] = []
            for msg in history or []:
                role = str(msg.get("role") or "").strip()
                content = str(msg.get("content") or "").strip()
                if role not in ("user", "assistant") or not content:
                    continue
                history_text_lines.append(f"{role}: {content}")
            # If the last user message equals the current query, drop it to avoid redundancy.
            if history_text_lines and history_text_lines[-1] == f"user: {retrieval_query}":
                history_text_lines.pop()
            history_text = "\n".join(history_text_lines) if history_text_lines else None

            def _prepare():
                kwargs: dict[str, Any] = {
                    "query": retrieval_query,
                    "owner_id": str(owner_id),
                    "return_subgraph": needs_subgraph,
                    "history_text": history_text,
                }
                try:
                    sig = inspect.signature(rag_inference_handler._build_messages_and_context)
                    accepts_var_kw = any(
                        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
                    )
                    if not accepts_var_kw and "history_text" not in sig.parameters:
                        kwargs.pop("history_text", None)
                except Exception as exc:  # noqa: BLE001
                    # Compatibility: some test stubs (and older handlers) do not accept history_text.
                    logger.debug(
                        "Failed to inspect rag_inference_handler._build_messages_and_context signature: %s",
                        exc,
                        exc_info=True,
                    )
                    kwargs.pop("history_text", None)
                return rag_inference_handler._build_messages_and_context(
                    **kwargs,
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
                # For chatbot answer generation, allow passing more Sources to the LLM than the UI shows.
                # This does not change the frontend payload; it only affects the LLM context.
                max_chunks=llm_max_sources,
                graph_store=None,
            )
            evidence_chunks = evidence.get("chunks") or []

            sources_for_llm = await anyio.to_thread.run_sync(
                _build_sources_for_frontend,
                evidence_chunks,
                llm_max_sources,
            )
            sources_for_frontend = sources_for_llm[: min(max_sources, len(sources_for_llm))]
            source_messages = _build_source_messages_v2(sources_for_frontend)
            # Use the same normalized history for the answer generation window.

            llm_messages = _build_llm_messages(
                {"role": "system", "content": CHATBOT_SYSTEM_PROMPT_V2},
                _build_source_messages_v2(sources_for_llm),
                history,
                payload.content.strip(),
            )

            if RAG_RETRIEVAL_OBSERVABILITY:
                from collections import Counter

                def _file_id(meta: Any) -> str | None:
                    if not isinstance(meta, dict):
                        return None
                    for key in ("source_file_id", "sourceFileId", "file_id", "fileId"):
                        token = str(meta.get(key) or "").strip()
                        if token:
                            return token
                    return None

                def _dist(serialized_chunks: list[dict[str, Any]], *, limit: int) -> list[tuple[str, int]]:
                    ctr: Counter[str] = Counter()
                    for entry in serialized_chunks:
                        fid = _file_id(entry.get("metadata") or {})
                        if fid:
                            ctr[fid] += 1
                    return ctr.most_common(max(int(limit), 0))

                logger.info(
                    "chatbot.retrieval_observe request_id=%s conversation_id=%s query=%r history_chars=%s "
                    "retrieved=%d evidence_llm=%d evidence_ui=%d top_files_llm=%s",
                    request_id,
                    str(conversation_uuid),
                    retrieval_query,
                    len(history_text or "") if history_text else 0,
                    len(chunks or []),
                    len(sources_for_llm or []),
                    len(sources_for_frontend or []),
                    _dist(list(evidence_chunks or []), limit=RAG_RETRIEVAL_LOG_TOP_FILES),
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
            full, sources_for_frontend, citation_key_map = _filter_and_renumber_sources_by_sup_keys_sorted(
                sources_for_frontend,
                full,
            )
            yield _sse_json(
                {
                    "type": "final",
                    "content": full,
                    "citation_key_map": {str(k): v for k, v in (citation_key_map or {}).items()},
                    "id": payload.id,
                }
            )
            yield _sse_json(
                {
                    "type": "sources",
                    "sources": [s.model_dump() for s in sources_for_frontend],
                    "citation_key_map": {str(k): v for k, v in (citation_key_map or {}).items()},
                    "id": payload.id,
                }
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
