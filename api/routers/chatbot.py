import asyncio
import json
import logging
import os
import time
import uuid
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import anyio
from fastapi import APIRouter, HTTPException, Query, Request, Response, WebSocket, WebSocketDisconnect, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from framework.register import Register
from core.presentation.evidence import build_chat_evidence
from config.output_limits import CHAT_TOP_CHUNKS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chatbot", tags=["chatbot"])

_COOKIE_NAME = "rag_arc_uid"
_registrator = Register()

_conversation_locks: Dict[str, asyncio.Lock] = {}
_conversation_last_used: Dict[str, float] = {}
_locks_guard = asyncio.Lock()
_global_semaphore = asyncio.Semaphore(int(os.getenv("CHATBOT_MAX_CONCURRENCY", "8")))

_CHATBOT_SYSTEM_PROMPT = (
    "You are a helpful RAG assistant.\n"
    "You may be given a list of numbered Sources.\n"
    "- Use Sources to answer. Do not fabricate citations.\n"
    "- When a sentence is supported by a source, add citation markers like [1] or [2] at the end of that sentence.\n"
    "- Always end your answer with a single line in the exact format: Sources: [1][2]...\n"
    "- If no sources are applicable, end with: Sources: []\n"
)

_CHATBOT_SYSTEM_PROMPT_NO_EVIDENCE = "You are a helpful assistant."


class ChatbotMessage(BaseModel):
    role: Literal["user", "assistant", "system"] = "user"
    content: str = Field(min_length=1)


class ChatbotMemory(BaseModel):
    version: int = 0
    summary: str = ""
    recent_messages: List[ChatbotMessage] = Field(default_factory=list)


class ChatbotOptions(BaseModel):
    include_evidence: bool = True
    top_k: int = Field(default=5, ge=1, le=5)
    return_subgraph: bool = False
    max_context_fraction: float = 0.9


class ChatbotChatRequest(BaseModel):
    conversation_id: str
    message: ChatbotMessage
    memory: Optional[ChatbotMemory] = None
    options: Optional[ChatbotOptions] = None


class ChatbotCitation(BaseModel):
    rank: int
    chunk_id: str
    score: Optional[float] = None
    filename: Optional[str] = None
    file_id: Optional[str] = None
    preview: Optional[str] = None
    chunk_url: str
    file_url: Optional[str] = None
    start_idx: Optional[int] = None
    end_idx: Optional[int] = None


class ChatbotDebugContext(BaseModel):
    max_context_tokens: int
    threshold_fraction: float
    estimated_context_tokens: int
    compressed: bool
    token_estimator: str


class ChatbotDebug(BaseModel):
    context: ChatbotDebugContext
    timing_ms: Dict[str, int]


class ChatbotChatResponse(BaseModel):
    request_id: str
    browser_user_id: str
    conversation_id: str
    memory: ChatbotMemory
    assistant: ChatbotMessage
    citations: List[ChatbotCitation]
    debug: ChatbotDebug


class ChatbotBootstrapCapabilities(BaseModel):
    streaming: bool = True
    evidence: bool = True


class ChatbotBootstrapResponse(BaseModel):
    browser_user_id: str
    cookie_name: str
    server_time: str
    capabilities: ChatbotBootstrapCapabilities


class ChatbotChunkResponse(BaseModel):
    chunk_id: str
    file_id: Optional[str] = None
    filename: Optional[str] = None
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatbotTitleRequest(BaseModel):
    conversation_id: str
    user: str = Field(min_length=1)
    assistant: Optional[str] = None


class ChatbotTitleResponse(BaseModel):
    request_id: str
    browser_user_id: str
    conversation_id: str
    title: str


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


def _ensure_cookie_user_exists(user_id: uuid.UUID) -> None:
    """Create a placeholder DB user so ingestion can store FK owner_id metadata."""
    try:
        from config.encapsulation.database.relational_db.postgresql_config import PostgreSQLConfig
        from encapsulation.data_model.orm_models import User
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="user store unavailable") from exc

    db = PostgreSQLConfig().build()
    now = datetime.now(tz=datetime.now().astimezone().tzinfo)
    username = f"chatbot_cookie_{str(user_id)[:8]}"

    with db.SessionMaker() as session:
        existing = session.query(User).filter_by(id=user_id).first()
        if existing is not None:
            return
        session.add(
            User(
                id=user_id,
                user_name=username,
                hashed_password="chatbot-cookie-placeholder",
                created_at=now,
                updated_at=now,
            )
        )
        try:
            session.commit()
        except Exception:
            session.rollback()
            existing = session.query(User).filter_by(id=user_id).first()
            if existing is None:
                raise


def _get_shared_document_owner_id() -> uuid.UUID:
    raw = os.getenv("CHATBOT_SHARED_DOCUMENT_OWNER_ID", "00000000-0000-0000-0000-000000000001")
    try:
        return uuid.UUID(str(raw))
    except ValueError as exc:
        raise RuntimeError("CHATBOT_SHARED_DOCUMENT_OWNER_ID must be a valid UUID") from exc


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


def _format_context(memory: ChatbotMemory, user_message: ChatbotMessage) -> str:
    lines: List[str] = []
    if memory.summary.strip():
        lines.append(f"[summary]\n{memory.summary.strip()}")
    for msg in memory.recent_messages:
        role = msg.role.strip()
        content = msg.content.strip()
        if content:
            lines.append(f"{role}: {content}")
    lines.append(f"{user_message.role}: {user_message.content.strip()}")
    return "\n".join(lines).strip()


def _format_context_for_estimate(memory: ChatbotMemory, user_message: Optional[ChatbotMessage]) -> str:
    lines: List[str] = []
    if memory.summary.strip():
        lines.append(memory.summary.strip())
    for msg in memory.recent_messages:
        content = msg.content.strip()
        if content:
            lines.append(content)
    if user_message is not None and user_message.content.strip():
        lines.append(user_message.content.strip())
    return "\n".join(lines).strip()


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
    memory: ChatbotMemory,
    sources: List[Dict[str, str]],
    user_message: ChatbotMessage,
) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [system_message]
    for m in memory.recent_messages:
        msgs.append({"role": m.role, "content": m.content})
    msgs.extend(sources or [])
    msgs.append({"role": "user", "content": user_message.content})
    return msgs


def _ensure_context_within_limit(
    messages: List[Dict[str, str]],
    *,
    max_context_tokens: int,
    threshold_fraction: float,
    token_estimator_suffix: str = "window_disabled",
) -> ChatbotDebugContext:
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
    return ChatbotDebugContext(
        max_context_tokens=max_context_tokens,
        threshold_fraction=threshold_fraction,
        estimated_context_tokens=estimated,
        compressed=False,
        token_estimator=f"{estimator}|{token_estimator_suffix}",
    )


def _last_n_turns_messages(memory: ChatbotMemory, *, turns: int) -> List[ChatbotMessage]:
    turns = max(int(turns or 0), 0)
    if turns <= 0:
        return []
    limit = turns * 2
    msgs = list(memory.recent_messages or [])
    return msgs[-limit:] if len(msgs) > limit else msgs


def _trim_memory_to_last_n_turns(memory: ChatbotMemory, *, turns: int) -> ChatbotMemory:
    memory.recent_messages = _last_n_turns_messages(memory, turns=turns)
    return memory


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


async def _maybe_send_ws_title(
    websocket: WebSocket,
    *,
    request_id: str,
    browser_user_id: uuid.UUID,
    conversation_id: uuid.UUID,
    user_text: str,
    assistant_text: str,
) -> None:
    try:
        title = await _generate_title_via_llm(user_text, assistant_text)
        await websocket.send_json(
            {
                "type": "title",
                "request_id": request_id,
                "browser_user_id": str(browser_user_id),
                "conversation_id": str(conversation_id),
                "title": title,
            }
        )
    except Exception:  # noqa: BLE001
        return


def _citations_from_chunks(chunks: List[Any], *, max_chunks: int) -> List[ChatbotCitation]:
    citations: List[ChatbotCitation] = []
    seen_chunk_ids: set[str] = set()
    seen_file_ids: set[str] = set()
    for chunk in chunks:
        if len(citations) >= max_chunks:
            break
        metadata = getattr(chunk, "metadata", {}) or {}
        chunk_id = str(getattr(chunk, "id", "") or "").strip()
        file_id = metadata.get("source_file_id")
        filename = metadata.get("filename")
        preview = (getattr(chunk, "content", "") or "").strip()
        preview = preview[:240] if preview else None

        if not chunk_id or chunk_id in seen_chunk_ids:
            continue
        if file_id and str(file_id) in seen_file_ids:
            continue

        seen_chunk_ids.add(chunk_id)
        if file_id:
            seen_file_ids.add(str(file_id))

        file_url = None
        if file_id:
            file_url = f"/chatbot/files/{file_id}?disposition=inline"

        start_idx = metadata.get("start_idx") or metadata.get("start") or metadata.get("chunk_start")
        end_idx = metadata.get("end_idx") or metadata.get("end") or metadata.get("chunk_end")

        citations.append(
            ChatbotCitation(
                rank=len(citations) + 1,
                chunk_id=chunk_id,
                score=metadata.get("score"),
                filename=filename,
                file_id=file_id,
                preview=preview,
                chunk_url=f"/chatbot/chunks/{chunk_id}",
                file_url=file_url,
                start_idx=start_idx if isinstance(start_idx, int) else None,
                end_idx=end_idx if isinstance(end_idx, int) else None,
            )
        )
    return citations


def _strip_trailing_sources_line(text: str) -> str:
    cleaned = (text or "").rstrip()
    lines = cleaned.splitlines()
    while lines and lines[-1].strip() == "":
        lines.pop()
    if lines and lines[-1].lstrip().startswith("Sources: "):
        lines.pop()
        while lines and lines[-1].strip() == "":
            lines.pop()
    result = "\n".join(lines).strip()
    return result or "(empty)"


def _append_sources_markers(text: str, citations: List[ChatbotCitation]) -> str:
    base = _strip_trailing_sources_line(text)
    markers = "".join(f"[{c.rank}]" for c in citations) if citations else "[]"
    return f"{base}\n\nSources: {markers}"


def _build_source_messages(citations: List[ChatbotCitation], chunk_lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    if not citations:
        return []
    max_chars = int(os.getenv("CHATBOT_SOURCE_MAX_CHARS", "1600"))
    messages: List[Dict[str, str]] = []
    for cite in citations:
        entry = chunk_lookup.get(cite.chunk_id) or {}
        content = (entry.get("content") or "").strip()
        if max_chars > 0 and len(content) > max_chars:
            content = f"{content[:max_chars].rstrip()}..."
        filename = cite.filename or (entry.get("metadata") or {}).get("filename")
        header = f"Source [{cite.rank}]"
        if filename:
            header += f" filename={filename}"
        header += f" chunk_id={cite.chunk_id}"
        messages.append({"role": "user", "content": f"{header}\n{content}"})
    return messages


@router.get("/bootstrap", response_model=ChatbotBootstrapResponse)
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


@router.post("/title", response_model=ChatbotTitleResponse)
async def generate_title(request: Request, payload: ChatbotTitleRequest):
    started = time.monotonic()
    request_id = str(uuid.uuid4())

    browser_user_id, should_set = _ensure_browser_user_id(request)
    conversation_uuid = _parse_uuid(payload.conversation_id, "conversation_id")

    title = await _generate_title_via_llm(payload.user, payload.assistant or "")

    result = ChatbotTitleResponse(
        request_id=request_id,
        browser_user_id=str(browser_user_id),
        conversation_id=str(conversation_uuid),
        title=title,
    )
    response = JSONResponse(content=result.model_dump())
    if should_set:
        _apply_browser_cookie(response, browser_user_id)

    elapsed_ms = int((time.monotonic() - started) * 1000)
    logger.info(
        "chatbot.title request_id=%s browser_user_id=%s conversation_id=%s total_ms=%d",
        request_id,
        str(browser_user_id),
        str(conversation_uuid),
        elapsed_ms,
    )
    return response


@router.post("/chat", response_model=ChatbotChatResponse)
async def chat(request: Request, payload: ChatbotChatRequest):
    started = time.monotonic()
    request_id = str(uuid.uuid4())

    browser_user_id, should_set = _ensure_browser_user_id(request)
    conversation_uuid = _parse_uuid(payload.conversation_id, "conversation_id")

    incoming_memory = payload.memory or ChatbotMemory()
    options = payload.options or ChatbotOptions()

    if payload.message.role != "user":
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="message.role must be 'user'")

    lock = await _get_conversation_lock(_conversation_key(browser_user_id, conversation_uuid))
    lock_timeout_s = float(os.getenv("CHATBOT_CONVERSATION_LOCK_TIMEOUT_S", "30"))
    semaphore_timeout_s = float(os.getenv("CHATBOT_GLOBAL_SEMAPHORE_TIMEOUT_S", "30"))

    acquired_lock = False
    try:
        await asyncio.wait_for(lock.acquire(), timeout=lock_timeout_s)
        acquired_lock = True
    except TimeoutError as exc:
        raise HTTPException(status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail="conversation is busy") from exc

    acquired_global = False
    try:
        await asyncio.wait_for(_global_semaphore.acquire(), timeout=semaphore_timeout_s)
        acquired_global = True

        max_context_tokens = int(os.getenv("CHATBOT_MAX_CONTEXT_TOKENS", "8192"))
        threshold_fraction = float(options.max_context_fraction or 0.9)
        context_turns = int(os.getenv("CHATBOT_CONTEXT_TURNS", "5"))

        memory = ChatbotMemory(**incoming_memory.model_dump())
        rag_inference_handler = _get_chatbot_rag_inference()

        needs_subgraph = bool(options.return_subgraph or options.include_evidence)
        retrieval_query = payload.message.content.strip()

        shared_owner_id = _get_shared_document_owner_id()

        def _prepare():
            return rag_inference_handler.prepare_chat(
                retrieval_query,
                owner_id=shared_owner_id,
                return_subgraph=needs_subgraph,
            )

        _, chunks, subgraph_data, subgraph_info, _prepared = await anyio.to_thread.run_sync(_prepare)

        citations: List[ChatbotCitation] = []
        chunk_lookup: Dict[str, Dict[str, Any]] = {}
        if options.include_evidence:
            graph_store = None
            try:
                graph_store = rag_inference_handler.get_graph_store()
            except Exception:  # noqa: BLE001
                graph_store = None
            evidence = build_chat_evidence(
                chunks or [],
                subgraph_data=subgraph_data,
                subgraph_info=subgraph_info,
                max_chunks=min(options.top_k, CHAT_TOP_CHUNKS),
                graph_store=graph_store,
            )
            evidence_chunks = evidence.get("chunks") or []
            chunk_objs = []
            for entry in evidence_chunks:
                chunk_id = str(entry.get("id") or "").strip()
                if chunk_id:
                    chunk_lookup[chunk_id] = entry
                md = dict(entry.get("metadata") or {})
                score = entry.get("score")
                if score is not None and "score" not in md:
                    md["score"] = score
                chunk_objs.append(type("ChunkView", (), {"id": entry.get("id"), "content": entry.get("content"), "metadata": md}))
            citations = _citations_from_chunks(chunk_objs, max_chunks=min(options.top_k, CHAT_TOP_CHUNKS))

        sources = _build_source_messages(citations, chunk_lookup)
        system_prompt = _CHATBOT_SYSTEM_PROMPT if options.include_evidence else _CHATBOT_SYSTEM_PROMPT_NO_EVIDENCE
        memory_for_llm = ChatbotMemory(
            version=memory.version,
            summary="",
            recent_messages=_last_n_turns_messages(memory, turns=context_turns),
        )
        llm_messages = _build_llm_messages({"role": "system", "content": system_prompt}, memory_for_llm, sources, payload.message)
        debug_ctx = _ensure_context_within_limit(
            llm_messages,
            max_context_tokens=max_context_tokens,
            threshold_fraction=threshold_fraction,
            token_estimator_suffix=f"window(last_turns={context_turns})",
        )

        response_text = await anyio.to_thread.run_sync(rag_inference_handler.llm.chat, llm_messages)

        assistant_message = ChatbotMessage(role="assistant", content=str(response_text or "").strip() or "(empty)")
        if options.include_evidence:
            assistant_message.content = _append_sources_markers(assistant_message.content, citations)

        memory.recent_messages = list(memory.recent_messages) + [payload.message, assistant_message]
        memory = _trim_memory_to_last_n_turns(memory, turns=context_turns)
        memory.version = int(memory.version or 0) + 1

        elapsed_ms = int((time.monotonic() - started) * 1000)
        debug = ChatbotDebug(context=debug_ctx, timing_ms={"total": elapsed_ms})

        result = ChatbotChatResponse(
            request_id=request_id,
            browser_user_id=str(browser_user_id),
            conversation_id=str(conversation_uuid),
            memory=memory,
            assistant=assistant_message,
            citations=citations,
            debug=debug,
        )
        response = JSONResponse(content=result.model_dump())
        if should_set:
            _apply_browser_cookie(response, browser_user_id)
        logger.info(
            "chatbot.chat request_id=%s browser_user_id=%s conversation_id=%s total_ms=%d compressed=%s citations=%d",
            request_id,
            str(browser_user_id),
            str(conversation_uuid),
            elapsed_ms,
            debug_ctx.compressed,
            len(citations),
        )
        return response
    finally:
        if acquired_global:
            _global_semaphore.release()
        if acquired_lock:
            lock.release()


@router.get("/chunks/{chunk_id}", response_model=ChatbotChunkResponse)
async def get_chunk(request: Request, chunk_id: str):
    _require_browser_user_id(request)

    knowledge = _get_chatbot_knowledge()
    chunk_storage = getattr(getattr(knowledge, "file_index", None), "chunk_storage", None)
    if chunk_storage is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="chunk storage unavailable")

    chunk_metadata = await anyio.to_thread.run_sync(chunk_storage.get_chunk_metadata, chunk_id)
    if chunk_metadata is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="chunk not found")

    content_bytes = await anyio.to_thread.run_sync(chunk_storage.get_chunk_content, chunk_id)
    if not content_bytes:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="chunk not found")

    try:
        raw = json.loads(content_bytes.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="chunk decode failed") from exc

    metadata = raw.get("metadata") if isinstance(raw, dict) else {}
    if not isinstance(metadata, dict):
        metadata = {}
    source_metadata = raw.get("source_metadata") if isinstance(raw, dict) else None
    if isinstance(source_metadata, dict) and source_metadata:
        merged = dict(source_metadata)
        merged.update(metadata)
        metadata = merged
    file_id = metadata.get("source_file_id")
    filename = metadata.get("filename")
    content = raw.get("content") if isinstance(raw, dict) else None
    if content is None:
        content = content_bytes.decode("utf-8", errors="replace")

    result = ChatbotChunkResponse(
        chunk_id=chunk_id,
        file_id=file_id,
        filename=filename,
        content=str(content),
        metadata=metadata,
    )
    return JSONResponse(content=result.model_dump())


@router.get("/files/{file_id}")
async def get_file(
    request: Request,
    file_id: str,
    disposition: Literal["inline", "attachment"] = Query(default="inline"),
):
    _require_browser_user_id(request)

    knowledge = _get_chatbot_knowledge()
    file_storage = getattr(knowledge, "file_storage", None)
    if file_storage is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="file storage unavailable")

    metadata = await anyio.to_thread.run_sync(file_storage.get_file_metadata, file_id)
    if metadata is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="file not found")

    content = await anyio.to_thread.run_sync(file_storage.get_file_content, file_id)
    if content is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="file content not found")

    filename = getattr(metadata, "filename", None) or file_id
    media_type = getattr(metadata, "content_type", None) or "application/octet-stream"
    response = Response(content=content, media_type=media_type)
    response.headers["Content-Disposition"] = f'{disposition}; filename="{filename}"'
    return response


@router.websocket("/ws")
async def websocket_chat(websocket: WebSocket, conversation_id: str):
    await websocket.accept()
    raw_identity = (
        websocket.headers.get("x-owner-id")
        or websocket.headers.get("x-browser-user-id")
        or websocket.query_params.get("owner_id")
        or websocket.cookies.get(_COOKIE_NAME)
    )
    if not raw_identity:
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    browser_user_id = _parse_uuid(raw_identity, _COOKIE_NAME)
    conversation_uuid = _parse_uuid(conversation_id, "conversation_id")
    shared_owner_id = _get_shared_document_owner_id()
    context_turns = int(os.getenv("CHATBOT_CONTEXT_TURNS", "5"))

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                incoming = json.loads(raw)
            except Exception:
                await websocket.send_json({"type": "error", "detail": "invalid_json"})
                continue

            message = incoming.get("message") or {}
            memory_raw = incoming.get("memory")
            options_raw = incoming.get("options")

            try:
                msg = ChatbotMessage(**message)
            except Exception:
                await websocket.send_json({"type": "error", "detail": "invalid_message"})
                continue
            if msg.role != "user":
                await websocket.send_json({"type": "error", "detail": "message.role must be 'user'"})
                continue

            memory = ChatbotMemory(**memory_raw) if isinstance(memory_raw, dict) else ChatbotMemory()
            options = ChatbotOptions(**options_raw) if isinstance(options_raw, dict) else ChatbotOptions()
            is_first_turn = len(memory.recent_messages or []) == 0

            request_id = str(uuid.uuid4())
            await websocket.send_json({"type": "start", "request_id": request_id})

            lock = await _get_conversation_lock(_conversation_key(browser_user_id, conversation_uuid))
            acquired_lock = False
            try:
                await asyncio.wait_for(lock.acquire(), timeout=float(os.getenv("CHATBOT_CONVERSATION_LOCK_TIMEOUT_S", "30")))
                acquired_lock = True
            except TimeoutError:
                await websocket.send_json({"type": "error", "request_id": request_id, "detail": "conversation_busy"})
                continue

            acquired_global = False
            started = time.monotonic()
            disconnected = False
            try:
                await asyncio.wait_for(_global_semaphore.acquire(), timeout=float(os.getenv("CHATBOT_GLOBAL_SEMAPHORE_TIMEOUT_S", "30")))
                acquired_global = True

                max_context_tokens = int(os.getenv("CHATBOT_MAX_CONTEXT_TOKENS", "8192"))
                threshold_fraction = float(options.max_context_fraction or 0.9)
                memory = ChatbotMemory(**memory.model_dump())
                rag_inference_handler = _get_chatbot_rag_inference()
                needs_subgraph = bool(options.return_subgraph or options.include_evidence)
                retrieval_query = msg.content.strip()

                def _prepare():
                    return rag_inference_handler.prepare_chat(
                        retrieval_query,
                        owner_id=shared_owner_id,
                        return_subgraph=needs_subgraph,
                    )

                _, chunks, subgraph_data, subgraph_info, _messages = await anyio.to_thread.run_sync(_prepare)

                citations: List[ChatbotCitation] = []
                chunk_lookup: Dict[str, Dict[str, Any]] = {}
                if options.include_evidence:
                    graph_store = None
                    try:
                        graph_store = rag_inference_handler.get_graph_store()
                    except Exception:  # noqa: BLE001
                        graph_store = None
                    evidence = build_chat_evidence(
                        chunks or [],
                        subgraph_data=subgraph_data,
                        subgraph_info=subgraph_info,
                        max_chunks=min(options.top_k, CHAT_TOP_CHUNKS),
                        graph_store=graph_store,
                    )
                    evidence_chunks = evidence.get("chunks") or []
                    chunk_objs = []
                    for entry in evidence_chunks:
                        chunk_id = str(entry.get("id") or "").strip()
                        if chunk_id:
                            chunk_lookup[chunk_id] = entry
                        md = dict(entry.get("metadata") or {})
                        score = entry.get("score")
                        if score is not None and "score" not in md:
                            md["score"] = score
                        chunk_objs.append(type("ChunkView", (), {"id": entry.get("id"), "content": entry.get("content"), "metadata": md}))
                    citations = _citations_from_chunks(chunk_objs, max_chunks=min(options.top_k, CHAT_TOP_CHUNKS))

                sources = _build_source_messages(citations, chunk_lookup)
                system_prompt = _CHATBOT_SYSTEM_PROMPT if options.include_evidence else _CHATBOT_SYSTEM_PROMPT_NO_EVIDENCE
                try:
                    memory_for_llm = ChatbotMemory(
                        version=memory.version,
                        summary="",
                        recent_messages=_last_n_turns_messages(memory, turns=context_turns),
                    )
                    llm_messages = _build_llm_messages(
                        {"role": "system", "content": system_prompt},
                        memory_for_llm,
                        sources,
                        msg,
                    )
                    debug_ctx = _ensure_context_within_limit(
                        llm_messages,
                        max_context_tokens=max_context_tokens,
                        threshold_fraction=threshold_fraction,
                        token_estimator_suffix=f"window(last_turns={context_turns})",
                    )
                except HTTPException as exc:
                    await websocket.send_json({"type": "error", "request_id": request_id, "detail": exc.detail})
                    continue

                loop = asyncio.get_running_loop()
                queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
                stop_event = threading.Event()

                def _producer():
                    try:
                        for piece in rag_inference_handler.llm.stream_chat(llm_messages):
                            if stop_event.is_set():
                                break
                            loop.call_soon_threadsafe(queue.put_nowait, piece)
                    finally:
                        loop.call_soon_threadsafe(queue.put_nowait, None)

                thread = asyncio.to_thread(_producer)
                asyncio.create_task(thread)

                parts: List[str] = []
                while True:
                    item = await queue.get()
                    if item is None:
                        break
                    parts.append(item)
                    try:
                        await websocket.send_json({"type": "delta", "request_id": request_id, "content": item})
                    except WebSocketDisconnect:
                        stop_event.set()
                        disconnected = True
                        break
                    except Exception:  # noqa: BLE001
                        stop_event.set()
                        disconnected = True
                        break

                if disconnected:
                    continue

                full = "".join(parts).strip() or "(empty)"

                assistant_message = ChatbotMessage(role="assistant", content=full)
                if options.include_evidence:
                    assistant_message.content = _append_sources_markers(assistant_message.content, citations)

                memory.recent_messages = list(memory.recent_messages) + [msg, assistant_message]
                memory = _trim_memory_to_last_n_turns(memory, turns=context_turns)
                memory.version = int(memory.version or 0) + 1

                elapsed_ms = int((time.monotonic() - started) * 1000)
                debug = ChatbotDebug(context=debug_ctx, timing_ms={"total": elapsed_ms})

                await websocket.send_json(
                    {
                        "type": "final",
                        "request_id": request_id,
                        "browser_user_id": str(browser_user_id),
                        "conversation_id": str(conversation_uuid),
                        "memory": memory.model_dump(),
                        "assistant": assistant_message.model_dump(),
                        "citations": [c.model_dump() for c in citations],
                        "debug": debug.model_dump(),
                    }
                )
                if is_first_turn:
                    asyncio.create_task(
                        _maybe_send_ws_title(
                            websocket,
                            request_id=request_id,
                            browser_user_id=browser_user_id,
                            conversation_id=conversation_uuid,
                            user_text=msg.content,
                            assistant_text=assistant_message.content,
                        )
                    )
                logger.info(
                    "chatbot.ws_final request_id=%s browser_user_id=%s conversation_id=%s total_ms=%d compressed=%s citations=%d",
                    request_id,
                    str(browser_user_id),
                    str(conversation_uuid),
                    elapsed_ms,
                    debug_ctx.compressed,
                    len(citations),
                )
            finally:
                if acquired_global:
                    _global_semaphore.release()
                if acquired_lock:
                    lock.release()
    except WebSocketDisconnect:
        return
