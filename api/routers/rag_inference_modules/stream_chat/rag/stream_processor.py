"""Stream processing logic for RAG inference."""
import os
import asyncio
import contextvars
import threading
import logging
from typing import Any
from api.routers.rag_inference_handlers import get_rag_inference_handler

logger = logging.getLogger(__name__)


def _run_stream_processing(
    enhanced_query: str,
    effective_owner: Any,
    return_subgraph: bool,
    include_evidence: bool,
    history_text: str | None,
    enable_web_search: bool,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    prepared: dict[str, Any],
    stream_error: list[Exception | None],
    emit_progress: callable
) -> None:
    """Run stream processing in a separate thread."""
    try:
        emit_progress({"stage": "prepare", "status": "start"})
        
        user_type_str = os.getenv("USER_TYPE", "0")
        try:
            user_type = int(user_type_str)
        except ValueError:
            logger.warning(
                "Invalid USER_TYPE environment variable: %s, defaulting to 0",
                user_type_str,
            )
            user_type = 0
        
        rag_inference_handler = get_rag_inference_handler()
        
        try:
            token_stream, chunks, subgraph_data, subgraph_info = (
                rag_inference_handler.stream_chat(
                    enhanced_query,
                    effective_owner,
                    return_subgraph=(return_subgraph or include_evidence),
                    progress_callback=emit_progress,
                    history_text=history_text if history_text else None,
                    enable_web_search=enable_web_search,
                    user_type=user_type,
                )
            )
        except TypeError:
            try:
                token_stream, chunks, subgraph_data, subgraph_info = (
                    rag_inference_handler.stream_chat(
                        enhanced_query,
                        effective_owner,
                        return_subgraph=(return_subgraph or include_evidence),
                        progress_callback=emit_progress,
                        history_text=history_text if history_text else None,
                        enable_web_search=enable_web_search,
                    )
                )
            except TypeError:
                try:
                    token_stream, chunks, subgraph_data, subgraph_info = (
                        rag_inference_handler.stream_chat(
                            enhanced_query,
                            effective_owner,
                            return_subgraph=(return_subgraph or include_evidence),
                            progress_callback=emit_progress,
                            history_text=history_text if history_text else None,
                            user_type=user_type,
                        )
                    )
                except TypeError:
                    token_stream, chunks, subgraph_data, subgraph_info = (
                        rag_inference_handler.stream_chat(
                            enhanced_query,
                            effective_owner,
                            return_subgraph=(return_subgraph or include_evidence),
                        )
                    )
        
        prepared["chunks"] = chunks
        prepared["subgraph_data"] = subgraph_data
        prepared["subgraph_info"] = subgraph_info
        prepared["raw_llm_response"] = None
        prepared["raw_mindmap_response"] = None
        emit_progress({"stage": "prepare", "status": "end"})
        emit_progress({"stage": "generate", "status": "start"})
        
        token_count = 0
        total_token_length = 0
        for chunk in token_stream:
            token_count += 1
            chunk_str = str(chunk) if chunk else ""
            total_token_length += len(chunk_str)
            if token_count <= 5:
                logger.debug(
                    "SSE token_stream chunk %d: chunk_type=%s chunk_length=%d chunk_preview=%s",
                    token_count,
                    type(chunk).__name__,
                    len(chunk_str),
                    chunk_str[:100] if chunk_str else "None",
                )
            asyncio.run_coroutine_threadsafe(
                queue.put({"kind": "token", "text": chunk}),
                loop
            )
        
        logger.info(
            "SSE token_stream collection completed: total_tokens=%d total_length=%d",
            token_count,
            total_token_length,
        )
        emit_progress({"stage": "generate", "status": "end"})
    except Exception as exc:  # noqa: BLE001
        stream_error[0] = exc
    finally:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)


def start_stream_processing(
    enhanced_query: str,
    effective_owner: Any,
    return_subgraph: bool,
    include_evidence: bool,
    history_text: str | None,
    enable_web_search: bool,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
    prepared: dict[str, Any],
    stream_error: list[Exception | None],
    emit_progress: callable
) -> None:
    """Start stream processing in a background thread."""
    ctx = contextvars.copy_context()
    threading.Thread(
        target=lambda: ctx.run(
            _run_stream_processing,
            enhanced_query,
            effective_owner,
            return_subgraph,
            include_evidence,
            history_text,
            enable_web_search,
            queue,
            loop,
            prepared,
            stream_error,
            emit_progress
        ),
        daemon=True
    ).start()
