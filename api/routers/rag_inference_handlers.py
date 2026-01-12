import uuid
from typing import Optional

from framework.register import Register
from framework.thread_pool import get_thread_pool

from api.utils.owner_scope import get_shared_document_owner_id
from application.account.chat_message import ChatMessageManager
from application.account.chat_session import ChatSessionManager
from application.account.user import Account
from application.rag_inference.module import RAGInference
from encapsulation.data_model.orm_models import User

from api.routers.chatbot import _fallback_title, _generate_title_messages, _sanitize_title

_registrator = Register()

_session_handler: ChatSessionManager | None = None
_message_handler: ChatMessageManager | None = None
_rag_inference_handler: RAGInference | None = None


def get_session_handler() -> ChatSessionManager:
    """Lazy loading function to get session handler after initialization."""

    global _session_handler
    if _session_handler is None:
        _session_handler = _registrator.get_object("chat_session")
    return _session_handler


def get_message_handler() -> ChatMessageManager:
    """Lazy loading function to get message handler after initialization."""

    global _message_handler
    if _message_handler is None:
        _message_handler = _registrator.get_object("chat_message")
    return _message_handler


def get_rag_inference_handler() -> RAGInference:
    """Lazy loading function to get rag inference handler after initialization."""

    global _rag_inference_handler
    if _rag_inference_handler is None:
        _rag_inference_handler = _registrator.get_object("rag_inference")
    return _rag_inference_handler


def get_account_handler() -> Account:
    """Lazy loading function to get account handler after initialization."""

    return _registrator.get_object("account")


def get_default_owner_id(current_user: User) -> uuid.UUID:
    raw_type = getattr(current_user, "type", 0)
    try:
        user_type = int(raw_type) if raw_type is not None else 0
    except (TypeError, ValueError):
        user_type = 0
    if user_type == 1:
        return get_shared_document_owner_id()
    return current_user.id


async def generate_title_via_llm(
    rag_inference_handler: RAGInference,
    user_text: str,
    assistant_text: str,
) -> str:
    """Generate title for chat session (shared logic with chatbot)."""

    llm = getattr(rag_inference_handler, "llm", None)
    if llm is None:
        return _fallback_title(user_text)

    messages = _generate_title_messages(user_text, assistant_text)

    def _run():
        return llm.chat(messages)

    try:
        raw = await get_thread_pool().run_blocking(_run)
    except Exception:  # noqa: BLE001
        return _fallback_title(user_text)

    title = _sanitize_title(str(raw or ""))
    return title or _fallback_title(user_text)

