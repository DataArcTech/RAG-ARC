"""Validation functions for stream chat endpoint."""
import uuid
import logging
from fastapi import HTTPException, status
from encapsulation.data_model.orm_models import User
from api.routers.auth import validate_user_session
from api.routers.rag_inference_handlers import get_session_handler
from framework.thread_pool import get_thread_pool

logger = logging.getLogger(__name__)


async def validate_user_authentication(current_user: User | None) -> None:
    """Validate user authentication."""
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )


async def validate_user_permissions(
    current_user: User,
    return_subgraph: bool,
    include_evidence: bool
) -> None:
    """Validate user permissions for subgraph/evidence generation.
    Note: 不在此处拦截非 livingKB 用户；由调用方根据 user_type 将 return_subgraph/include_evidence 置为 False，
    使 chatKB 用户仍能正常获得对话流，仅不返回子图/证据。
    """


async def validate_session_access(
    session_id: uuid.UUID,
    current_user: User
) -> None:
    """Validate session ownership and access."""
    session = await get_thread_pool().run_blocking(
        get_session_handler().get_session,
        session_id
    )
    if session is None:
        logger.warning(
            "Session not found: session_id=%s, user_id=%s",
            session_id,
            current_user.id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Session {session_id} not found or you don't have permission to access it"
        )
    if not validate_user_session(session, current_user):
        logger.warning(
            "Session validation failed: session_id=%s, session_user_id=%s, current_user_id=%s",
            session_id,
            session.user_id,
            current_user.id
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Session {session_id} does not belong to current user. "
                   f"Session belongs to user {session.user_id}, but current user is {current_user.id}"
        )
