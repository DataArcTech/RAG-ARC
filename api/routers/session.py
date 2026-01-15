import uuid
from datetime import datetime
from typing import Annotated, Any, Optional, List
from fastapi import APIRouter, Depends, status, HTTPException
from pydantic import BaseModel
from api.routers.auth import get_current_user
from encapsulation.data_model.orm_models import ChatMessage, ChatSession
from encapsulation.database.relational_db.postgresql import PostgreSQLDB
from encapsulation.data_model.orm_models import User
from framework.register import Register
from api.routers.auth import validate_user_session
from application.account.chat_session import ChatSessionManager
from application.account.chat_message import ChatMessageManager
from application.account.user import Account
from framework.thread_pool import get_thread_pool


class MessageContent(BaseModel):
    content: str


class MessageRequest(BaseModel):
    """Compatibility request model for `/api/messages` (session_id is in the body)."""
    session_id: uuid.UUID
    content: str


class ChatMessageResponse(BaseModel):
    """Response model for chat messages"""
    id: uuid.UUID
    session_id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    user_type: Optional[int] = None
    content: dict
    # Supports UUID and string IDs (e.g. tavily-* web-search chunk IDs).
    source_file_ids: Optional[List[str]] = None
    sources: Optional[List[dict]] = None
    subgraph_data: Optional[dict] = None
    raw_llm_response: Optional[dict] = None
    deepsearch_trace: Optional[dict] = None  # DeepSearch trace events（如果存在）
    created_at: datetime

    model_config = {"from_attributes": True}
    
    @classmethod
    def model_validate_with_fallback(cls, obj: Any) -> "ChatMessageResponse":
        """Validate model with fallback handling for source_file_ids and trace events"""
        try:
            return cls.model_validate(obj)
        except Exception as e:
            # If validation fails, try to sanitize source_file_ids.
            if hasattr(obj, "__dict__"):
                obj_dict = obj.__dict__.copy()
            elif isinstance(obj, dict):
                obj_dict = obj.copy()
            else:
                obj_dict = {}
            
            # Normalize source_file_ids: cast non-strings to strings.
            if "source_file_ids" in obj_dict and obj_dict["source_file_ids"]:
                fixed_ids = []
                for item in obj_dict["source_file_ids"]:
                    if item is not None:
                        fixed_ids.append(str(item))
                obj_dict["source_file_ids"] = fixed_ids if fixed_ids else None
            
            # 列表接口不需要返回这两个大字段，直接移除
            obj_dict.pop("deepsearch_trace", None)
            obj_dict.pop("raw_llm_response", None)
            
            return cls.model_validate(obj_dict)

router = APIRouter(prefix="/session", tags=["session"])

registry = Register()

def get_session_handler() -> ChatSessionManager:
    """Lazy loading function to get session handler after initialization."""
    return registry.get_object("chat_session")

def get_message_handler() -> ChatMessageManager:
    """Lazy loading function to get message handler after initialization."""
    return registry.get_object("chat_message")

def get_account_handler() -> Account:
    """Lazy loading function to get account handler after initialization."""
    return registry.get_object("account")

async def list_session_messages(
    session_id: uuid.UUID,
):
    """List session messages asynchronously using thread pool."""
    try:
        messages = await get_thread_pool().run_blocking(
            get_message_handler().list_messages_by_session,
            session_id
        )
        # Convert SQLAlchemy models to Pydantic models to ensure proper serialization
        # Use fallback validation to handle potential source_file_ids format issues.
        result = []
        for msg in messages:
            try:
                msg_dict = msg.__dict__.copy() if hasattr(msg, "__dict__") else {}
                
                # 列表接口不需要返回这两个大字段，移除以减小响应体积
                msg_dict.pop("deepsearch_trace", None)
                msg_dict.pop("raw_llm_response", None)
                
                result.append(ChatMessageResponse.model_validate(msg_dict))
            except Exception as e:
                # If standard validation fails, try the fallback method.
                try:
                    msg_dict = msg.__dict__.copy() if hasattr(msg, "__dict__") else {}
                    # 列表接口不需要返回这两个大字段，移除以减小响应体积
                    msg_dict.pop("deepsearch_trace", None)
                    msg_dict.pop("raw_llm_response", None)
                    result.append(ChatMessageResponse.model_validate_with_fallback(msg_dict))
                except Exception as fallback_error:
                    # If fallback also fails, log the error and try a minimal safe conversion.
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Failed to validate message {getattr(msg, 'id', 'unknown')}: {fallback_error}. "
                        f"Original error: {e}. Attempting basic conversion."
                    )
                    # Build a safe response payload manually.
                    if hasattr(msg, "__dict__"):
                        msg_dict = msg.__dict__.copy()
                    else:
                        msg_dict = {}
                    
                    # 列表接口不需要返回这两个大字段，移除以减小响应体积
                    msg_dict.pop("deepsearch_trace", None)
                    msg_dict.pop("raw_llm_response", None)
                    
                    # Ensure source_file_ids is a list of strings.
                    if "source_file_ids" in msg_dict and msg_dict["source_file_ids"]:
                        msg_dict["source_file_ids"] = [
                            str(item) if item is not None else None
                            for item in msg_dict["source_file_ids"]
                            if item is not None
                        ] or None
                    
                    try:
                        result.append(ChatMessageResponse.model_validate(msg_dict))
                    except Exception as final_error:
                        logger.error(
                            f"Failed to create ChatMessageResponse for message {getattr(msg, 'id', 'unknown')}: {final_error}"
                        )
                        # Skip this malformed message and continue.
                        continue
        return result
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error listing session messages for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list session messages: {str(e)}"
        )

@router.post("")
async def create_session(
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """Create a new chat session asynchronously using thread pool."""
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    chat_name = f"Chat {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    return await get_thread_pool().run_blocking(
        get_session_handler().create_session,
        current_user.id,
        chat_name
    )


@router.get("")
async def list_sessions(
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """List user sessions asynchronously using thread pool."""
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    return await get_thread_pool().run_blocking(
        get_session_handler().list_sessions_by_user,
        current_user.id
    )


@router.post("/{session_id}/messages")
async def create_message(
    session_id: uuid.UUID,
    message_content: MessageContent,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """Create a message asynchronously using thread pool."""
    # Validate user has access to the session
    session = await get_thread_pool().run_blocking(
        get_session_handler().get_session,
        session_id
    )
    if session is None or not validate_user_session(session, current_user):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    messages = await get_thread_pool().run_blocking(
        get_message_handler().create_message,
        ChatMessage(
            session_id=session_id,
            user_id=current_user.id,
            user_type=getattr(current_user, "type", None),
            content={"role": "user", "content": message_content.content},
            created_at=datetime.now()
        )
    )
    return messages




@router.get("/messages", response_model=List[ChatMessageResponse])
async def list_messages_by_user(
    current_user: Annotated[User, Depends(get_current_user)],
    limit: int = 100,
    offset: int = 0,
):
    """List messages by current user with pagination. Users can only query their own messages."""
    try:
        # Users can only query their own messages.
        messages = await get_thread_pool().run_blocking(
            get_message_handler().list_messages_by_user,
            current_user.id,
            limit,
            offset,
        )
        # Use fallback validation to handle potential source_file_ids format issues.
        result = []
        for msg in messages:
            try:
                msg_dict = msg.__dict__.copy() if hasattr(msg, "__dict__") else {}
                # 列表接口不需要返回这两个大字段，移除以减小响应体积
                msg_dict.pop("deepsearch_trace", None)
                msg_dict.pop("raw_llm_response", None)
                result.append(ChatMessageResponse.model_validate(msg_dict))
            except Exception as e:
                try:
                    msg_dict = msg.__dict__.copy() if hasattr(msg, "__dict__") else {}
                    # 列表接口不需要返回这两个大字段，移除以减小响应体积
                    msg_dict.pop("deepsearch_trace", None)
                    msg_dict.pop("raw_llm_response", None)
                    result.append(ChatMessageResponse.model_validate(msg_dict))
                except Exception as fallback_error:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Failed to validate message {getattr(msg, 'id', 'unknown')}: {fallback_error}. "
                        f"Original error: {e}. Attempting basic conversion."
                    )
                    # Build a safe response payload manually.
                    if hasattr(msg, "__dict__"):
                        msg_dict = msg.__dict__.copy()
                    else:
                        msg_dict = {}
                    
                    # 列表接口不需要返回这两个大字段，移除以减小响应体积
                    msg_dict.pop("deepsearch_trace", None)
                    msg_dict.pop("raw_llm_response", None)
                    
                    # Ensure source_file_ids is a list of strings.
                    if "source_file_ids" in msg_dict and msg_dict["source_file_ids"]:
                        msg_dict["source_file_ids"] = [
                            str(item) if item is not None else None
                            for item in msg_dict["source_file_ids"]
                            if item is not None
                        ] or None
                    
                    try:
                        result.append(ChatMessageResponse.model_validate(msg_dict))
                    except Exception as final_error:
                        logger.error(
                            f"Failed to create ChatMessageResponse for message {getattr(msg, 'id', 'unknown')}: {final_error}"
                        )
                        continue
        return result
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error listing messages for user {current_user.id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list messages: {str(e)}"
        )


@router.get("/{session_id}/messages", response_model=List[ChatMessageResponse])
async def list_messages(
    session_id: uuid.UUID,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """List messages asynchronously using thread pool."""
    try:
        session = await get_thread_pool().run_blocking(
            get_session_handler().get_session,
            session_id
        )
        if session is None or not validate_user_session(session, current_user):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        return await list_session_messages(session_id)
    except HTTPException:
        # Re-raise HTTPException to preserve the original error response.
        raise
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error listing messages for session {session_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list messages: {str(e)}"
        )


@router.delete("/{session_id}")
async def delete_session(
    session_id: uuid.UUID,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """Delete a session asynchronously using thread pool."""
    # Validate user permissions.
    session = await get_thread_pool().run_blocking(
        get_session_handler().get_session,
        session_id
    )
    if session is None:
        # Not found: return 200 with a clear message.
        return {"message": "Session not found"}
    if not validate_user_session(session, current_user):
        # Exists but not owned by current user: return 401.
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    # Delete the session (cascades to all associated messages).
    success = await get_thread_pool().run_blocking(
        get_session_handler().delete_session,
        session_id
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete session")
    
    # Return a clear message; middleware will wrap it into the standard response format.
    return {"message": "Session deleted successfully"}
