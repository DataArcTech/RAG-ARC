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
from api.routers.rag_inference_modules.stream_chat.deepsearch_handler import load_trace_events_from_file


class MessageContent(BaseModel):
    content: str


class MessageRequest(BaseModel):
    """兼容 /api/messages 的请求格式（session_id 在请求体中）"""
    session_id: uuid.UUID
    content: str


class ChatMessageResponse(BaseModel):
    """Response model for chat messages"""
    id: uuid.UUID
    session_id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    user_type: Optional[int] = None
    content: dict
    # 支持 UUID 和字符串 ID（如 tavily-* 格式的 web search chunk IDs）
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
            # 如果验证失败，尝试修复 source_file_ids
            if hasattr(obj, "__dict__"):
                obj_dict = obj.__dict__.copy()
            elif isinstance(obj, dict):
                obj_dict = obj.copy()
            else:
                obj_dict = {}
            
            # 修复 source_file_ids：将非字符串转换为字符串
            if "source_file_ids" in obj_dict and obj_dict["source_file_ids"]:
                fixed_ids = []
                for item in obj_dict["source_file_ids"]:
                    if item is not None:
                        fixed_ids.append(str(item))
                obj_dict["source_file_ids"] = fixed_ids if fixed_ids else None
            
            # 检查是否有 DeepSearch trace file path
            if "deepsearch_trace" not in obj_dict:
                deepsearch_trace = None
                if "raw_llm_response" in obj_dict and obj_dict["raw_llm_response"]:
                    if isinstance(obj_dict["raw_llm_response"], dict):
                        trace_file_path = obj_dict["raw_llm_response"].get("deepsearch_trace_file_path")
                        if trace_file_path:
                            deepsearch_trace = load_trace_events_from_file(trace_file_path)
                
                if deepsearch_trace:
                    obj_dict["deepsearch_trace"] = deepsearch_trace
            
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
        # 使用 fallback 方法处理可能的 source_file_ids 格式问题
        result = []
        for msg in messages:
            try:
                msg_dict = msg.__dict__.copy() if hasattr(msg, "__dict__") else {}
                
                # 检查是否有 DeepSearch trace file path，如果有则加载 trace events
                deepsearch_trace = None
                if hasattr(msg, "raw_llm_response") and msg.raw_llm_response:
                    if isinstance(msg.raw_llm_response, dict):
                        trace_file_path = msg.raw_llm_response.get("deepsearch_trace_file_path")
                        if trace_file_path:
                            deepsearch_trace = load_trace_events_from_file(trace_file_path)
                
                # 如果加载成功，添加到响应中
                if deepsearch_trace:
                    msg_dict["deepsearch_trace"] = deepsearch_trace
                
                result.append(ChatMessageResponse.model_validate(msg_dict))
            except Exception as e:
                # 如果标准验证失败，尝试使用 fallback 方法
                try:
                    result.append(ChatMessageResponse.model_validate_with_fallback(msg))
                except Exception as fallback_error:
                    # 如果 fallback 也失败，记录错误但尝试创建一个基本响应
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Failed to validate message {getattr(msg, 'id', 'unknown')}: {fallback_error}. "
                        f"Original error: {e}. Attempting basic conversion."
                    )
                    # 手动创建一个安全的响应对象
                    if hasattr(msg, "__dict__"):
                        msg_dict = msg.__dict__.copy()
                    else:
                        msg_dict = {}
                    
                    # 确保 source_file_ids 是字符串列表
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
                        # 跳过这个有问题的消息，继续处理其他消息
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
        # 用户只能查询自己的消息
        messages = await get_thread_pool().run_blocking(
            get_message_handler().list_messages_by_user,
            current_user.id,
            limit,
            offset,
        )
        # 使用 fallback 方法处理可能的 source_file_ids 格式问题
        result = []
        for msg in messages:
            try:
                result.append(ChatMessageResponse.model_validate(msg))
            except Exception as e:
                try:
                    result.append(ChatMessageResponse.model_validate_with_fallback(msg))
                except Exception as fallback_error:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Failed to validate message {getattr(msg, 'id', 'unknown')}: {fallback_error}. "
                        f"Original error: {e}. Attempting basic conversion."
                    )
                    # 手动创建一个安全的响应对象
                    if hasattr(msg, "__dict__"):
                        msg_dict = msg.__dict__.copy()
                    else:
                        msg_dict = {}
                    
                    # 确保 source_file_ids 是字符串列表
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
        # 重新抛出 HTTPException，保持原有的错误响应
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
    # 验证用户权限
    session = await get_thread_pool().run_blocking(
        get_session_handler().get_session,
        session_id
    )
    if session is None:
        # 不存在，返回200但提示不存在
        return {"message": "Session not found"}
    if not validate_user_session(session, current_user):
        # 存在但不是当前用户的，返回401
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    # 删除会话（会级联删除所有关联的消息）
    success = await get_thread_pool().run_blocking(
        get_session_handler().delete_session,
        session_id
    )
    if not success:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to delete session")
    
    # 返回明确的消息，中间件会自动包装为标准响应格式
    return {"message": "Session deleted successfully"}
