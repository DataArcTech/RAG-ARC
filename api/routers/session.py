import inspect
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


class SessionMessagesResponse(BaseModel):
    """分页会话消息：messages + total、total_pages、page、page_size"""
    messages: List[ChatMessageResponse]
    total: int
    total_pages: int
    page: int
    page_size: int


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


def _sort_messages_question_then_answer_newest_first(messages: List[Any]) -> List[Any]:
    """按「提问-回答」成对排序：先时间正序成对，再按轮次从新到旧；同轮内提问在前、回答在后。不改 DB，仅内存排序。"""
    if not messages:
        return messages
    by_time = sorted(messages, key=lambda m: (getattr(m, "created_at", None) or datetime.min))
    pairs: List[List[Any]] = []
    i = 0
    while i < len(by_time):
        role = (getattr(by_time[i], "content", None) or {}).get("role", "")
        if role == "user" and i + 1 < len(by_time) and (getattr(by_time[i + 1], "content", None) or {}).get("role") == "assistant":
            pairs.append([by_time[i], by_time[i + 1]])
            i += 2
        else:
            pairs.append([by_time[i]])
            i += 1
    return [m for pair in reversed(pairs) for m in pair]


async def list_session_messages(
    session_id: uuid.UUID,
    limit: int = 20,
    offset: int = 0,
    include_subgraph: bool = True,
):
    """List session messages with pagination. Returns messages, total, total_pages, page, page_size."""
    import logging
    import math
    logger = logging.getLogger(__name__)
    try:
        logger.info("[QUERY] Starting list_session_messages: session_id=%s, limit=%d, offset=%d, include_subgraph=%s",
                    session_id, limit, offset, include_subgraph)
        messages = await get_thread_pool().run_blocking(
            get_message_handler().list_messages_by_session,
            session_id,
            limit,
            offset,
        )
        messages = _sort_messages_question_then_answer_newest_first(messages or [])
        logger.info("[QUERY] Retrieved %d messages from database", len(messages) if messages else 0)
        # Convert SQLAlchemy models to Pydantic models to ensure proper serialization
        # Use fallback validation to handle potential source_file_ids format issues.
        result = []
        for msg in messages:
            try:
                # 记录从数据库读取的原始数据
                msg_id = str(getattr(msg, 'id', 'unknown'))
                msg_role = getattr(msg, 'content', {}).get('role', 'unknown') if isinstance(getattr(msg, 'content', None), dict) else 'unknown'
                db_has_subgraph = getattr(msg, 'subgraph_data', None) is not None
                db_has_sources = getattr(msg, 'sources', None) is not None
                db_has_source_file_ids = getattr(msg, 'source_file_ids', None) is not None
                logger.info(
                    "[QUERY] Reading message from DB: id=%s, role=%s, db_has_subgraph=%s, db_has_sources=%s, db_has_source_file_ids=%s, include_subgraph=%s",
                    msg_id, msg_role, db_has_subgraph, db_has_sources, db_has_source_file_ids, include_subgraph
                )
                
                # 直接使用 Pydantic 的 from_attributes，而不是 msg.__dict__
                # 这样可以正确获取所有字段，包括值为 None 的字段
                response = ChatMessageResponse.model_validate(msg)
                
                # 记录验证后的数据
                logger.info(
                    "[QUERY] After model_validate: id=%s, response_has_subgraph=%s, response_has_sources=%s, response_has_source_file_ids=%s",
                    msg_id,
                    response.subgraph_data is not None,
                    response.sources is not None,
                    response.source_file_ids is not None
                )
                
                # 列表接口不需要返回这两个大字段，移除以减小响应体积
                response.deepsearch_trace = None
                response.raw_llm_response = None
                
                # 根据参数决定是否移除subgraph_data
                if not include_subgraph:
                    response.subgraph_data = None
                    logger.info("[QUERY] Removed subgraph_data due to include_subgraph=False for message %s", msg_id)
                
                logger.info(
                    "[QUERY] Final response: id=%s, final_has_subgraph=%s, final_has_sources=%s, final_has_source_file_ids=%s",
                    msg_id,
                    response.subgraph_data is not None,
                    response.sources is not None,
                    response.source_file_ids is not None
                )
                
                result.append(response)
            except Exception as e:
                # If standard validation fails, try the fallback method.
                try:
                    # 直接使用 Pydantic 的 from_attributes
                    response = ChatMessageResponse.model_validate_with_fallback(msg)
                    
                    # 列表接口不需要返回这两个大字段，移除以减小响应体积
                    response.deepsearch_trace = None
                    response.raw_llm_response = None
                    
                    # 根据参数决定是否移除subgraph_data
                    if not include_subgraph:
                        response.subgraph_data = None
                    
                    result.append(response)
                except Exception as fallback_error:
                    # If fallback also fails, log the error and try a minimal safe conversion.
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Failed to validate message {getattr(msg, 'id', 'unknown')}: {fallback_error}. "
                        f"Original error: {e}. Attempting basic conversion."
                    )
                    # Build a safe response payload manually.
                    # 直接使用属性访问，确保所有字段都被包含
                    try:
                        try:
                            response = ChatMessageResponse.model_validate(msg)
                        except Exception:
                            # 如果还是失败，使用字典方式（兼容旧代码）
                            msg_dict = {}
                            for key in ['id', 'session_id', 'user_id', 'user_type', 'content', 
                                       'source_file_ids', 'sources', 'subgraph_data', 
                                       'raw_llm_response', 'deepsearch_trace', 'created_at']:
                                if hasattr(msg, key):
                                    msg_dict[key] = getattr(msg, key)
                            
                            # Ensure source_file_ids is a list of strings.
                            if "source_file_ids" in msg_dict and msg_dict["source_file_ids"]:
                                msg_dict["source_file_ids"] = [
                                    str(item) if item is not None else None
                                    for item in msg_dict["source_file_ids"]
                                    if item is not None
                                ] or None
                            
                            response = ChatMessageResponse.model_validate(msg_dict)
                        
                        # 列表接口不需要返回这两个大字段，移除以减小响应体积
                        response.deepsearch_trace = None
                        response.raw_llm_response = None
                        
                        # 根据参数决定是否移除subgraph_data
                        if not include_subgraph:
                            response.subgraph_data = None
                        
                        result.append(response)
                    except Exception as final_error:
                        logger.error(
                            f"Failed to create ChatMessageResponse for message {getattr(msg, 'id', 'unknown')}: {final_error}"
                        )
                        # Skip this malformed message and continue.
                        continue
        total = await get_thread_pool().run_blocking(
            get_message_handler().count_messages_by_session,
            session_id,
        )
        total_pages = math.ceil(total / limit) if limit else 1
        page_num = (offset // limit) + 1 if limit else 1
        return SessionMessagesResponse(
            messages=result,
            total=total,
            total_pages=total_pages,
            page=page_num,
            page_size=limit,
        )
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
    page: int = 1,
    page_size: int = 20,
):
    """List user sessions asynchronously using thread pool with pagination."""
    if current_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    # 参数验证
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 20
    if page_size > 100:
        page_size = 100  # 限制最大页面大小
    
    # 转换为 offset
    offset = (page - 1) * page_size

    handler = get_session_handler()
    list_kwargs: dict[str, Any] = {}
    try:
        sig = inspect.signature(handler.list_sessions_by_user)
        accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        if accepts_var_kw or "limit" in sig.parameters:
            list_kwargs["limit"] = page_size
        elif accepts_var_kw or "page_size" in sig.parameters:
            list_kwargs["page_size"] = page_size
        if accepts_var_kw or "offset" in sig.parameters:
            list_kwargs["offset"] = offset
    except Exception:  # noqa: BLE001
        # Compatibility with older handlers/test stubs that only accept user_id.
        list_kwargs = {}

    return await get_thread_pool().run_blocking(
        handler.list_sessions_by_user,
        current_user.id,
        **list_kwargs,
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
    page: int = 1,
    page_size: int = 20,
):
    """List messages by current user with pagination. Users can only query their own messages."""
    try:
        # 参数验证
        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 20
        if page_size > 100:
            page_size = 100  # 限制最大页面大小
        
        # 转换为 offset
        offset = (page - 1) * page_size
        
        # Users can only query their own messages.
        messages = await get_thread_pool().run_blocking(
            get_message_handler().list_messages_by_user,
            current_user.id,
            page_size,
            offset,
        )
        # Use fallback validation to handle potential source_file_ids format issues.
        result = []
        for msg in messages:
            try:
                # 直接使用 Pydantic 的 from_attributes
                response = ChatMessageResponse.model_validate(msg)
                
                # 列表接口不需要返回这两个大字段，移除以减小响应体积
                response.deepsearch_trace = None
                response.raw_llm_response = None
                
                result.append(response)
            except Exception as e:
                try:
                    # 直接使用 Pydantic 的 from_attributes
                    response = ChatMessageResponse.model_validate_with_fallback(msg)
                    
                    # 列表接口不需要返回这两个大字段，移除以减小响应体积
                    response.deepsearch_trace = None
                    response.raw_llm_response = None
                    
                    result.append(response)
                except Exception as fallback_error:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.warning(
                        f"Failed to validate message {getattr(msg, 'id', 'unknown')}: {fallback_error}. "
                        f"Original error: {e}. Attempting basic conversion."
                    )
                    # Build a safe response payload manually.
                    try:
                        try:
                            response = ChatMessageResponse.model_validate(msg)
                        except Exception:
                            # 如果还是失败，使用字典方式（兼容旧代码）
                            msg_dict = {}
                            for key in ['id', 'session_id', 'user_id', 'user_type', 'content', 
                                       'source_file_ids', 'sources', 'subgraph_data', 
                                       'raw_llm_response', 'deepsearch_trace', 'created_at']:
                                if hasattr(msg, key):
                                    msg_dict[key] = getattr(msg, key)
                            
                            # Ensure source_file_ids is a list of strings.
                            if "source_file_ids" in msg_dict and msg_dict["source_file_ids"]:
                                msg_dict["source_file_ids"] = [
                                    str(item) if item is not None else None
                                    for item in msg_dict["source_file_ids"]
                                    if item is not None
                                ] or None
                            
                            response = ChatMessageResponse.model_validate(msg_dict)
                        
                        # 列表接口不需要返回这两个大字段，移除以减小响应体积
                        response.deepsearch_trace = None
                        response.raw_llm_response = None
                        
                        result.append(response)
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


@router.get("/{session_id}/messages", response_model=SessionMessagesResponse)
async def list_messages(
    session_id: uuid.UUID,
    current_user: Annotated[User | None, Depends(get_current_user)],
    page: int = 1,
    page_size: int = 20,
    include_subgraph: bool = True,
):
    """List messages asynchronously using thread pool with pagination.
    
    Args:
        include_subgraph: If True, returns subgraph_data field. Default True.
    """
    try:
        session = await get_thread_pool().run_blocking(
            get_session_handler().get_session,
            session_id
        )
        if session is None or not validate_user_session(session, current_user):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        
        # 参数验证
        if page < 1:
            page = 1
        if page_size < 1:
            page_size = 20
        if page_size > 100:
            page_size = 100  # 限制最大页面大小
        
        # 转换为 offset
        offset = (page - 1) * page_size
        
        return await list_session_messages(session_id, page_size, offset, include_subgraph)
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


@router.get("/{session_id}/messages/{message_id}/subgraph")
async def get_message_subgraph(
    session_id: uuid.UUID,
    message_id: uuid.UUID,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """Get subgraph_data for a specific message."""
    try:
        # 验证会话权限
        session = await get_thread_pool().run_blocking(
            get_session_handler().get_session,
            session_id
        )
        if session is None or not validate_user_session(session, current_user):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
        
        # 获取消息
        message = await get_thread_pool().run_blocking(
            get_message_handler().get_message,
            message_id
        )
        
        if message is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Message not found")
        
        # 验证消息属于该会话
        if message.session_id != session_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Message does not belong to this session")
        
        # 返回subgraph_data
        subgraph_data = getattr(message, "subgraph_data", None)
        return {"subgraph_data": subgraph_data}
    except HTTPException:
        raise
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting message subgraph for message {message_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get message subgraph: {str(e)}"
        )
