"""辅助函数：任务状态管理和取消检查"""
import logging
from datetime import datetime
from typing import Optional, Any
from encapsulation.data_model.orm_models import ChatMessageStatus
from framework.thread_pool import get_thread_pool
from api.routers.rag_inference_handlers import get_message_handler, get_session_handler
from .task_registry import ChatTaskInfo, get_chat_task_registry
from api.sse import sse_json_wrapped, sse_done

logger = logging.getLogger(__name__)


async def create_and_register_task(
    session_id: Any,
    user_message_id: Any,
    query: str,
    request_id: str
) -> Optional[ChatTaskInfo]:
    """创建任务并更新数据库状态"""
    registry = get_chat_task_registry()
    try:
        task_info = await registry.create(
            session_id=session_id,
            user_message_id=user_message_id,
            query=query,
            request_id=request_id
        )
        
        # 更新消息状态
        await get_thread_pool().run_blocking(
            get_message_handler().update_message,
            user_message_id,
            {
                "status": ChatMessageStatus.PROCESSING,
                "task_id": task_info.task_id,
                "request_id": request_id
            }
        )
        
        # 更新会话状态
        await get_thread_pool().run_blocking(
            get_session_handler().update_session,
            session_id,
            {
                "current_task_id": task_info.task_id,
                "current_task_status": "PROCESSING",
                "current_task_started_at": datetime.now()
            }
        )
        
        return task_info
    except ValueError as e:
        from fastapi import HTTPException, status
        logger.warning("Task creation failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "error": "Another task is in progress",
                "message": str(e)
            }
        )
    except Exception as e:
        logger.error("Failed to create task: %s", e)
        return None


async def check_and_handle_cancellation(
    task_info: Optional[ChatTaskInfo],
    session_id: Any,
    user_message_id: Any
) -> bool:
    """检查任务是否被取消，如果是则更新状态并返回 True"""
    if not task_info:
        return False
    
    registry = get_chat_task_registry()
    updated_info = await registry.get(task_info.task_id)
    
    if updated_info and updated_info.cancelled:
        logger.info("Task cancelled, updating status")
        await registry.mark_done(task_info.task_id, error="Task cancelled by user")
        
        # 更新消息状态
        await get_thread_pool().run_blocking(
            get_message_handler().update_message,
            user_message_id,
            {"status": ChatMessageStatus.CANCELLED}
        )
        
        # 更新会话状态
        await get_thread_pool().run_blocking(
            get_session_handler().update_session,
            session_id,
            {
                "current_task_id": None,
                "current_task_status": None,
                "current_task_started_at": None
            }
        )
        
        return True
    return False


async def yield_cancellation_event(request_id: str) -> Any:
    """生成任务取消事件"""
    yield sse_json_wrapped({
        "type": "task_cancelled",
        "message": "Task cancelled by user"
    }, request_id)
    yield sse_done()


async def mark_task_completed(
    task_info: Optional[ChatTaskInfo],
    session_id: Any,
    user_message_id: Any,
    assistant_message_id: Optional[Any] = None,
    error: Optional[str] = None
) -> None:
    """标记任务完成并更新数据库状态"""
    if not task_info:
        return
    
    registry = get_chat_task_registry()
    await registry.mark_done(task_info.task_id, assistant_message_id=assistant_message_id, error=error)
    
    # 更新消息状态
    status = ChatMessageStatus.FAILED if error else ChatMessageStatus.COMPLETED
    await get_thread_pool().run_blocking(
        get_message_handler().update_message,
        user_message_id,
        {"status": status}
    )
    
    # 更新会话状态
    await get_thread_pool().run_blocking(
        get_session_handler().update_session,
        session_id,
        {
            "current_task_id": None,
            "current_task_status": None,
            "current_task_started_at": None
        }
    )


async def cache_deepsearch_event(
    task_info: Optional[ChatTaskInfo],
    event: str
) -> None:
    """缓存 DeepSearch 事件到任务注册表"""
    if not task_info:
        return
    
    try:
        registry = get_chat_task_registry()
        import json
        try:
            if '\n\n' in event:
                event_data = json.loads(event.split('\n\n', 1)[-1])
            else:
                event_data = {"raw": event}
        except json.JSONDecodeError:
            event_data = {"raw": event}
        await registry.append_event(task_info.task_id, event_data, event_type="progress")
    except Exception:
        pass
