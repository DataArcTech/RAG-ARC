"""Task registry for chat task recovery and state management."""
import os
import json
import time
import uuid
import asyncio
import logging
from typing import Any, Optional, Dict, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ChatTaskInfo:
    """Chat task information for recovery"""
    task_id: str
    session_id: uuid.UUID
    user_message_id: uuid.UUID
    query: str
    request_id: str
    created_at_ms: int
    done: bool = False
    error: Optional[str] = None
    assistant_message_id: Optional[uuid.UUID] = None
    
    # 响应内容摘要
    response_text: str = ""
    response_length: int = 0
    events_count: int = 0
    last_event_timestamp_ms: int = 0
    
    # 关键事件缓存（最多 10 个）
    key_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Redis 存储键
    redis_events_key: Optional[str] = None
    
    # 任务控制
    cancelled: bool = False
    cancellation_reason: Optional[str] = None
    
    # 任务对象
    task: Optional[asyncio.Task] = field(default=None, repr=False)
    event_queue: Optional[asyncio.Queue] = field(default=None, repr=False)


class ChatTaskRegistry:
    """Process-local task store for chat tasks with recovery support and Redis integration."""

    # Event types that must always be retained for task-resume UX even if the in-memory
    # key-event ring buffer is full (e.g. citations/sources alignment events).
    _ALWAYS_CACHE_EVENT_TYPES = frozenset(
        {
            "initial",
            "error",
            "done",
            "progress",
            "sources",
            "final_text",
            "payload_chunk",
        }
    )
    _MAX_KEY_EVENTS = 20

    def __init__(
        self, 
        *, 
        ttl_seconds: int = 3600,
        redis_client: Optional[Any] = None,
        use_redis_events: bool = False
    ):
        self._ttl_seconds = max(300, int(ttl_seconds))
        self._items: Dict[str, ChatTaskInfo] = {}
        self._session_tasks: Dict[uuid.UUID, str] = {}
        self._lock = asyncio.Lock()
        # Finalization must be exactly-once per task_id (even if both the main coroutine and
        # the background callback attempt to finalize concurrently).
        self._finalization_locks: Dict[str, asyncio.Lock] = {}
        
        self._redis_client = redis_client
        self._use_redis_events = use_redis_events

    async def get_finalization_lock(self, task_id: str) -> asyncio.Lock:
        """Return a per-task lock used to guard final message creation and completion marking."""
        await self._cleanup()
        async with self._lock:
            lock = self._finalization_locks.get(task_id)
            if lock is None:
                lock = asyncio.Lock()
                self._finalization_locks[task_id] = lock
            return lock
    
    async def create(
        self, 
        session_id: uuid.UUID,
        user_message_id: uuid.UUID,
        query: str,
        request_id: str,
        task_id: Optional[str] = None
    ) -> ChatTaskInfo:
        """Create a new chat task."""
        task_id = task_id or f"chat_task_{uuid.uuid4().hex}"
        
        async with self._lock:
            existing_task_id = self._session_tasks.get(session_id)
            if existing_task_id:
                existing_task = self._items.get(existing_task_id)
                if existing_task and not existing_task.done and not existing_task.cancelled:
                    raise ValueError(f"Session {session_id} already has an active task: {existing_task_id}")
            
            info = ChatTaskInfo(
                task_id=task_id,
                session_id=session_id,
                user_message_id=user_message_id,
                query=query,
                request_id=request_id,
                created_at_ms=int(time.time() * 1000)
            )
            
            if self._use_redis_events and self._redis_client:
                info.redis_events_key = f"chat:task:{task_id}:events"
            
            self._items[task_id] = info
            self._session_tasks[session_id] = task_id
        
        return info
    
    async def get(self, task_id: str) -> Optional[ChatTaskInfo]:
        """Get task by task_id."""
        await self._cleanup()
        async with self._lock:
            return self._items.get(task_id)
    
    async def get_by_session(self, session_id: uuid.UUID) -> Optional[ChatTaskInfo]:
        """Get active task for a session."""
        await self._cleanup()
        async with self._lock:
            task_id = self._session_tasks.get(session_id)
            if task_id:
                return self._items.get(task_id)
        return None
    
    async def append_event(
        self, 
        task_id: str, 
        event: Dict[str, Any], 
        event_type: str = "data"
    ) -> None:
        """Append event with optimized caching strategy."""
        info = await self.get(task_id)
        if not info:
            return
        
        now_ms = int(time.time() * 1000)
        info.events_count += 1
        info.last_event_timestamp_ms = now_ms
        
        # 提取文本内容
        if event_type == "data":
            text_content = self._extract_text_from_event(event)
            if text_content:
                info.response_text += text_content
                info.response_length = len(info.response_text)
        
        # 缓存关键事件（用于断线重连 / 任务恢复）
        should_cache = (
            event_type in self._ALWAYS_CACHE_EVENT_TYPES
            or len(info.key_events) < self._MAX_KEY_EVENTS
        )
        if should_cache:
            info.key_events.append({
                "id": len(info.key_events),
                "type": event_type,
                "timestamp_ms": now_ms,
                "data": event
            })
            if len(info.key_events) > self._MAX_KEY_EVENTS:
                info.key_events.pop(0)
        
        # 存储到 Redis（如果启用）
        if self._use_redis_events and info.redis_events_key and self._redis_client:
            try:
                event_data = {
                    "id": info.events_count,
                    "type": event_type,
                    "timestamp_ms": now_ms,
                    "data": event
                }
                self._redis_client.rpush(
                    info.redis_events_key,
                    json.dumps(event_data, ensure_ascii=False)
                )
                self._redis_client.expire(info.redis_events_key, self._ttl_seconds)
            except Exception as e:
                logger.warning("Failed to store event to Redis: %s", e)
    
    def _extract_text_from_event(self, event: Dict[str, Any]) -> str:
        """Extract text content from SSE event."""
        if isinstance(event, dict):
            delta = event.get("delta", {})
            if isinstance(delta, dict):
                return delta.get("content", "") or ""
            return event.get("content", "") or ""
        elif isinstance(event, str):
            return event
        return ""
    
    async def mark_done(
        self, 
        task_id: str, 
        assistant_message_id: Optional[uuid.UUID] = None,
        error: Optional[str] = None
    ) -> None:
        """Mark task as done and clean up Redis cache."""
        info = await self.get(task_id)
        if not info:
            return
        
        info.done = True
        info.assistant_message_id = assistant_message_id
        info.error = error
        
        # 任务完成后立即删除 Redis 缓存
        if info.redis_events_key and self._redis_client:
            try:
                self._redis_client.delete(info.redis_events_key)
                logger.debug("Deleted Redis events cache for completed task: %s", task_id)
            except Exception as e:
                logger.warning("Failed to delete Redis events for task %s: %s", task_id, e)
        
        async with self._lock:
            if self._session_tasks.get(info.session_id) == task_id:
                del self._session_tasks[info.session_id]
            # Task completion: drop the finalization lock to avoid unbounded growth.
            self._finalization_locks.pop(task_id, None)
    
    async def cancel(self, task_id: str, reason: str = "User cancelled") -> bool:
        """Cancel a task and clean up Redis cache."""
        info = await self.get(task_id)
        if not info:
            return False
        
        if info.done:
            return False
        
        info.cancelled = True
        info.cancellation_reason = reason
        
        if info.task and not info.task.done():
            info.task.cancel()
        
        # 任务取消后立即删除 Redis 缓存
        if info.redis_events_key and self._redis_client:
            try:
                self._redis_client.delete(info.redis_events_key)
                logger.debug("Deleted Redis events cache for cancelled task: %s", task_id)
            except Exception as e:
                logger.warning("Failed to delete Redis events for cancelled task %s: %s", task_id, e)
        
        async with self._lock:
            if self._session_tasks.get(info.session_id) == task_id:
                del self._session_tasks[info.session_id]
            # Task cancellation: drop the finalization lock to avoid unbounded growth.
            self._finalization_locks.pop(task_id, None)
        
        return True
    
    async def get_redis_events(self, task_id: str) -> List[Dict[str, Any]]:
        """Get all events from Redis for a task."""
        if not self._use_redis_events or not self._redis_client:
            return []
        
        info = await self.get(task_id)
        if not info or not info.redis_events_key:
            return []
        
        try:
            events_json = self._redis_client.lrange(info.redis_events_key, 0, -1)
            events = []
            for event_json in events_json:
                try:
                    events.append(json.loads(event_json))
                except json.JSONDecodeError:
                    continue
            return events
        except Exception as e:
            logger.warning("Failed to get Redis events for task %s: %s", task_id, e)
            return []
    
    async def _cleanup(self) -> None:
        """Clean up expired tasks."""
        now_ms = int(time.time() * 1000)
        cutoff_ms = now_ms - (self._ttl_seconds * 1000)
        
        async with self._lock:
            expired_tasks = [
                task_id for task_id, info in self._items.items()
                if info.created_at_ms < cutoff_ms
            ]
            for task_id in expired_tasks:
                info = self._items.pop(task_id, None)
                if info:
                    self._session_tasks.pop(info.session_id, None)
                    self._finalization_locks.pop(task_id, None)
                    if info.redis_events_key and self._redis_client:
                        try:
                            self._redis_client.delete(info.redis_events_key)
                        except Exception:
                            pass


# Global registry instances.
#
# NOTE:
# `asyncio.Lock` is event-loop bound. Pytest (and some embedded runtimes) may create
# a new event loop per test/request. If we reuse one global ChatTaskRegistry across
# loops, acquiring its locks can hang indefinitely. Keep one registry per running
# loop to guarantee lock affinity.
_chat_task_registry_by_loop: dict[int, ChatTaskRegistry] = {}
_chat_task_registry_fallback: Optional[ChatTaskRegistry] = None


def _build_registry() -> ChatTaskRegistry:
    ttl = int(os.getenv("CHAT_TASK_TTL_SECONDS", "3600"))
    use_redis_events = os.getenv("CHAT_TASK_USE_REDIS_EVENTS", "false").lower() == "true"

    redis_client = None
    if use_redis_events:
        try:
            from app_registration import registry
            from core.user_management.chat_message import ChatMessageStorage

            message_storage = registry.get_object("chat_message")
            if isinstance(message_storage, ChatMessageStorage) and message_storage.cache_store:
                redis_client = message_storage.cache_store
                logger.info("ChatTaskRegistry initialized with Redis event storage (TTL=%d seconds)", ttl)
            else:
                logger.warning("CHAT_TASK_USE_REDIS_EVENTS=true but Redis not configured")
                use_redis_events = False
        except Exception as e:
            logger.warning("Failed to get Redis client: %s", e)
            use_redis_events = False

    return ChatTaskRegistry(ttl_seconds=ttl, redis_client=redis_client, use_redis_events=use_redis_events)


def get_chat_task_registry() -> ChatTaskRegistry:
    """Get global chat task registry with Redis support."""
    global _chat_task_registry_fallback
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        if _chat_task_registry_fallback is None:
            _chat_task_registry_fallback = _build_registry()
        return _chat_task_registry_fallback

    key = id(loop)
    registry = _chat_task_registry_by_loop.get(key)
    if registry is None:
        registry = _build_registry()
        _chat_task_registry_by_loop[key] = registry
    return registry
