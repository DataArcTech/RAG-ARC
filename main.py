import logging
import uuid
import os
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from core.graph_adapter.scope_provider import configure_scope_provider

# Load environment variables from .env file
load_dotenv()
configure_scope_provider()

# 先配置日志，避免被其他模块的basicConfig覆盖
from api.utils.logging_handler import DailySizeRotatingHandler
from asgi_correlation_id.log_filters import CorrelationIdFilter

class AutoUUIDCorrelationFilter(CorrelationIdFilter):
    """自动生成 UUID 的 CorrelationIdFilter（当 correlation_id 不存在时）"""
    def filter(self, record):
        result = super().filter(record)
        if hasattr(record, 'correlation_id') and record.correlation_id == 'NO-ID':
            record.correlation_id = str(uuid.uuid4())
        return result

class BeijingFormatter(logging.Formatter):
    """使用北京时间的日志格式化器"""
    def formatTime(self, record, datefmt=None):
        beijing_tz = timezone(timedelta(hours=8))
        ct = datetime.fromtimestamp(record.created, tz=beijing_tz)
        if datefmt:
            return ct.strftime(datefmt)
        return ct.strftime('%Y-%m-%d %H:%M:%S')

correlation_filter = AutoUUIDCorrelationFilter(uuid_length=36, default_value='NO-ID')
log_base_dir = Path(__file__).parent / "log"
log_base_dir.mkdir(exist_ok=True)

file_handler = DailySizeRotatingHandler(
    base_dir=str(log_base_dir),
    maxBytes=100*1024*1024,
    backupCount=30,
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(BeijingFormatter(
    '%(asctime)s - [request_id: %(correlation_id)s] - %(name)s - %(levelname)s - %(message)s'
))
file_handler.addFilter(correlation_filter)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [request_id: %(correlation_id)s] - %(name)s - %(levelname)s - %(message)s',
    handlers=[file_handler],
    force=True
)

for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler.formatter, BeijingFormatter):
        handler.setFormatter(BeijingFormatter(
            '%(asctime)s - [request_id: %(correlation_id)s] - %(name)s - %(levelname)s - %(message)s'
        ))

for handler in logging.root.handlers:
    if correlation_filter not in handler.filters:
        handler.addFilter(correlation_filter)

original_addHandler = logging.Logger.addHandler
def addHandler_with_filter(self, handler):
    if correlation_filter not in handler.filters:
        handler.addFilter(correlation_filter)
    return original_addHandler(self, handler)
logging.Logger.addHandler = addHandler_with_filter

import app_registration

# initialize components BEFORE importing routers that depend on them
app_registration.initialize()

from api.routers import mcp
from api.routers import knowledge as knowledge_router
from api.routers import rag_inference
from api.routers import deepsearch as deepsearch_router
from api.routers import session as session_router
from api.routers import auth as auth_router
from api.routers import user as user_router
from api.routers import chatbot as chatbot_router
from asgi_correlation_id import CorrelationIdMiddleware
from asgi_correlation_id.middleware import is_valid_uuid4
from api.middleware.response_wrapper import RequestIdResponseWrapper

logger = logging.getLogger(__name__)


async def shutdown_knowledge_module():
    """Shutdown Knowledge module to flush pending BM25 chunks."""
    logger.info("Application shutting down...")
    try:
        knowledge = app_registration.registrator.get_object("knowledge")
        if knowledge and hasattr(knowledge, 'shutdown'):
            await knowledge.shutdown()
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    # Shutdown global thread pool
    try:
        from framework.thread_pool import get_thread_pool
        thread_pool = get_thread_pool()
        thread_pool.shutdown(wait=True)
    except Exception as e:
        logger.error(f"Error shutting down thread pool: {e}")


@asynccontextmanager
async def _noop_context():
    """No-op async context manager for when MCP lifespan is not available."""
    yield


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI application.
    Handles startup and shutdown events for both FastAPI and FastMCP.
    """
    # Startup
    logger.info("Application starting up...")
    
    # Get MCP lifespan if it exists
    mcp_lifespan = getattr(mcp.mcp_app, 'lifespan', None)
    
    # Use MCP lifespan if available, otherwise use a no-op context manager
    mcp_context = mcp_lifespan(mcp.mcp_app) if mcp_lifespan else _noop_context()
    
    async with mcp_context:
        yield
        # Shutdown Knowledge module to flush pending BM25 chunks
        await shutdown_knowledge_module()


app = FastAPI(title="RAG-ARC HTTP Server", lifespan=lifespan)

# Add Correlation ID middleware (must be first to capture all requests)
app.add_middleware(
    CorrelationIdMiddleware,
    header_name='X-Request-ID',
    update_request_header=False,
    generator=lambda: str(uuid.uuid4()),
    validator=is_valid_uuid4,
)

# Add Response Wrapper middleware (after CorrelationIdMiddleware to access correlation_id)
app.add_middleware(RequestIdResponseWrapper)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
    expose_headers=["X-Request-ID"],  # 暴露 request_id 给前端
)


@app.get("/")
async def health_check():
    logger.info("Health check endpoint called")
    return "ok"


# 兼容路由：/api/messages（session_id 在请求体中）
@app.post("/api/messages")
async def create_message_compat(
    message_request: "MessageRequest",
    current_user: Annotated["User | None", Depends(get_current_user)],
):
    """兼容 /api/messages 的请求格式（session_id 在请求体中）"""
    from api.routers.session import MessageRequest, get_session_handler, get_message_handler, validate_user_session
    from encapsulation.data_model.orm_models import ChatMessage
    from fastapi import HTTPException, status
    from framework.thread_pool import get_thread_pool
    
    session_id = message_request.session_id
    # Validate user has access to the session
    session = await get_thread_pool().run_blocking(
        get_session_handler().get_session,
        session_id
    )
    if session is None or not validate_user_session(session, current_user):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Unauthorized")
    
    messages = await get_thread_pool().run_blocking(
        get_message_handler().create_message,
        ChatMessage(session_id=session_id, content={"role": "user", "content": message_request.content}, created_at=datetime.now())
    )
    return messages

app.mount("/mcp", mcp.mcp_app)
app.include_router(knowledge_router.router)
app.include_router(rag_inference.router)
app.include_router(deepsearch_router.router)
app.include_router(session_router.router)
app.include_router(auth_router.router)
app.include_router(user_router.router)
app.include_router(chatbot_router.router)
