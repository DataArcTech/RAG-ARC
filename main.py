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
from asgi_correlation_id import CorrelationIdMiddleware
from asgi_correlation_id.middleware import is_valid_uuid4
from asgi_correlation_id.log_filters import CorrelationIdFilter
from api.middleware.response_wrapper import RequestIdResponseWrapper
from api.utils.logging_handler import DailySizeRotatingHandler


# 自定义Formatter，使用北京时间（UTC+8）
class BeijingFormatter(logging.Formatter):
    """使用北京时间的日志格式化器"""
    def formatTime(self, record, datefmt=None):
        beijing_tz = timezone(timedelta(hours=8))
        ct = datetime.fromtimestamp(record.created, tz=beijing_tz)
        if datefmt:
            return ct.strftime(datefmt)
        return ct.strftime('%Y-%m-%d %H:%M:%S')


# Configure logging with correlation ID
# 先创建filter（UUID 格式：xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx，共36字符）
correlation_filter = CorrelationIdFilter(uuid_length=36, default_value='NO-ID')

# 配置日志文件路径（按天文件夹 + 按大小轮转）
log_base_dir = Path(__file__).parent / "log"
log_base_dir.mkdir(exist_ok=True)  # 确保基础 log 目录存在

# 创建自定义Handler（按天文件夹 + 单文件最大100MB，保留30天）
file_handler = DailySizeRotatingHandler(
    base_dir=str(log_base_dir),
    maxBytes=100*1024*1024,  # 单文件最大100MB
    backupCount=30,  # 保留30天的日志
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - [request_id: %(correlation_id)s] - %(name)s - %(levelname)s - %(message)s'
))
file_handler.addFilter(correlation_filter)

# 配置logging（移除默认handler，使用自定义handler）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [request_id: %(correlation_id)s] - %(name)s - %(levelname)s - %(message)s',
    handlers=[file_handler],  # 使用文件handler
    force=True  # 强制重新配置
)
# 为 root logger 设置自定义 formatter（确保所有日志都使用北京时间）
for handler in logging.root.handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler.formatter, BeijingFormatter):
        handler.setFormatter(BeijingFormatter(
            '%(asctime)s - [request_id: %(correlation_id)s] - %(name)s - %(levelname)s - %(message)s'
        ))

# 为所有现有和未来的handler添加correlation_id过滤器
for handler in logging.root.handlers:
    if not handler.filters:  # 避免重复添加
        handler.addFilter(correlation_filter)

# 确保新创建的handler也添加filter
original_addHandler = logging.Logger.addHandler
def addHandler_with_filter(self, handler):
    if correlation_filter not in handler.filters:
        handler.addFilter(correlation_filter)
    return original_addHandler(self, handler)
logging.Logger.addHandler = addHandler_with_filter

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

app.mount("/mcp", mcp.mcp_app)
app.include_router(knowledge_router.router)
app.include_router(rag_inference.router)
app.include_router(deepsearch_router.router)
app.include_router(session_router.router)
app.include_router(auth_router.router)
app.include_router(user_router.router)
