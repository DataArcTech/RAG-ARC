import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.graph_adapter.scope_provider import configure_scope_provider

# Load environment variables from .env file
load_dotenv()
configure_scope_provider()

import app_registration

# Initialize components BEFORE importing routers that depend on them
app_registration.initialize()

from api.middleware.response_wrapper import RequestIdResponseWrapper
from api.routers import auth as auth_router
from api.routers import chatbot as chatbot_router
from api.routers import deepsearch as deepsearch_router
from api.routers import knowledge as knowledge_router
from api.routers import mcp
from api.routers import rag_inference
from api.routers import session as session_router
from api.routers import user as user_router
from api.utils.logging_handler import DailySizeRotatingHandler
from asgi_correlation_id import CorrelationIdMiddleware
from asgi_correlation_id.log_filters import CorrelationIdFilter
from asgi_correlation_id.middleware import is_valid_uuid4


class BeijingFormatter(logging.Formatter):
    """Log formatter that uses Beijing time (UTC+8) for timestamps."""

    def formatTime(self, record, datefmt=None):  # noqa: N802
        beijing_tz = timezone(timedelta(hours=8))
        ct = datetime.fromtimestamp(record.created, tz=beijing_tz)
        if datefmt:
            return ct.strftime(datefmt)
        return ct.strftime("%Y-%m-%d %H:%M:%S")


def _configure_logging() -> None:
    """Configure application logging without monkey patching."""

    correlation_filter = CorrelationIdFilter(uuid_length=36, default_value="NO-ID")

    log_dir = Path(__file__).parent / "log"
    log_dir.mkdir(exist_ok=True)

    fmt = "%(asctime)s - [request_id: %(correlation_id)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = BeijingFormatter(fmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    handler = next((h for h in root.handlers if isinstance(h, DailySizeRotatingHandler)), None)
    if handler is None:
        handler = DailySizeRotatingHandler(
            base_dir=str(log_dir),
            maxBytes=100 * 1024 * 1024,
            backupCount=30,
            encoding="utf-8",
        )
        root.addHandler(handler)

    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    if not any(isinstance(f, CorrelationIdFilter) for f in getattr(handler, "filters", []) or []):
        handler.addFilter(correlation_filter)

    if not any(isinstance(f, CorrelationIdFilter) for f in getattr(root, "filters", []) or []):
        root.addFilter(correlation_filter)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        target_logger = logging.getLogger(name)
        if any(isinstance(h, DailySizeRotatingHandler) for h in target_logger.handlers):
            continue
        target_logger.addHandler(handler)


_configure_logging()

logger = logging.getLogger(__name__)


async def shutdown_knowledge_module() -> None:
    """Shutdown Knowledge module to flush pending BM25 chunks."""

    logger.info("Application shutting down...")
    try:
        knowledge = app_registration.registrator.get_object("knowledge")
        if knowledge and hasattr(knowledge, "shutdown"):
            await knowledge.shutdown()
    except Exception as exc:  # noqa: BLE001
        logger.error("Error during shutdown: %s", exc)

    # Shutdown global thread pool
    try:
        from framework.thread_pool import get_thread_pool

        get_thread_pool().shutdown(wait=True)
    except Exception as exc:  # noqa: BLE001
        logger.error("Error shutting down thread pool: %s", exc)


@asynccontextmanager
async def _noop_context():
    """No-op async context manager for when MCP lifespan is not available."""

    yield


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI startup/shutdown."""

    logger.info("Application starting up...")
    try:
        from core.utils.dependency_health import check_dependencies

        health = check_dependencies(default_mode="strict")
        checks = health.get("checks") or {}
        logger.info(
            "Dependency health: postgres=%s redis=%s neo4j=%s (mode=%s)",
            bool((checks.get("postgres") or {}).get("ok")),
            bool((checks.get("redis") or {}).get("ok")),
            bool((checks.get("neo4j") or {}).get("ok")),
            health.get("mode"),
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Dependency health check failed at startup: %s", exc)
        raise

    mcp_lifespan = getattr(mcp.mcp_app, "lifespan", None)
    mcp_context = mcp_lifespan(mcp.mcp_app) if mcp_lifespan else _noop_context()

    async with mcp_context:
        yield
        await shutdown_knowledge_module()


app = FastAPI(title="RAG-ARC HTTP Server", lifespan=lifespan)

# Add Correlation ID middleware (must be first to capture all requests)
app.add_middleware(
    CorrelationIdMiddleware,
    header_name="X-Request-ID",
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
    expose_headers=["X-Request-ID"],  # Expose request id to clients
)


@app.get("/")
async def health_check():
    logger.info("Health check endpoint called")
    return "ok"


# Compatibility route note:
# `/api/messages` (session_id in request body) was previously served here, but it conflicts with the
# chatbot SSE route and had different auth expectations. Keep it disabled unless legacy clients
# must be supported.

app.mount("/mcp", mcp.mcp_app)
app.include_router(knowledge_router.router)
app.include_router(rag_inference.router)
app.include_router(deepsearch_router.router)
app.include_router(session_router.router)
app.include_router(auth_router.router)
app.include_router(user_router.router)
app.include_router(chatbot_router.router)

