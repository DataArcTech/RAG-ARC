import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

import app_registration

# initialize components BEFORE importing routers that depend on them
app_registration.initialize()

from api.routers import mcp
from api.routers import knowledge as knowledge_router
from api.routers import rag_inference
from api.routers import session as session_router
from api.routers import auth as auth_router
from api.routers import user as user_router


# Configure logging
logging.basicConfig(level=logging.INFO)
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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.get("/")
async def health_check():
    logger.info("Health check endpoint called")
    return "ok"

app.mount("/mcp", mcp.mcp_app)
app.include_router(knowledge_router.router)
app.include_router(rag_inference.router)
app.include_router(session_router.router)
app.include_router(auth_router.router)
app.include_router(user_router.router)
