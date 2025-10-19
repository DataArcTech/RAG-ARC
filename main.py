import logging
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


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG-ARC HTTP Server", lifespan=mcp.mcp_app.lifespan)

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
