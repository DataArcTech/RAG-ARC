"""
MCP Server implementation using FastMCP
"""
from datetime import datetime
import uuid
from fastmcp import Context, FastMCP
from typing import Dict, Any, Optional
from framework.register import Register
from api.routers.auth import SECRET_KEY, ALGORITHM
from encapsulation.data_model.orm_models import ChatMessage, User
import logging
import jwt
from jwt.exceptions import InvalidTokenError

logger = logging.getLogger(__name__)

mcp = FastMCP("RAG-ARC MCP Server")
registrator = Register()

def get_user_from_token(token: str) -> Optional[User]:
    """
    Get user from JWT token without raising HTTPException.
    Returns None if authentication fails.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            logger.error("No username found in token payload")
            return None

        logger.info(f"Decoded username from token: {username}")

        # Get account handler and fetch user
        account_handler = registrator.get_object("account")
        user = account_handler.get_user_by_username(username)

        logger.info(f"User fetched: type={type(user)}, user={user}")

        if user:
            logger.info(f"User attributes: id={getattr(user, 'id', 'NO_ID')}, user_name={getattr(user, 'user_name', 'NO_USERNAME')}")

        return user
    except InvalidTokenError as e:
        logger.error(f"Invalid token: {e}")
        return None
    except Exception as e:
        logger.error(f"Error getting user from token: {e}", exc_info=True)
        return None

def validate_user_session(session, current_user: User) -> bool:
    """Validate that session belongs to user."""
    if session is None:
        logger.warning(f"Session validation failed for user {current_user.id}")
        return False
    if session.user_id != current_user.id:
        logger.warning(f"Session validation failed for session {session.id} and user {current_user.id}")
        return False
    logger.info(f"Validating session {session.id} for user {current_user.id}")
    return True

@mcp.tool(name="hello_world", description="test")
async def hello_world_tool() -> Dict[str, Any]:
    """
    A simple Hello World tool for MCP.
    """
    return {"message": "Hello, world!"}

@mcp.tool(name="create_chat", description="Create a new chat session")
async def create_chat(auth_token: str) -> Dict[str, Any]:
    """
    Create a new chat session for authenticated user

    Args:
        auth_token: JWT authentication token

    Returns:
        dict: Response containing session_id
    """
    try:
        # Authenticate user from token
        current_user = get_user_from_token(auth_token)
        if not current_user:
            return {"isError": True, "message": "Authentication failed"}

        # Create session using session handler
        session_handler = registrator.get_object("chat_session")
        import datetime
        chat_name = f"Chat {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        # create_session returns session_id as string, not a session object
        session_id = session_handler.create_session(current_user.id, chat_name)

        return {"session_id": session_id}

    except Exception as e:
        logger.error(f"Error in create_chat function: {str(e)}")
        return {"isError": True, "message": f"Internal server error: {str(e)}"}


@mcp.tool(name="chat", description="Streamable chat interface")
async def chat(
    session_id: str,
    query: str,
    auth_token: str,
    ctx: Context = None,
) -> dict:
    """
    Chat with RAG system using authenticated user and session validation

    Args:
        session_id: UUID of the chat session
        query: User's question/query
        auth_token: JWT authentication token
        ctx: MCP context for progress reporting

    Returns:
        dict: Response containing session_id and reply
    """
    try:
        # Authenticate user from token
        current_user = get_user_from_token(auth_token)
        if not current_user:
            return {"isError": True, "message": "Authentication failed"}

        # Validate session_id format
        try:
            session_uuid = uuid.UUID(session_id)
        except ValueError:
            return {"isError": True, "message": "Invalid session_id format"}

        # Get session handler and validate session ownership
        session_handler = registrator.get_object("chat_session")
        session = session_handler.get_session(session_uuid)

        if not session or not validate_user_session(session, current_user):
            return {"isError": True, "message": "Session not found or unauthorized access"}

        await ctx.report_progress(0, 100, "generating")

        # Get RAG inference and chat with user isolation
        rag_inference = registrator.get_object("rag_inference")
        reply = rag_inference.chat(query, owner_id=current_user.id)

        # Create message in the session
        message_handler = registrator.get_object("chat_message")
        message_handler.create_message(ChatMessage(session_id=session_uuid, content={"role": "user", "content": query}, created_at=datetime.now()))
        message_handler.create_message(ChatMessage(session_id=session_uuid, content={"role": "assistant", "content": reply}, created_at=datetime.now()))

        await ctx.report_progress(100, 100, "done")

        return {"session_id": session_id, "reply": reply}

    except Exception as e:
        logger.error(f"Error in chat function: {str(e)}")
        return {"isError": True, "message": f"Internal server error: {str(e)}"}
