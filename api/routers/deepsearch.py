import uuid
from typing import Annotated, Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.routers.auth import get_current_user
from encapsulation.data_model.orm_models import User
from framework.register import Register
from core.utils.owner_guard import is_admin_owner
from core.presentation.deepsearch_payload import trim_deepsearch_payload

router = APIRouter(prefix="/deepsearch", tags=["deepsearch"])
registrator = Register()


class DeepSearchRequest(BaseModel):
    question: str = Field(..., description="User question, must be a non-empty string")
    owner_id: Optional[uuid.UUID] = Field(
        default=None,
        description="Optional owner override, limited to admin users",
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata merged into DeepSearch state",
    )
    include_evidence: bool = Field(
        default=False,
        description="When true, attach chunk/graph summaries to the response.",
    )


def _get_deepsearch_service():
    try:
        return registrator.get_object("deepsearch_service")
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="DeepSearch service is not initialized. Check DEEPSEARCH_SERVICE_CONFIG_PATH.",
        ) from exc


@router.post("/run", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def run_deepsearch(
    request: DeepSearchRequest,
    current_user: Annotated[User | None, Depends(get_current_user)],
):
    """Graph-first DeepSearch entry point."""

    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    effective_owner = request.owner_id or current_user.id
    if request.owner_id and not is_admin_owner(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only administrators may override owner scope",
        )

    service = _get_deepsearch_service()
    try:
        result = await service.run(
            request.question,
            owner_id=str(effective_owner),
            metadata=request.metadata,
        )
    except Exception as exc:  # pragma: no cover - rely on logging upstream
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"DeepSearch execution failed: {exc}",
        ) from exc

    trimmed = trim_deepsearch_payload(result, include_evidence=request.include_evidence)
    return JSONResponse(content=trimmed)
