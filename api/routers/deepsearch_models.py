import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


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


class DeepSearchAsyncRequest(BaseModel):
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

