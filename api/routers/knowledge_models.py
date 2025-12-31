from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from config.output_limits import KNOWLEDGE_GRAPH_EXPORT_MAX_EDGES, KNOWLEDGE_GRAPH_EXPORT_MAX_NODES
from encapsulation.data_model.orm_models import PermissionReceiverType, PermissionType


class FileInfo(BaseModel):
    """Response model for file information."""

    file_id: str
    filename: str
    status: str
    created_at: str
    updated_at: str
    file_size: int
    content_type: str

    model_config = {"from_attributes": True}


class FileListResponse(BaseModel):
    """Response model for file list."""

    files: List[FileInfo]
    total: int


class FileTaskStatusResponse(BaseModel):
    file_id: str
    file_status: Optional[str] = None
    task_run_id: Optional[str] = None
    task_state: Optional[str] = None
    progress_percent: Optional[int] = None
    error_message: Optional[str] = None
    updated_at_ms: Optional[int] = None


class TaskRunStatusResponse(BaseModel):
    run_id: str
    task_type: Optional[str] = None
    state: Optional[str] = None
    progress_percent: Optional[int] = None
    error_message: Optional[str] = None
    resource_id: Optional[str] = None
    updated_at_ms: Optional[int] = None


class IndexTriggerRequest(BaseModel):
    """Request model for triggering indexing."""

    file_ids: List[str]


class IndexTriggerResponse(BaseModel):
    """Response model for index triggering results."""

    message: str


class GraphExportRequest(BaseModel):
    """Request model for graph export."""

    max_nodes: int = Field(default=500, ge=1, le=KNOWLEDGE_GRAPH_EXPORT_MAX_NODES)
    max_edges: int = Field(default=2000, ge=0, le=KNOWLEDGE_GRAPH_EXPORT_MAX_EDGES)
    include_node_types: Optional[List[str]] = None  # e.g., ['chunk', 'entity', 'fact']


class MindmapNode(BaseModel):
    """Mind map node structure."""

    id: str
    name: str
    category: str
    weight: int


class MindmapEdge(BaseModel):
    """Mind map edge structure."""

    id: str
    source: str
    target: str
    relation: str
    weight: float


class MindmapExportRequest(BaseModel):
    """Request model for exporting merged mind map."""

    file_id: str


class MindmapExportResponse(BaseModel):
    """Response model for exported mind map."""

    tsv: str
    nodes: List[MindmapNode]
    edges: List[MindmapEdge]


class DepartmentInfo(BaseModel):
    """Department information model for API responses."""

    id: str
    name: str
    description: Optional[str] = None
    path: str

    model_config = {"from_attributes": True}


class UserInfo(BaseModel):
    """User information model for API responses."""

    id: str
    user_name: str
    department: Optional[DepartmentInfo] = None
    status: str

    model_config = {"from_attributes": True}


class GrantPermissionRequest(BaseModel):
    """Request model for granting file permission."""

    receiver_type: PermissionReceiverType = Field(..., description="Type of receiver: 'user', 'department', or 'all'")
    permission_type: PermissionType = Field(..., description="Type of permission: 'view' or 'edit'")
    user_id: Optional[str] = Field(None, description="User ID if receiver_type is 'user'")
    department_id: Optional[str] = Field(None, description="Department ID if receiver_type is 'department'")


class GrantPermissionResponse(BaseModel):
    """Response model for granting file permission."""

    permission_id: str
    message: str


class RevokePermissionRequest(BaseModel):
    """Request model for revoking file permission."""

    receiver_type: Optional[PermissionReceiverType] = Field(
        None, description="Type of receiver: 'user', 'department', or 'all'"
    )
    user_id: Optional[str] = Field(None, description="User ID if receiver_type is 'user'")
    department_id: Optional[str] = Field(None, description="Department ID if receiver_type is 'department'")


class PermissionInfo(BaseModel):
    """Response model for permission information."""

    permission_id: str
    file_id: str
    receiver_type: str
    permission_type: str
    user: Optional[UserInfo] = None
    department: Optional[DepartmentInfo] = None
    granted_by: str
    granted_at: str

    model_config = {"from_attributes": True}


class PermissionListResponse(BaseModel):
    """Response model for permission list."""

    permissions: List[PermissionInfo]
    total: int


class CheckAccessResponse(BaseModel):
    """Response model for access check."""

    has_access: bool
    permission_type: Optional[str] = None


JsonDict = Dict[str, Any]

