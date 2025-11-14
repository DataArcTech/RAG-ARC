from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    status,
    Query,
    Body,
)
from typing import Annotated, Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field
from api.routers.auth import get_current_user
from encapsulation.data_model.orm_models import (
    User,
    Department,
    FilePermission,
    PermissionReceiverType,
    PermissionType
)
from framework.register import Register
import uuid
from application.knowledge.module import Knowledge
from application.account.user import Account

router = APIRouter(prefix="/knowledge", tags=["files"])

registrator = Register()
account_handler: Account = registrator.get_object("account")
knowledge_handler: Knowledge = registrator.get_object("knowledge")


# Response models
class FileInfo(BaseModel):
    """Response model for file information"""
    file_id: str
    filename: str
    status: str
    created_at: str
    updated_at: str
    file_size: int
    content_type: str

    model_config = {"from_attributes": True}


class FileListResponse(BaseModel):
    """Response model for file list"""
    files: List[FileInfo]
    total: int

@router.post(
    "",
    response_model=str,
    status_code=status.HTTP_201_CREATED,
)
async def upload_file(
    file: UploadFile,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """
    Upload a file to the knowledge base

    Args:
        file: File to upload
        owner_id: User ID string. Defaults to a placeholder string.
                  After adding JWT authentication, this will be extracted from the token.

    Returns:
        Document ID
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    try:
        print(f"Uploading file: {file.filename} for owner_id: {user.id}")
        # Convert string UUID to UUID object
        doc_id = await knowledge_handler.upload_file(file, user.id)
        return doc_id
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid user.id format: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}",
        )


@router.get("/{file_id}/download")
async def download_file(file_id: str, user: Annotated[User | None, Depends(get_current_user)]):
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    try:
        return knowledge_handler.get_file(file_id, user.id)
    except HTTPException:
        # re-raise 404s from underlying module
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download file: {str(e)}",
        )


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(file_id: str, user: Annotated[User | None, Depends(get_current_user)]):
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    try:
        knowledge_handler.delete_file(file_id, user.id)
        return None
    except HTTPException:
        # surface 404s and 403s if thrown by storage layer
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid user.id format: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {str(e)}",
        )

@router.get(
    "/list_files",
    response_model=FileListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_files(
    user: Annotated[User | None, Depends(get_current_user)],
    limit: Optional[int] = Query(default=100, ge=1, le=1000, description="Maximum number of files to return"),
    offset: Optional[int] = Query(default=0, ge=0, description="Number of files to skip"),
):
    """
    Get all files accessible to the current user (files with permissions only).
    
    Returns a list of files with their metadata including:
    - file_id: Unique identifier for the file
    - filename: Original filename
    - status: Current processing status (STORED, PARSED, CHUNKED, INDEXED, FAILED, DELETED)
    - created_at: Timestamp when file was uploaded
    - updated_at: Timestamp when file was last updated
    - file_size: Size of the file in bytes
    - content_type: MIME type of the file
    
    Args:
        user: Current authenticated user (automatically injected)
        limit: Maximum number of files to return (default: 100, max: 1000)
        offset: Number of files to skip for pagination (default: 0)
        
    Returns:
        FileListResponse with list of files and total count
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )
    try:
        # Get files for current page
        files = knowledge_handler.list_user_files(
            user_id=user.id,
            limit=limit,
            offset=offset
        )
        
        # Get total count of files for the user
        total_count = knowledge_handler.count_user_files(user.id)
        
        # Convert FileMetadata objects to FileInfo response models
        file_infos = [
            FileInfo(
                file_id=file.file_id,
                filename=file.filename,
                status=file.status.value,  # Convert enum to string
                created_at=file.created_at.isoformat(),
                updated_at=file.updated_at.isoformat(),
                file_size=file.file_size,
                content_type=file.content_type
            )
            for file in files
        ]
        
        return FileListResponse(
            files=file_infos,
            total=total_count
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve files: {str(e)}",
        )


class IndexTriggerRequest(BaseModel):
    """Request model for triggering indexing"""
    file_ids: List[str]

class IndexTriggerResponse(BaseModel):
    """Response model for index triggering results"""
    message: str

class GraphExportRequest(BaseModel):
    """Request model for graph export"""
    max_nodes: int = 500
    max_edges: int = 2000
    include_node_types: Optional[List[str]] = None  # e.g., ['chunk', 'entity', 'fact']


@router.post(
    "/trigger_indexing",
    response_model=IndexTriggerResponse,
    status_code=status.HTTP_200_OK,
)
async def trigger_indexing(
    request: IndexTriggerRequest,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """
    Trigger indexing for multiple files.

    Args:
        request: IndexTriggerRequest containing list of file IDs
        user: Current authenticated user

    Returns:
        IndexTriggerResponse with indexing results
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    if not request.file_ids:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="file_ids list cannot be empty"
        )

    try:
        result = await knowledge_handler.trigger_indexing(request.file_ids, user.id)
        
        return IndexTriggerResponse(
            message=result
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions (404, 403)
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to trigger indexing: {str(e)}",
        )


@router.post("/graph/export", status_code=status.HTTP_200_OK)
def export_knowledge_graph(
    request: GraphExportRequest,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """
    Export the complete knowledge graph for the current user

    Args:
        request: GraphExportRequest with export parameters
        user: Current authenticated user

    Returns:
        Graph data in Cytoscape.js format with nodes and edges
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    try:
        # Get the RAG inference handler to access the retriever
        rag_inference = registrator.get_object("rag_inference")

        # Find graph_store from retriever (support both direct and multipath retrievers)
        graph_store = None

        # Check if retriever has graph_store directly
        if hasattr(rag_inference.retriever, 'graph_store'):
            graph_store = rag_inference.retriever.graph_store
        # Check if it's a multipath retriever with sub-retrievers
        elif hasattr(rag_inference.retriever, 'retrievers'):
            # Find the first retriever with graph_store
            for sub_retriever in rag_inference.retriever.retrievers:
                if hasattr(sub_retriever, 'graph_store'):
                    graph_store = sub_retriever.graph_store
                    break

        if graph_store is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current retriever does not support graph visualization"
            )

        # Import appropriate GraphExporter based on graph_store type
        # Check by class name to avoid import issues
        graph_store_class_name = graph_store.__class__.__name__

        if graph_store_class_name == 'PrunedHippoRAGNeo4jStore':
            from encapsulation.database.utils.graph_export_utils_neo4j import GraphExporterNeo4j as GraphExporter
        else:
            from encapsulation.database.utils.graph_export_utils import GraphExporter

        # Export full graph
        graph_data = GraphExporter.export_full_graph(
            graph_store=graph_store,
            max_nodes=request.max_nodes,
            max_edges=request.max_edges,
            include_node_types=request.include_node_types
        )

        return graph_data

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export knowledge graph: {str(e)}",
        )


# ==================== FILE PERMISSION MANAGEMENT ====================

class DepartmentInfo(BaseModel):
    """Department information model for API responses"""
    id: str
    name: str
    description: Optional[str] = None
    path: str

    model_config = {"from_attributes": True}


class UserInfo(BaseModel):
    """User information model for API responses"""
    id: str
    user_name: str
    department: Optional[DepartmentInfo] = None
    status: str

    model_config = {"from_attributes": True}


class GrantPermissionRequest(BaseModel):
    """Request model for granting file permission"""
    receiver_type: PermissionReceiverType = Field(..., description="Type of receiver: 'user', 'department', or 'all'")
    permission_type: PermissionType = Field(..., description="Type of permission: 'view' or 'edit'")
    user_id: Optional[str] = Field(None, description="User ID if receiver_type is 'user'")
    department_id: Optional[str] = Field(None, description="Department ID if receiver_type is 'department'")


class GrantPermissionResponse(BaseModel):
    """Response model for granting file permission"""
    permission_id: str
    message: str


class RevokePermissionRequest(BaseModel):
    """Request model for revoking file permission"""
    receiver_type: Optional[PermissionReceiverType] = Field(None, description="Type of receiver: 'user', 'department', or 'all'")
    user_id: Optional[str] = Field(None, description="User ID if receiver_type is 'user'")
    department_id: Optional[str] = Field(None, description="Department ID if receiver_type is 'department'")


class PermissionInfo(BaseModel):
    """Response model for permission information"""
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
    """Response model for permission list"""
    permissions: List[PermissionInfo]
    total: int


class CheckAccessResponse(BaseModel):
    """Response model for access check"""
    has_access: bool
    permission_type: Optional[str] = None


@router.post(
    "/{file_id}/permissions/grant",
    response_model=GrantPermissionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def grant_file_permission(
    file_id: str,
    request: GrantPermissionRequest,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """
    Grant file permission to a user, department, or all users.
    
    Only users with EDIT permission can grant permissions.
    
    Args:
        file_id: File ID to grant permission for
        request: GrantPermissionRequest with permission details
        user: Current authenticated user (must have EDIT permission)
    
    Returns:
        GrantPermissionResponse with permission ID and message
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    if knowledge_handler.check_file_access(file_id, user.id) != PermissionType.EDIT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to grant permissions for this file"
        )

    # Pydantic automatically validates and converts enum types
    receiver_type: PermissionReceiverType = request.receiver_type
    permission_type: PermissionType = request.permission_type

    # Parse and validate required fields based on receiver_type
    receiver_user_id = None
    receiver_department_id = None
    
    if receiver_type == PermissionReceiverType.USER:
        if not request.user_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="user_id is required when receiver_type is 'user'"
            )
        try:
            receiver_user_id = uuid.UUID(request.user_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid user_id format: {request.user_id}"
            )
    elif receiver_type == PermissionReceiverType.DEPARTMENT:
        if not request.department_id:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="department_id is required when receiver_type is 'department'"
            )
        try:
            receiver_department_id = uuid.UUID(request.department_id)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid department_id format: {request.department_id}"
            )
    # For ALL receiver_type, both user_id and department_id should be None

    try:
        permission_id = knowledge_handler.grant_file_permission(
            file_id=file_id,
            receiver_type=receiver_type,
            permission_type=permission_type,
            granted_by=user.id,
            user_id=receiver_user_id,
            department_id=receiver_department_id
        )
        if permission_id:
            return GrantPermissionResponse(
                permission_id=str(permission_id),
                message=f"Permission granted successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to grant permission"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to grant permission: {str(e)}",
        )


@router.delete(
    "/permissions/{permission_id}",
    status_code=status.HTTP_200_OK,
)
async def revoke_file_permission(
    permission_id: str,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """
    Revoke a file permission by permission ID.
    
    Only users with EDIT permission can revoke permissions.
    
    Args:
        permission_id: Permission ID to revoke
        user: Current authenticated user (must have EDIT permission)
    
    Returns:
        Success message
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    try:
        perm_id = uuid.UUID(permission_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid permission_id format: {permission_id}"
        )

    file_id = knowledge_handler.get_file_id_by_permission_id(perm_id)
    if not file_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Permission not found"
        )

    # Only users with EDIT permission can revoke permissions
    permission_type = knowledge_handler.check_file_access(file_id, user.id)
    if permission_type != PermissionType.EDIT:
        raise HTTPException(status_code=403, detail="Only users with EDIT permission can revoke permissions")

    try:
        knowledge_handler.revoke_file_permission(perm_id, user.id)
        return {"message": "Permission revoked successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to revoke permission: {str(e)}",
        )

@router.get(
    "/{file_id}/permissions",
    response_model=PermissionListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_file_permissions(
    file_id: str,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """
    List all permissions for a specific file.
    
    Users with VIEW or EDIT permission can view permissions.
    
    Args:
        file_id: File ID to list permissions for
        user: Current authenticated user (must have VIEW or EDIT permission)
    
    Returns:
        PermissionListResponse with list of permissions
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    # Check if user has VIEW permission to list permissions
    permission_type = knowledge_handler.check_file_access(file_id, user.id)
    if permission_type is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to list permissions for this file"
        )
    try:
        permissions = knowledge_handler.list_file_permissions(file_id, user.id)
        
        permission_infos = []
        for perm in permissions:
            # Build UserInfo if user relationship is loaded and receiver type is USER
            user_info = None
            if perm.permission_receiver_type == PermissionReceiverType.USER and perm.user:
                # Build DepartmentInfo if user has a department
                user_department_info = None
                if perm.user.department:
                    user_department_info = DepartmentInfo(
                        id=str(perm.user.department.id),
                        name=perm.user.department.name,
                        description=perm.user.department.description,
                        path=perm.user.department.path
                    )
                
                user_info = UserInfo(
                    id=str(perm.user.id),
                    user_name=perm.user.user_name,
                    department=user_department_info,
                    status=perm.user.status.value
                )
            
            # Build DepartmentInfo if department relationship is loaded and receiver type is DEPARTMENT
            department_info = None
            if perm.permission_receiver_type == PermissionReceiverType.DEPARTMENT and perm.department:
                department_info = DepartmentInfo(
                    id=str(perm.department.id),
                    name=perm.department.name,
                    description=perm.department.description,
                    path=perm.department.path
                )
            
            permission_infos.append(
                PermissionInfo(
                    permission_id=str(perm.id),
                    file_id=perm.file_id,
                    receiver_type=perm.permission_receiver_type.value,
                    permission_type=perm.permission_type.value,
                    user=user_info,
                    department=department_info,
                    granted_by=str(perm.granted_by),
                    granted_at=perm.granted_at.isoformat()
                )
            )
        
        return PermissionListResponse(
            permissions=permission_infos,
            total=len(permission_infos)
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list permissions: {str(e)}",
        )


@router.get(
    "/{file_id}/permissions/check",
    response_model=CheckAccessResponse,
    status_code=status.HTTP_200_OK,
)
async def check_file_access(
    file_id: str,
    user: Annotated[User | None, Depends(get_current_user)],
):
    """
    Check if the current user has access to a file and return the permission type.
    
    Args:
        file_id: File ID to check
        user: Current authenticated user
    
    Returns:
        CheckAccessResponse with access status and permission type
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    try:
        permission_type = knowledge_handler.check_file_access(file_id, user.id)
        
        return CheckAccessResponse(
            has_access=permission_type is not None,
            permission_type=permission_type.value if permission_type else None
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to check access: {str(e)}",
        )


@router.put(
    "/permissions/{permission_id}",
    status_code=status.HTTP_200_OK,
)
async def update_file_permission(
    permission_id: str,
    user: Annotated[User | None, Depends(get_current_user)],
    permission_type: PermissionType = Body(..., embed=True, description="New permission type: 'view' or 'edit'"),
):
    """
    Update an existing file permission.
    
    Only users with EDIT permission can update permissions.
    
    Args:
        permission_id: Permission ID to update
        permission_type: New permission type ('view' or 'edit')
        user: Current authenticated user (must have EDIT permission)
    
    Returns:
        Success message
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    try:
        perm_id = uuid.UUID(permission_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid permission_id format: {permission_id}"
        )

    # Get file_id first to check ownership
    file_id = knowledge_handler.get_file_id_by_permission_id(perm_id)
    if not file_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Permission not found"
        )

    # Only file editor can update permissions
    if knowledge_handler.check_file_access(file_id, user.id) != PermissionType.EDIT:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not allowed to update permissions for this file"
        )

    # Pydantic automatically validates and converts enum types
    try:
        result = knowledge_handler.update_file_permission(
            permission_id=perm_id,
            permission_type=permission_type,
            user_id=user.id
        )
        
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Permission not found"
            )
        
        return {"message": "Permission updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update permission: {str(e)}",
        )
