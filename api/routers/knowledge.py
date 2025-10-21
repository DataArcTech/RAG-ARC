import uuid
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    status,
    Query,
)
from typing import Annotated, Optional, List
from pydantic import BaseModel
from api.routers.auth import get_current_user
from encapsulation.data_model.orm_models import User, FileStatus
from framework.register import Register
import uuid

router = APIRouter(prefix="/knowledge", tags=["files"])

registrator = Register()
account_handler = registrator.get_object("account")
knowledge_handler = registrator.get_object("knowledge")


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

    class Config:
        from_attributes = True


class FileListResponse(BaseModel):
    """Response model for file list"""
    files: List[FileInfo]
    total: int

@router.post(
    "",
    response_model=str,
    status_code=status.HTTP_201_CREATED,
)
def upload_file(
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
        doc_id = knowledge_handler.upload_file(file, user.id)
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
    Get all files for the current user.
    
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
        files = knowledge_handler.list_user_files(
            user_id=user.id,
            limit=limit,
            offset=offset
        )
        
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
            total=len(file_infos)
        )
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve files: {str(e)}",
        )
