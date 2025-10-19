import uuid
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    UploadFile,
    status,
)
from typing import Annotated
from api.routers.auth import get_current_user
from encapsulation.data_model.orm_models import User
from framework.register import Register
import uuid

router = APIRouter(prefix="/knowledge", tags=["files"])

registrator = Register()
account_handler = registrator.get_object("account")
knowledge_handler = registrator.get_object("knowledge")


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
