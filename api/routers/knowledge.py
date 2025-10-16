import uuid
from fastapi import (
    APIRouter,
    HTTPException,
    UploadFile,
    status,
    Query,
)
from pydantic import BaseModel
from typing import Optional
from framework.register import Register


router = APIRouter(prefix="/knowledge", tags=["files"])

registrator = Register()


@router.post(
    "",
    response_model=str,
    status_code=status.HTTP_201_CREATED,
)
def upload_file(
    file: UploadFile,
    owner_id: Optional[str] = Query(default="00000000-0000-0000-0000-000000000000", description="User ID (UUID format)")
):
    """
    Upload a file to the knowledge base

    Args:
        file: File to upload
        owner_id: User ID (UUID format). Defaults to a placeholder UUID.
                  After adding JWT authentication, this will be extracted from the token.

    Returns:
        Document ID
    """
    try:
        print(f"Uploading file: {file.filename} for owner_id: {owner_id}")
        knowledge = registrator.get_object("knowledge")
        # Convert string UUID to UUID object
        owner_uuid = uuid.UUID(owner_id)
        doc_id = knowledge.upload_file(file, owner_uuid)
        return doc_id
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid owner_id format: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to upload file: {str(e)}",
        )


@router.get("/{file_id}/download")
async def download_file(file_id: str):
    knowledge = registrator.get_object("knowledge")
    try:
        return knowledge.get_file(str(file_id))
    except HTTPException:
        # re-raise 404s from underlying module
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to download file: {str(e)}",
        )


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(file_id: str, owner_id: str = Query(..., description="Owner ID of the file")):
    knowledge = registrator.get_object("knowledge")
    try:
        import uuid
        owner_uuid = uuid.UUID(owner_id)
        knowledge.delete_file(file_id, owner_uuid)
        return None
    except HTTPException:
        # surface 404s and 403s if thrown by storage layer
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid owner_id format: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {str(e)}",
        )
