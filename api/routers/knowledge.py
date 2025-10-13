import uuid
from fastapi import (
    APIRouter,
    HTTPException,
    UploadFile,
    status,
)
from pydantic import BaseModel
from framework.register import Register


router = APIRouter(prefix="/knowledge", tags=["files"])

registrator = Register()


@router.post(
    "",
    response_model=str,
    status_code=status.HTTP_201_CREATED,
)
def upload_file(file: UploadFile):
    try:
        print(f"Uploading file: {file.filename}")
        knowledge = registrator.get_object("knowledge")
        doc_id = knowledge.upload_file(file, 0)
        return doc_id
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
async def delete_file(file_id: str):
    knowledge = registrator.get_object("knowledge")
    try:
        knowledge.delete_file(file_id, 0)
        return None
    except HTTPException:
        # surface 404s if thrown by storage layer
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete file: {str(e)}",
        )
