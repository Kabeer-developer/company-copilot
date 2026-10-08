from bson import ObjectId
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, Query, Response, UploadFile, status
from pymongo.database import Database
from starlette.concurrency import run_in_threadpool

from app.routes.auth import get_current_user, get_database
from app.models.user import UserRole
from app.schemas.document import DocumentListResponse, DocumentResponse
from app.services.document_processor import process_document
from app.services.document_service import (
    CloudinaryDeleteError,
    CloudinaryNotConfiguredError,
    CloudinaryUploadError,
    DocumentMetadataDeletionError,
    DocumentNotFoundError,
    DocumentPersistenceError,
    DocumentTooLargeError,
    InvalidCloudinaryMetadataError,
    InvalidDocumentError,
    InvalidDocumentFilterError,
    InvalidDocumentIdError,
    delete_document as delete_document_service,
    get_document,
    list_documents,
    persist_uploaded_document,
    validate_upload,
)


router = APIRouter()


@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
    database: Database = Depends(get_database),
) -> dict:
    try:
        filename, file_type, resource_type, file_size = await validate_upload(file)
    except DocumentTooLargeError as exc:
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail="Maximum file size is 10 MB") from exc
    except InvalidDocumentError as exc:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail=str(exc)) from exc

    try:
        document = await run_in_threadpool(
            persist_uploaded_document,
            file,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            resource_type=resource_type,
            uploaded_by=ObjectId(str(current_user["_id"])),
            documents=database["documents"],
        )
        background_tasks.add_task(process_document, document["id"], database["documents"])
        return document
    except CloudinaryNotConfiguredError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Cloudinary is not configured") from exc
    except CloudinaryUploadError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Cloudinary upload failed") from exc
    except DocumentPersistenceError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Document metadata could not be saved") from exc


def _is_admin(user: dict) -> bool:
    return user.get("role") == UserRole.ADMIN.value


@router.get("", response_model=DocumentListResponse)
def read_documents(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    uploaded_by: str | None = Query(default=None),
    file_type: str | None = Query(default=None),
    current_user: dict = Depends(get_current_user),
    database: Database = Depends(get_database),
) -> dict:
    try:
        return list_documents(
            database["documents"],
            current_user=current_user,
            is_admin=_is_admin(current_user),
            page=page,
            page_size=page_size,
            uploaded_by=uploaded_by,
            file_type=file_type,
        )
    except InvalidDocumentIdError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid uploaded_by ObjectId") from exc
    except InvalidDocumentFilterError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Unsupported file_type filter") from exc


@router.get("/{document_id}", response_model=DocumentResponse)
def read_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    database: Database = Depends(get_database),
) -> dict:
    try:
        return get_document(
            database["documents"],
            document_id,
            current_user=current_user,
            is_admin=_is_admin(current_user),
        )
    except InvalidDocumentIdError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document ID") from exc
    except DocumentNotFoundError as exc:
        # Conceal documents owned by another employee as well as nonexistent IDs.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found") from exc


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    database: Database = Depends(get_database),
) -> Response:
    try:
        delete_document_service(
            database["documents"],
            document_id,
            current_user=current_user,
            is_admin=_is_admin(current_user),
        )
    except InvalidDocumentIdError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid document ID") from exc
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found") from exc
    except CloudinaryNotConfiguredError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Cloudinary is not configured") from exc
    except CloudinaryDeleteError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Cloudinary deletion failed; document metadata was preserved") from exc
    except InvalidCloudinaryMetadataError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Document storage metadata is invalid") from exc
    except DocumentMetadataDeletionError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Cloudinary asset was deleted but document metadata cleanup failed; retry deletion") from exc
    return Response(status_code=status.HTTP_204_NO_CONTENT)
