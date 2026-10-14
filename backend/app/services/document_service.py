import codecs
import logging
import math
import os
import zipfile
from datetime import datetime, timezone

import cloudinary
import cloudinary.uploader
from bson import ObjectId
from fastapi import UploadFile
from pymongo.collection import Collection

from app.core.config import CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET, CLOUDINARY_CLOUD_NAME
from app.models.document import ChunkingStatus, DocumentStatus, EmbeddingStatus


logger = logging.getLogger(__name__)
MAX_FILE_SIZE = 10 * 1024 * 1024
# A small compressed DOCX can expand into a very large XML payload. Keep the
# expanded ZIP content bounded separately from the uploaded file-size limit.
MAX_DOCX_UNCOMPRESSED_SIZE = 50 * 1024 * 1024
MAX_DOCX_COMPRESSION_RATIO = 100
READ_CHUNK_SIZE = 64 * 1024
GENERIC_BINARY_MIME = "application/octet-stream"
FILE_TYPES = {
    ".pdf": ("application/pdf", "raw"),
    ".docx": ("application/vnd.openxmlformats-officedocument.wordprocessingml.document", "raw"),
    ".txt": ("text/plain", "raw"),
    ".png": ("image/png", "image"),
    ".jpg": ("image/jpeg", "image"),
    ".jpeg": ("image/jpeg", "image"),
}


class InvalidDocumentError(Exception):
    pass


class DocumentTooLargeError(Exception):
    pass


class CloudinaryNotConfiguredError(Exception):
    pass


class CloudinaryUploadError(Exception):
    pass


class DocumentPersistenceError(Exception):
    pass


class InvalidDocumentIdError(Exception):
    pass


class InvalidDocumentFilterError(Exception):
    pass


class DocumentNotFoundError(Exception):
    pass


class CloudinaryDeleteError(Exception):
    pass


class InvalidCloudinaryMetadataError(Exception):
    pass


class DocumentMetadataDeletionError(Exception):
    pass


def _safe_filename(filename: str | None) -> tuple[str, str]:
    if not filename:
        raise InvalidDocumentError("A filename is required")
    filename = filename.replace("\\", "/").split("/")[-1].strip()
    if not filename or len(filename) > 255 or "\x00" in filename:
        raise InvalidDocumentError("Invalid filename")
    extension = os.path.splitext(filename)[1].lower()
    if extension not in FILE_TYPES:
        raise InvalidDocumentError("Unsupported file extension")
    return filename, extension


async def validate_upload(upload: UploadFile) -> tuple[str, str, str, int]:
    filename, extension = _safe_filename(upload.filename)
    canonical_mime, resource_type = FILE_TYPES[extension]
    declared_mime = (upload.content_type or GENERIC_BINARY_MIME).split(";", 1)[0].strip().lower()
    if declared_mime not in {canonical_mime, GENERIC_BINARY_MIME}:
        raise InvalidDocumentError("File extension and content type do not match")

    size = 0
    prefix = bytearray()
    decoder = codecs.getincrementaldecoder("utf-8")("strict") if extension == ".txt" else None
    await upload.seek(0)
    while chunk := await upload.read(READ_CHUNK_SIZE):
        size += len(chunk)
        if size > MAX_FILE_SIZE:
            raise DocumentTooLargeError
        if len(prefix) < 1024:
            prefix.extend(chunk[: 1024 - len(prefix)])
        if decoder is not None:
            if b"\x00" in chunk:
                raise InvalidDocumentError("Text files must not contain NUL bytes")
            try:
                decoder.decode(chunk, final=False)
            except UnicodeDecodeError as exc:
                raise InvalidDocumentError("Text files must use UTF-8 encoding") from exc

    if size == 0:
        raise InvalidDocumentError("Empty files are not allowed")
    if decoder is not None:
        try:
            decoder.decode(b"", final=True)
        except UnicodeDecodeError as exc:
            raise InvalidDocumentError("Text files must use UTF-8 encoding") from exc

    signatures = {
        ".pdf": b"%PDF-" in prefix,
        ".png": bytes(prefix).startswith(b"\x89PNG\r\n\x1a\n"),
        ".jpg": bytes(prefix).startswith(b"\xff\xd8\xff"),
        ".jpeg": bytes(prefix).startswith(b"\xff\xd8\xff"),
    }
    if extension in signatures and not signatures[extension]:
        raise InvalidDocumentError("File contents do not match the file type")

    if extension == ".docx":
        await upload.seek(0)
        try:
            with zipfile.ZipFile(upload.file) as archive:
                entries = archive.infolist()
                expanded_size = sum(entry.file_size for entry in entries)
                if expanded_size > MAX_DOCX_UNCOMPRESSED_SIZE:
                    raise InvalidDocumentError("DOCX expanded content exceeds the supported limit")
                if any(
                    entry.file_size > 0
                    and (entry.compress_size == 0 or entry.file_size / entry.compress_size > MAX_DOCX_COMPRESSION_RATIO)
                    for entry in entries
                ):
                    raise InvalidDocumentError("DOCX compression ratio exceeds the supported limit")
                entry_names = {entry.filename for entry in entries}
        except (OSError, zipfile.BadZipFile) as exc:
            raise InvalidDocumentError("DOCX must be a valid Office document") from exc
        if "[Content_Types].xml" not in entry_names or "word/document.xml" not in entry_names:
            raise InvalidDocumentError("DOCX must contain a Word document")

    await upload.seek(0)
    return filename, canonical_mime, resource_type, size


def _configure_cloudinary() -> None:
    if not all((CLOUDINARY_CLOUD_NAME, CLOUDINARY_API_KEY, CLOUDINARY_API_SECRET)):
        raise CloudinaryNotConfiguredError
    cloudinary.config(
        cloud_name=CLOUDINARY_CLOUD_NAME,
        api_key=CLOUDINARY_API_KEY,
        api_secret=CLOUDINARY_API_SECRET,
        secure=True,
    )


def _delete_cloudinary_asset(public_id: str, resource_type: str) -> bool:
    try:
        result = cloudinary.uploader.destroy(
            public_id,
            resource_type=resource_type,
            type="authenticated",
            invalidate=True,
        )
        return result.get("result") == "ok"
    except Exception:
        return False


def persist_uploaded_document(
    upload: UploadFile,
    *,
    filename: str,
    file_type: str,
    file_size: int,
    resource_type: str,
    uploaded_by: ObjectId,
    documents: Collection,
) -> dict:
    _configure_cloudinary()
    try:
        cloud_result = cloudinary.uploader.upload(
            upload.file,
            resource_type=resource_type,
            type="authenticated",
            folder="company-knowledge-copilot/documents",
            use_filename=False,
            unique_filename=True,
        )
    except Exception as exc:
        raise CloudinaryUploadError from exc

    public_id = cloud_result.get("public_id")
    secure_url = cloud_result.get("secure_url")
    if not public_id or not secure_url:
        if public_id and not _delete_cloudinary_asset(public_id, resource_type):
            logger.error("Cloudinary cleanup failed after an incomplete upload response")
        raise CloudinaryUploadError

    record = {
        "filename": filename,
        "file_type": file_type,
        "file_size": file_size,
        "cloudinary_url": secure_url,
        "cloudinary_public_id": public_id,
        "cloudinary_resource_type": resource_type,
        "cloudinary_delivery_type": "authenticated",
        "uploaded_by": uploaded_by,
        "uploaded_at": datetime.now(timezone.utc),
        "status": DocumentStatus.UPLOADED.value,
        "chunking_status": ChunkingStatus.PENDING.value,
        "chunking_error": None,
        "embedding_status": EmbeddingStatus.PENDING.value,
        "embedding_error": None,
        "processing_error": None,
        "processed_at": None,
    }
    try:
        documents.create_index("uploaded_by")
        result = documents.insert_one(record)
    except Exception as exc:
        if not _delete_cloudinary_asset(public_id, resource_type):
            logger.error("Cloudinary cleanup failed after MongoDB metadata persistence failed")
        raise DocumentPersistenceError from exc

    return {
        "id": str(result.inserted_id),
        "filename": record["filename"],
        "file_type": record["file_type"],
        "file_size": record["file_size"],
        "cloudinary_url": record["cloudinary_url"],
        "uploaded_by": str(record["uploaded_by"]),
        "uploaded_at": record["uploaded_at"],
        "status": record["status"],
        "chunking_status": record.get("chunking_status", ChunkingStatus.PENDING.value),
        "embedding_status": record.get("embedding_status", EmbeddingStatus.PENDING.value),
    }


def _public_document(record: dict) -> dict:
    return {
        "id": str(record["_id"]),
        "filename": record["filename"],
        "file_type": record["file_type"],
        "file_size": record["file_size"],
        "cloudinary_url": record["cloudinary_url"],
        "uploaded_by": str(record["uploaded_by"]),
        "uploaded_at": record["uploaded_at"],
        "status": record["status"],
        "chunking_status": record.get("chunking_status", ChunkingStatus.PENDING.value),
        "embedding_status": record.get("embedding_status", EmbeddingStatus.PENDING.value),
    }


def _document_filter(
    *,
    current_user: dict,
    is_admin: bool,
    requested_uploader: str | None,
    file_type: str | None,
) -> dict:
    owner_id = ObjectId(str(current_user["_id"]))
    query = {} if is_admin else {"uploaded_by": owner_id}
    if requested_uploader is not None:
        if not ObjectId.is_valid(requested_uploader):
            raise InvalidDocumentIdError
        if is_admin:
            query["uploaded_by"] = ObjectId(requested_uploader)
    if file_type is not None:
        normalized_type = file_type.strip().lower()
        if normalized_type.startswith("."):
            normalized_type = FILE_TYPES.get(normalized_type, (None, None))[0]
        if normalized_type not in {value[0] for value in FILE_TYPES.values()}:
            raise InvalidDocumentFilterError
        query["file_type"] = normalized_type
    return query


def list_documents(
    documents: Collection,
    *,
    current_user: dict,
    is_admin: bool,
    page: int,
    page_size: int,
    uploaded_by: str | None = None,
    file_type: str | None = None,
) -> dict:
    query = _document_filter(
        current_user=current_user,
        is_admin=is_admin,
        requested_uploader=uploaded_by,
        file_type=file_type,
    )
    documents.create_index([("uploaded_by", 1), ("uploaded_at", -1)])
    documents.create_index([("uploaded_at", -1)])
    total = documents.count_documents(query)
    cursor = (
        documents.find(query)
        .sort([("uploaded_at", -1), ("_id", -1)])
        .skip((page - 1) * page_size)
        .limit(page_size)
    )
    return {
        "items": [_public_document(record) for record in cursor],
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": math.ceil(total / page_size),
    }


def get_document(
    documents: Collection,
    document_id: str,
    *,
    current_user: dict,
    is_admin: bool,
) -> dict:
    if not ObjectId.is_valid(document_id):
        raise InvalidDocumentIdError
    query = {"_id": ObjectId(document_id)}
    if not is_admin:
        query["uploaded_by"] = ObjectId(str(current_user["_id"]))
    record = documents.find_one(query)
    if record is None:
        raise DocumentNotFoundError
    return _public_document(record)


def delete_document(
    documents: Collection,
    document_id: str,
    *,
    current_user: dict,
    is_admin: bool,
) -> None:
    if not ObjectId.is_valid(document_id):
        raise InvalidDocumentIdError
    object_id = ObjectId(document_id)
    query = {"_id": object_id}
    if not is_admin:
        query["uploaded_by"] = ObjectId(str(current_user["_id"]))
    record = documents.find_one(query)
    if record is None:
        raise DocumentNotFoundError

    public_id = record.get("cloudinary_public_id")
    resource_type = record.get("cloudinary_resource_type")
    delivery_type = record.get("cloudinary_delivery_type")
    if (
        not isinstance(public_id, str)
        or not public_id.strip()
        or "\x00" in public_id
        or resource_type not in {"image", "raw"}
        or delivery_type != "authenticated"
    ):
        raise InvalidCloudinaryMetadataError

    _configure_cloudinary()
    try:
        result = cloudinary.uploader.destroy(
            public_id,
            resource_type=resource_type,
            type=delivery_type,
            invalidate=True,
        )
    except Exception as exc:
        raise CloudinaryDeleteError from exc
    if not isinstance(result, dict) or result.get("result") not in {"ok", "not found"}:
        raise CloudinaryDeleteError

    try:
        documents.delete_one({"_id": object_id})
    except Exception as exc:
        try:
            marker_result = documents.update_one(
                {"_id": object_id},
                {
                    "$set": {
                        "status": DocumentStatus.DELETION_PENDING.value,
                        "cloudinary_deleted_at": datetime.now(timezone.utc),
                    }
                },
            )
            if marker_result.matched_count == 0:
                return
            logger.error("Cloudinary asset deleted but document metadata deletion failed; retry required for document_id=%s", object_id)
        except Exception:
            logger.error("Cloudinary asset deleted but metadata and retry marker could not be saved for document_id=%s", object_id)
        raise DocumentMetadataDeletionError from exc
