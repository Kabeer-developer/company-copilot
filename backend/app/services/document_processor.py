import logging
import re
import tempfile
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import BinaryIO

import cloudinary.utils
from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.collection import Collection

from app.extractors import ExtractionError
from app.extractors.docx_extractor import extract_docx
from app.extractors.image_ocr import extract_image_text
from app.extractors.pdf_extractor import extract_pdf
from app.extractors.txt_extractor import extract_txt
from app.models.document import ChunkingStatus, DocumentStatus, EmbeddingStatus
from app.services.chunk_service import chunk_document
from app.services.document_service import FILE_TYPES, CloudinaryNotConfiguredError, _configure_cloudinary
from app.services.embedding_service import embed_document


logger = logging.getLogger(__name__)
MAX_STORED_FILE_SIZE = 10 * 1024 * 1024
DOWNLOAD_CHUNK_SIZE = 64 * 1024
MIME_TO_EXTENSION = {mime: extension for extension, (mime, _) in FILE_TYPES.items()}


class DocumentProcessingError(Exception):
    pass


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _download_cloudinary_asset(record: dict) -> BinaryIO:
    public_id = record.get("cloudinary_public_id")
    resource_type = record.get("cloudinary_resource_type")
    delivery_type = record.get("cloudinary_delivery_type")
    extension = MIME_TO_EXTENSION.get(record.get("file_type"))
    if (
        not isinstance(public_id, str)
        or not public_id.strip()
        or resource_type not in {"image", "raw"}
        or delivery_type != "authenticated"
        or extension is None
    ):
        raise DocumentProcessingError("Cloudinary document metadata is incomplete or invalid.")

    downloaded = None
    try:
        _configure_cloudinary()
        expires_at = int(time.time()) + 300
        signed_url = cloudinary.utils.private_download_url(
            public_id,
            extension.lstrip("."),
            resource_type=resource_type,
            type=delivery_type,
            expires_at=expires_at,
        )
        downloaded = tempfile.SpooledTemporaryFile(max_size=2 * 1024 * 1024, mode="w+b")
        total_size = 0
        with urllib.request.urlopen(signed_url, timeout=30) as response:
            while chunk := response.read(DOWNLOAD_CHUNK_SIZE):
                total_size += len(chunk)
                if total_size > MAX_STORED_FILE_SIZE:
                    downloaded.close()
                    raise DocumentProcessingError("Stored file exceeds the supported 10 MB processing limit.")
                downloaded.write(chunk)
        if total_size == 0:
            downloaded.close()
            raise DocumentProcessingError("Cloudinary returned an empty file.")
        downloaded.seek(0)
        return downloaded
    except DocumentProcessingError:
        if downloaded is not None:
            downloaded.close()
        raise
    except CloudinaryNotConfiguredError as exc:
        if downloaded is not None:
            downloaded.close()
        raise DocumentProcessingError("Cloudinary is not configured for document processing.") from exc
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
        if downloaded is not None:
            downloaded.close()
        raise DocumentProcessingError("Could not download the document from Cloudinary.") from exc
    except Exception as exc:
        if downloaded is not None:
            downloaded.close()
        raise DocumentProcessingError("Could not download the document from Cloudinary.") from exc


def extract_document_text(record: dict, stream: BinaryIO) -> str:
    extension = MIME_TO_EXTENSION.get(record.get("file_type"))
    if extension is None:
        raise DocumentProcessingError("Unsupported stored document type.")

    try:
        if extension == ".pdf":
            extracted = extract_pdf(stream)
        elif extension == ".docx":
            extracted = extract_docx(stream)
        elif extension == ".txt":
            extracted = extract_txt(stream)
        elif extension in {".png", ".jpg", ".jpeg"}:
            extracted = extract_image_text(stream)
        else:
            raise DocumentProcessingError("Unsupported stored document type.")
    except ExtractionError as exc:
        raise DocumentProcessingError(str(exc)) from exc
    normalized = normalize_text(extracted)
    if not normalized:
        raise DocumentProcessingError("No extractable text was found in the document.")
    return normalized


def _mark_processing_failed(documents: Collection, document_id: ObjectId, message: str) -> None:
    try:
        documents.update_one(
            {"_id": document_id, "status": DocumentStatus.PROCESSING.value},
            {
                "$set": {
                    "status": DocumentStatus.FAILED.value,
                    "processing_error": message[:500],
                },
                "$unset": {"extracted_text": "", "processed_at": ""},
            },
        )
    except Exception:
        logger.error("Could not store document processing failure for document_id=%s", document_id)


def process_document(document_id: str, documents: Collection) -> None:
    if not ObjectId.is_valid(document_id):
        logger.error("Background processing received an invalid document id")
        return
    object_id = ObjectId(document_id)
    try:
        record = documents.find_one_and_update(
            {"_id": object_id, "status": DocumentStatus.UPLOADED.value},
            {
                "$set": {
                    "status": DocumentStatus.PROCESSING.value,
                    "processing_error": None,
                },
                "$unset": {"extracted_text": "", "processed_at": ""},
            },
            return_document=ReturnDocument.AFTER,
        )
    except Exception:
        logger.error("Could not claim document for processing; document_id=%s", object_id)
        return
    if record is None:
        # Missing, already processing, processed, failed, or being deleted: do not start another job.
        return

    try:
        with _download_cloudinary_asset(record) as stream:
            extracted_text = extract_document_text(record, stream)
        try:
            result = documents.update_one(
                {"_id": object_id, "status": DocumentStatus.PROCESSING.value},
                {
                    "$set": {
                        "status": DocumentStatus.PROCESSED.value,
                        "extracted_text": extracted_text,
                        "processed_at": datetime.now(timezone.utc),
                        "processing_error": None,
                        "chunking_status": ChunkingStatus.PENDING.value,
                        "chunking_error": None,
                        "embedding_status": EmbeddingStatus.PENDING.value,
                        "embedding_error": None,
                    }
                },
            )
        except Exception:
            _mark_processing_failed(documents, object_id, "Extracted text could not be saved to MongoDB.")
            return
        if result.matched_count == 0:
            logger.warning("Document state changed before processing could finish; document_id=%s", object_id)
            return
        chunk_document(str(object_id), documents, documents.database["document_chunks"])
        embed_document(str(object_id), documents, documents.database["document_chunks"])
    except DocumentProcessingError as exc:
        _mark_processing_failed(documents, object_id, str(exc))
    except Exception:
        logger.error("Unexpected document processing failure; document_id=%s", object_id)
        _mark_processing_failed(documents, object_id, "An unexpected error occurred during document processing.")
