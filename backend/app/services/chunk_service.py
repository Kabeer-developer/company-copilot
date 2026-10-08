import logging
import re
from datetime import datetime, timezone

from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.collection import Collection

from app.models.document import ChunkingStatus, DocumentStatus, EmbeddingStatus


logger = logging.getLogger(__name__)

# Approximate 800 tokens at ~4 characters/token, with roughly 80 tokens of overlap.
TARGET_CHUNK_TOKENS = 800
CHUNK_OVERLAP_TOKENS = 80
CHARS_PER_TOKEN_ESTIMATE = 4
TARGET_CHUNK_CHARS = TARGET_CHUNK_TOKENS * CHARS_PER_TOKEN_ESTIMATE
CHUNK_OVERLAP_CHARS = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN_ESTIMATE

NO_TEXT_ERROR = "No extractable text available for chunking."
CHUNKING_ERROR = "Chunk creation failed."


class ChunkingError(Exception):
    pass


def _split_long_sentence(sentence: str, limit: int) -> list[str]:
    """Split only overlong sentences, preferring whitespace boundaries."""
    words = sentence.split()
    pieces: list[str] = []
    current = ""
    for word in words:
        if len(word) > limit:
            if current:
                pieces.append(current)
                current = ""
            for offset in range(0, len(word), limit):
                pieces.append(word[offset:offset + limit])
            continue
        candidate = f"{current} {word}" if current else word
        if len(candidate) > limit and current:
            pieces.append(current)
            current = word
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def _units(text: str, limit: int) -> list[tuple[str, bool]]:
    """Return (sentence, starts_new_paragraph), splitting long sentences safely."""
    result: list[tuple[str, bool]] = []
    for paragraph in re.split(r"\n\s*\n", text.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        first_in_paragraph = True
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            for part in _split_long_sentence(sentence, limit):
                result.append((part, first_in_paragraph))
                first_in_paragraph = False
    return result


def chunk_text(
    text: str,
    *,
    target_chars: int = TARGET_CHUNK_CHARS,
    overlap_chars: int = CHUNK_OVERLAP_CHARS,
) -> list[str]:
    """Pack paragraph/sentence units to a character budget and add suffix overlap."""
    normalized = text.strip()
    if not normalized:
        return []
    if target_chars < 1 or overlap_chars < 0 or overlap_chars >= target_chars:
        raise ValueError("Chunk target must be positive and overlap smaller than target")

    units = _units(normalized, target_chars)
    raw_chunks: list[str] = []
    current = ""
    for sentence, starts_paragraph in units:
        separator = "\n\n" if starts_paragraph and current else " "
        candidate = f"{current}{separator}{sentence}" if current else sentence
        if len(candidate) > target_chars and current:
            raw_chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        raw_chunks.append(current)

    chunks = [raw_chunks[0]]
    for next_chunk in raw_chunks[1:]:
        prior = chunks[-1]
        overlap_start = max(0, len(prior) - overlap_chars)
        while overlap_start < len(prior) and not prior[overlap_start - 1:overlap_start].isspace():
            overlap_start += 1
        overlap = prior[overlap_start:].strip()
        chunks.append(f"{overlap} {next_chunk}".strip() if overlap else next_chunk)
    return chunks


def _mark_failed(documents: Collection, document_id: ObjectId, message: str) -> None:
    try:
        documents.update_one(
            {"_id": document_id, "chunking_status": ChunkingStatus.PROCESSING.value},
            {"$set": {
                "chunking_status": ChunkingStatus.FAILED.value,
                "chunking_error": message[:500],
            }},
        )
    except Exception:
        logger.error("Could not store chunking failure for document_id=%s", document_id)


def chunk_document(document_id: str, documents: Collection, chunks: Collection) -> None:
    """Chunk one successfully extracted document. Safe against concurrent/repeated runs."""
    if not ObjectId.is_valid(document_id):
        logger.error("Chunking received an invalid document id")
        return
    object_id = ObjectId(document_id)
    try:
        chunks.create_index([("document_id", 1), ("chunk_index", 1)], unique=True)
        document = documents.find_one_and_update(
            {
                "_id": object_id,
                "status": DocumentStatus.PROCESSED.value,
                "chunking_status": {"$in": [ChunkingStatus.PENDING.value, ChunkingStatus.FAILED.value]},
            },
            {"$set": {
                "chunking_status": ChunkingStatus.PROCESSING.value,
                "chunking_error": None,
                "embedding_status": EmbeddingStatus.PENDING.value,
                "embedding_error": None,
            }},
            return_document=ReturnDocument.AFTER,
        )
    except Exception:
        logger.error("Could not claim document for chunking; document_id=%s", object_id)
        return
    if document is None:
        logger.info(
            "Chunking skipped because document is missing, not processed, or already claimed/completed; document_id=%s",
            object_id,
        )
        return

    extracted_text = document.get("extracted_text")
    if not isinstance(extracted_text, str) or not extracted_text.strip():
        _mark_failed(documents, object_id, NO_TEXT_ERROR)
        return
    uploaded_by = document.get("uploaded_by")
    if not isinstance(uploaded_by, ObjectId):
        _mark_failed(documents, object_id, "Document owner metadata is unavailable for chunk creation.")
        return

    try:
        generated = chunk_text(extracted_text)
        if not generated:
            _mark_failed(documents, object_id, NO_TEXT_ERROR)
            return

        # A retry is allowed only from FAILED; remove any leftovers from an earlier partial write.
        chunks.delete_many({"document_id": object_id})
        created_at = datetime.now(timezone.utc)
        records = [
            {
                "document_id": object_id,
                "uploaded_by": uploaded_by,
                "chunk_index": index,
                "text": chunk,
                "created_at": created_at,
            }
            for index, chunk in enumerate(generated)
        ]
        chunks.insert_many(records, ordered=True)
        result = documents.update_one(
            {"_id": object_id, "chunking_status": ChunkingStatus.PROCESSING.value},
            {"$set": {
                "chunking_status": ChunkingStatus.COMPLETED.value,
                "chunking_error": None,
                "chunk_count": len(records),
                "chunked_at": created_at,
                "embedding_status": EmbeddingStatus.PENDING.value,
                "embedding_error": None,
            }},
        )
        if result.matched_count == 0:
            chunks.delete_many({"document_id": object_id})
            logger.warning("Document state changed before chunking could finish; document_id=%s", object_id)
    except Exception:
        try:
            chunks.delete_many({"document_id": object_id})
        except Exception:
            logger.error("Could not clean up partial chunks for document_id=%s", object_id)
        _mark_failed(documents, object_id, CHUNKING_ERROR)
        logger.error("Chunk creation failed for document_id=%s", object_id)
