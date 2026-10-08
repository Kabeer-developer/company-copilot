import logging
import math
import threading
from datetime import datetime, timezone
from typing import Any

from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.collection import Collection

from app.core.config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL_NAME
from app.models.document import ChunkingStatus, DocumentStatus, EmbeddingStatus


logger = logging.getLogger(__name__)
EMBEDDING_DIMENSION = 384
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
NO_CHUNKS_ERROR = "No document chunks are available for embedding."
EMPTY_CHUNK_ERROR = "A document chunk contains no text and cannot be embedded."
EMBEDDING_ERROR = "One or more document chunks could not be embedded."

_model: Any | None = None
_model_lock = threading.Lock()


class EmbeddingServiceError(Exception):
    pass


class EmbeddingModelLoadError(EmbeddingServiceError):
    pass


class EmbeddingDimensionError(EmbeddingServiceError):
    pass


def _get_model() -> Any:
    global _model
    if _model is not None:
        return _model
    with _model_lock:
        if _model is None:
            try:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
            except Exception as exc:
                raise EmbeddingModelLoadError(
                    "The local embedding model could not be loaded. Check the sentence-transformers installation and model download."
                ) from exc
    return _model


def _encode(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise EmbeddingServiceError("Embedding input must contain non-empty text.")
    try:
        encoded = _get_model().encode(
            texts,
            batch_size=EMBEDDING_BATCH_SIZE,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        rows = encoded.tolist() if hasattr(encoded, "tolist") else encoded
    except EmbeddingServiceError:
        raise
    except Exception as exc:
        raise EmbeddingServiceError("The local embedding model failed to encode text.") from exc

    if len(texts) == 1 and rows and isinstance(rows[0], (int, float)):
        rows = [rows]
    if not isinstance(rows, (list, tuple)) or len(rows) != len(texts):
        raise EmbeddingServiceError("The embedding model returned an unexpected number of vectors.")

    vectors: list[list[float]] = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) != EMBEDDING_DIMENSION:
            raise EmbeddingDimensionError(f"The embedding model must return {EMBEDDING_DIMENSION}-dimensional vectors.")
        vector: list[float] = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise EmbeddingServiceError("The embedding model returned a non-numeric vector value.")
            vector.append(float(value))
        vectors.append(vector)
    return vectors


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode passage text using normalized BGE vectors."""
    return _encode(texts)


def embed_text(text: str) -> list[float]:
    """Encode one document passage; exposed for reuse by later ingestion code."""
    return embed_texts([text])[0]


def embed_query(query: str) -> list[float]:
    """Encode a future short search query compatibly with the stored passage vectors."""
    if not isinstance(query, str) or not query.strip():
        raise EmbeddingServiceError("Query text must not be empty.")
    return _encode([QUERY_INSTRUCTION + query.strip()])[0]


def _is_valid_embedding(chunk: dict) -> bool:
    embedding = chunk.get("embedding")
    if chunk.get("embedding_model") != EMBEDDING_MODEL_NAME:
        return False
    if not isinstance(embedding, list) or len(embedding) != EMBEDDING_DIMENSION:
        return False
    return all(
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        for value in embedding
    )


def _mark_failed(documents: Collection, document_id: ObjectId, message: str) -> None:
    try:
        documents.update_one(
            {"_id": document_id, "embedding_status": EmbeddingStatus.PROCESSING.value},
            {"$set": {
                "embedding_status": EmbeddingStatus.FAILED.value,
                "embedding_error": message[:500],
            }},
        )
    except Exception:
        logger.error("Could not store embedding failure for document_id=%s", document_id)


def _generate_batch(texts: list[str]) -> tuple[list[list[float] | None], list[str | None]]:
    """Try one batch, then isolate individual failures if batch encoding fails."""
    try:
        vectors = embed_texts(texts)
        return list(vectors), [None] * len(texts)
    except EmbeddingModelLoadError:
        raise
    except Exception:
        vectors: list[list[float] | None] = [None] * len(texts)
        errors: list[str | None] = [None] * len(texts)
        for index, text in enumerate(texts):
            try:
                vectors[index] = embed_text(text)
            except EmbeddingModelLoadError:
                raise
            except Exception:
                errors[index] = "The local embedding model failed to encode this chunk."
        return vectors, errors


def embed_document(document_id: str, documents: Collection, chunks: Collection) -> None:
    """Embed chunks for one M6/M7-complete document and preserve successful partial results."""
    if not ObjectId.is_valid(document_id):
        logger.error("Embedding received an invalid document id")
        return
    object_id = ObjectId(document_id)
    try:
        document = documents.find_one_and_update(
            {
                "_id": object_id,
                "status": DocumentStatus.PROCESSED.value,
                "chunking_status": ChunkingStatus.COMPLETED.value,
                "embedding_status": {
                    "$in": [
                        EmbeddingStatus.PENDING.value,
                        EmbeddingStatus.FAILED.value,
                        EmbeddingStatus.COMPLETED.value,
                    ]
                },
            },
            {"$set": {"embedding_status": EmbeddingStatus.PROCESSING.value, "embedding_error": None}},
            return_document=ReturnDocument.AFTER,
        )
    except Exception:
        logger.error("Could not claim document for embedding; document_id=%s", object_id)
        return
    if document is None:
        logger.info("Embedding skipped because document is missing, incomplete, or already claimed/completed; document_id=%s", object_id)
        return

    try:
        document_chunks = list(
            chunks.find(
                {"document_id": object_id},
                {"_id": 1, "document_id": 1, "chunk_index": 1, "text": 1, "embedding": 1, "embedding_model": 1},
            ).sort("chunk_index", 1)
        )
        if not document_chunks:
            _mark_failed(documents, object_id, NO_CHUNKS_ERROR)
            return

        pending: list[dict] = []
        failures: list[str] = []
        for chunk in document_chunks:
            if _is_valid_embedding(chunk):
                continue
            text = chunk.get("text")
            if not isinstance(text, str) or not text.strip():
                failures.append(EMPTY_CHUNK_ERROR)
                continue
            pending.append(chunk)

        for offset in range(0, len(pending), EMBEDDING_BATCH_SIZE):
            batch = pending[offset:offset + EMBEDDING_BATCH_SIZE]
            try:
                vectors, errors = _generate_batch([chunk["text"] for chunk in batch])
            except EmbeddingModelLoadError:
                raise
            except Exception:
                failures.extend([EMBEDDING_ERROR] * len(batch))
                logger.error("Embedding batch failed for document_id=%s", object_id)
                continue

            for chunk, vector, error in zip(batch, vectors, errors):
                if error or vector is None:
                    failures.append(error or EMBEDDING_ERROR)
                    logger.error("Embedding failed for document_id=%s chunk_id=%s", object_id, chunk.get("_id"))
                    continue
                try:
                    result = chunks.update_one(
                        {"_id": chunk["_id"], "document_id": object_id},
                        {"$set": {
                            "embedding": vector,
                            "embedding_model": EMBEDDING_MODEL_NAME,
                            "embedded_at": datetime.now(timezone.utc),
                        }},
                    )
                    if result.matched_count == 0:
                        failures.append(EMBEDDING_ERROR)
                except Exception:
                    failures.append(EMBEDDING_ERROR)
                    logger.error("Could not store embedding for document_id=%s chunk_id=%s", object_id, chunk.get("_id"))

        if failures:
            message = failures[0] if failures[0] == EMPTY_CHUNK_ERROR else EMBEDDING_ERROR
            _mark_failed(documents, object_id, message)
            return

        result = documents.update_one(
            {"_id": object_id, "embedding_status": EmbeddingStatus.PROCESSING.value},
            {"$set": {
                "embedding_status": EmbeddingStatus.COMPLETED.value,
                "embedding_error": None,
                "embedded_at": datetime.now(timezone.utc),
            }},
        )
        if result.matched_count == 0:
            logger.warning("Document state changed before embedding could finish; document_id=%s", object_id)
    except EmbeddingModelLoadError:
        _mark_failed(
            documents,
            object_id,
            "The local embedding model could not be loaded. Check the installation and model download.",
        )
        logger.error("Embedding model load failed for document_id=%s", object_id)
    except Exception:
        _mark_failed(documents, object_id, EMBEDDING_ERROR)
        logger.error("Unexpected embedding failure for document_id=%s", object_id)
