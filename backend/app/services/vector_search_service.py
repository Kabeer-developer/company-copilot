import logging
import math

from bson import ObjectId
from pymongo.collection import Collection
from pymongo.errors import PyMongoError

from app.core.config import VECTOR_SEARCH_INDEX_NAME
from app.models.document import ChunkingStatus, DocumentStatus, EmbeddingStatus
from app.services.embedding_service import EMBEDDING_DIMENSION, EmbeddingServiceError, embed_query


logger = logging.getLogger(__name__)
MAX_TOP_K = 20
CANDIDATE_MULTIPLIER = 10
MIN_CANDIDATES = 50


class VectorSearchUnavailableError(Exception):
    pass


class QueryEmbeddingError(Exception):
    pass


def search_chunks(
    chunks: Collection,
    *,
    query: str,
    top_k: int,
    similarity_threshold: float,
    current_user: dict,
    is_admin: bool,
) -> list[dict]:
    try:
        vector = embed_query(query)
    except EmbeddingServiceError as exc:
        raise QueryEmbeddingError("The search query could not be embedded.") from exc
    if len(vector) != EMBEDDING_DIMENSION:
        raise QueryEmbeddingError("The query embedding has an unexpected dimension.")

    vector_stage = {
        "index": VECTOR_SEARCH_INDEX_NAME,
        "path": "embedding",
        "queryVector": vector,
        "numCandidates": max(MIN_CANDIDATES, top_k * CANDIDATE_MULTIPLIER),
        "limit": max(MIN_CANDIDATES, top_k * CANDIDATE_MULTIPLIER),
    }
    if not is_admin:
        owner_id = current_user.get("_id")
        if not isinstance(owner_id, ObjectId):
            raise VectorSearchUnavailableError("Authenticated user ownership metadata is invalid.")
        # Owner filtering happens inside Atlas Vector Search, before candidate chunks are returned.
        vector_stage["filter"] = {"uploaded_by": owner_id}

    pipeline = [
        {"$vectorSearch": vector_stage},
        {
            "$lookup": {
                "from": "documents",
                "localField": "document_id",
                "foreignField": "_id",
                "as": "parent_document",
            }
        },
        {"$unwind": "$parent_document"},
        {
            "$match": {
                "parent_document.status": DocumentStatus.PROCESSED.value,
                "parent_document.chunking_status": ChunkingStatus.COMPLETED.value,
                "parent_document.embedding_status": EmbeddingStatus.COMPLETED.value,
                **({"parent_document.uploaded_by": current_user.get("_id")} if not is_admin else {}),
            }
        },
        {
            "$project": {
                "document_id": 1,
                "chunk_index": 1,
                "text": 1,
                "filename": "$parent_document.filename",
                "score": {"$meta": "vectorSearchScore"},
            }
        },
        {"$match": {"score": {"$gte": similarity_threshold}}},
        {"$sort": {"score": -1}},
        {"$limit": top_k},
    ]
    try:
        hits = list(chunks.aggregate(pipeline, maxTimeMS=15000))
    except PyMongoError as exc:
        logger.error("MongoDB vector search failed for authenticated user")
        raise VectorSearchUnavailableError(
            "Vector search is unavailable. Confirm the Atlas index is active and MongoDB is reachable."
        ) from exc
    except Exception as exc:
        logger.error("Unexpected vector search failure")
        raise VectorSearchUnavailableError("Vector search is currently unavailable.") from exc

    results = []
    for hit in hits:
        score = hit.get("score")
        if not isinstance(score, (int, float)) or not math.isfinite(score):
            continue
        results.append(
            {
                "chunk_id": str(hit["_id"]),
                "document_id": str(hit["document_id"]),
                "text": hit["text"],
                "score": float(score),
                "filename": hit.get("filename") if isinstance(hit.get("filename"), str) else None,
            }
        )
    return results
