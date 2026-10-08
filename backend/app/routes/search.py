from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from app.models.user import UserRole
from app.routes.auth import get_current_user, get_database
from app.schemas.search import SearchRequest, SearchResponse
from app.services.embedding_service import EmbeddingModelLoadError
from app.services.vector_search_service import (
    QueryEmbeddingError,
    VectorSearchUnavailableError,
    search_chunks,
)


router = APIRouter()


@router.post("", response_model=SearchResponse)
def search(
    request: SearchRequest,
    current_user: dict = Depends(get_current_user),
    database: Database = Depends(get_database),
) -> dict:
    try:
        results = search_chunks(
            database["document_chunks"],
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold,
            current_user=current_user,
            is_admin=current_user.get("role") == UserRole.ADMIN.value,
        )
    except QueryEmbeddingError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Query embedding is unavailable. Check the local embedding model.",
        ) from exc
    except EmbeddingModelLoadError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local embedding model could not be loaded.",
        ) from exc
    except VectorSearchUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    return {"results": results, "count": len(results)}
