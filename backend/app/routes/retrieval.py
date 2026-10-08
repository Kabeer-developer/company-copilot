from fastapi import APIRouter, Depends, HTTPException, status
from pymongo.database import Database

from app.models.user import UserRole
from app.routes.auth import get_current_user, get_database
from app.schemas.retrieval import RetrievalRequest, RetrievalResponse
from app.services.embedding_service import EmbeddingModelLoadError
from app.services.retrieval_service import retrieve_context
from app.services.vector_search_service import QueryEmbeddingError, VectorSearchUnavailableError


router = APIRouter()


@router.post("", response_model=RetrievalResponse)
def retrieve(
    request: RetrievalRequest,
    current_user: dict = Depends(get_current_user),
    database: Database = Depends(get_database),
) -> dict:
    try:
        return retrieve_context(
            database["document_chunks"],
            query=request.query,
            top_k=request.top_k,
            current_user=current_user,
            is_admin=current_user.get("role") == UserRole.ADMIN.value,
        )
    except QueryEmbeddingError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Query embedding is unavailable. Check the local embedding model.") from exc
    except EmbeddingModelLoadError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="The local embedding model could not be loaded.") from exc
    except VectorSearchUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
