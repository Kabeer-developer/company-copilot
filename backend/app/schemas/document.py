from datetime import datetime

from pydantic import BaseModel, ConfigDict, HttpUrl

from app.models.document import ChunkingStatus, DocumentStatus, EmbeddingStatus


class DocumentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    filename: str
    file_type: str
    file_size: int
    cloudinary_url: HttpUrl
    uploaded_by: str
    uploaded_at: datetime
    status: DocumentStatus
    chunking_status: ChunkingStatus = ChunkingStatus.PENDING
    embedding_status: EmbeddingStatus = EmbeddingStatus.PENDING


class DocumentListResponse(BaseModel):
    items: list[DocumentResponse]
    page: int
    page_size: int
    total: int
    total_pages: int
