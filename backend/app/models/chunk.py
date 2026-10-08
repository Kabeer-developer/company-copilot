from datetime import datetime
from typing import NotRequired, TypedDict

from bson import ObjectId


class DocumentChunk(TypedDict):
    _id: ObjectId
    document_id: ObjectId
    uploaded_by: ObjectId
    chunk_index: int
    text: str
    created_at: datetime
    embedding: NotRequired[list[float]]
    embedding_model: NotRequired[str]
    embedded_at: NotRequired[datetime]
