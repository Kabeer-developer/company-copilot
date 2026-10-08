from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.core.config import RETRIEVAL_TOP_K


class RetrievalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=RETRIEVAL_TOP_K, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query must contain non-whitespace text.")
        return value


class RetrievalSource(BaseModel):
    chunk_id: str
    document_id: str
    filename: str | None = None
    score: float
    text: str


class RetrievalResponse(BaseModel):
    query: str
    context: str
    sources: list[RetrievalSource]
