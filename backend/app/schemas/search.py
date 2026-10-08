from pydantic import BaseModel, ConfigDict, Field


class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)


class SearchResult(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    filename: str | None = None


class SearchResponse(BaseModel):
    results: list[SearchResult]
    count: int
