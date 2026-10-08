from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)

    @field_validator("query")
    @classmethod
    def query_must_contain_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Query must contain non-whitespace text.")
        return value


class ChatSource(BaseModel):
    chunk_id: str
    document_id: str
    filename: str | None = None
    score: float


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[ChatSource]
