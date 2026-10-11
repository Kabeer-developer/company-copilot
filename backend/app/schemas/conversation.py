from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.schemas.chat import ChatSource


class ConversationSummary(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str
    sources: list[ChatSource] | None = None
    created_at: datetime


class ConversationDetail(ConversationSummary):
    messages: list[ConversationMessage]


class ConversationListResponse(BaseModel):
    items: list[ConversationSummary]
    page: int
    page_size: int
    total: int
    total_pages: int


class ConversationExchangeResponse(BaseModel):
    conversation_id: str
    user_message: ConversationMessage
    assistant_message: ConversationMessage
