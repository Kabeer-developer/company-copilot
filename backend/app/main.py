import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from pymongo import MongoClient

from app.routes.auth import router as auth_router
from app.routes.documents import router as documents_router
from app.routes.users import router as users_router
from app.routes.search import router as search_router
from app.routes.retrieval import router as retrieval_router
from app.routes.chat import router as chat_router
from app.routes.conversations import router as conversations_router


load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    mongodb_uri = os.getenv("MONGODB_URI", "").strip()
    app.state.mongodb_client = MongoClient(mongodb_uri) if mongodb_uri else None
    try:
        yield
    finally:
        if app.state.mongodb_client is not None:
            app.state.mongodb_client.close()


app = FastAPI(title="Company Knowledge Copilot API", lifespan=lifespan)
app.include_router(auth_router, prefix="/api/auth", tags=["auth"])
app.include_router(users_router, prefix="/api/users", tags=["users"])
app.include_router(documents_router, prefix="/api/documents", tags=["documents"])
app.include_router(search_router, prefix="/api/search", tags=["search"])
app.include_router(retrieval_router, prefix="/api/retrieval", tags=["retrieval"])
app.include_router(chat_router, prefix="/api/chat", tags=["chat"])
app.include_router(conversations_router, prefix="/api/conversations", tags=["conversations"])


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
