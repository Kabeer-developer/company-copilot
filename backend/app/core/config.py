import os

from dotenv import load_dotenv


load_dotenv()


def _integer_setting(name: str, default: str, *, minimum: int, maximum: int | None = None) -> int:
    try:
        value = int(os.getenv(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"{minimum} or greater" if maximum is None else f"between {minimum} and {maximum}"
        raise ValueError(f"{name} must be {bounds}")
    return value


def _float_setting(name: str, default: str, *, minimum: float, maximum: float | None = None) -> float:
    try:
        value = float(os.getenv(name, default))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise ValueError(f"{name} must be finite")
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"{minimum} or greater" if maximum is None else f"between {minimum} and {maximum}"
        raise ValueError(f"{name} must be {bounds}")
    return value


MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "company_knowledge_copilot")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = _integer_setting("ACCESS_TOKEN_EXPIRE_MINUTES", "30", minimum=1)
CLOUDINARY_CLOUD_NAME = os.getenv("CLOUDINARY_CLOUD_NAME", "")
CLOUDINARY_API_KEY = os.getenv("CLOUDINARY_API_KEY", "")
CLOUDINARY_API_SECRET = os.getenv("CLOUDINARY_API_SECRET", "")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDING_BATCH_SIZE = _integer_setting("EMBEDDING_BATCH_SIZE", "32", minimum=1)
VECTOR_SEARCH_INDEX_NAME = os.getenv("VECTOR_SEARCH_INDEX_NAME", "chunk_vector_index")
RETRIEVAL_TOP_K = _integer_setting("RETRIEVAL_TOP_K", "5", minimum=1, maximum=20)
RETRIEVAL_CONTEXT_MAX_CHARS = _integer_setting("RETRIEVAL_CONTEXT_MAX_CHARS", "12000", minimum=1)
RETRIEVAL_SIMILARITY_THRESHOLD = _float_setting(
    "RETRIEVAL_SIMILARITY_THRESHOLD", "0.0", minimum=0.0, maximum=1.0
)
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b").strip()
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").strip()
LLM_TIMEOUT_SECONDS = _float_setting("LLM_TIMEOUT_SECONDS", "120", minimum=0.001)
CONVERSATION_HISTORY_LIMIT = _integer_setting(
    "CONVERSATION_HISTORY_LIMIT", "10", minimum=0, maximum=50
)
