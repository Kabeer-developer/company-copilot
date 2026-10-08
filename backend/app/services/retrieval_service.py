from app.core.config import RETRIEVAL_CONTEXT_MAX_CHARS, RETRIEVAL_SIMILARITY_THRESHOLD
from app.services.vector_search_service import search_chunks


def _format_source(result: dict) -> str:
    filename = result.get("filename")
    label = filename if isinstance(filename, str) and filename else "Unknown document"
    return f"[Source]\nDocument: {label}\nContent:\n{result['text']}"


def retrieve_context(
    chunks,
    *,
    query: str,
    top_k: int,
    current_user: dict,
    is_admin: bool,
    max_context_chars: int = RETRIEVAL_CONTEXT_MAX_CHARS,
) -> dict:
    """Search with M9 and assemble exact chunk text within a character budget.

    An individually oversized chunk is skipped without truncation; lower-ranked
    chunks are still considered. Once a block would overflow the remaining
    budget, assembly stops to preserve relevance order.
    """
    results = search_chunks(
        chunks,
        query=query,
        top_k=top_k,
        similarity_threshold=RETRIEVAL_SIMILARITY_THRESHOLD,
        current_user=current_user,
        is_admin=is_admin,
    )

    context_parts: list[str] = []
    sources: list[dict] = []
    context_length = 0
    for result in results:
        if not isinstance(result.get("text"), str):
            continue
        block = _format_source(result)
        addition_length = len(block) + (2 if context_parts else 0)
        if addition_length > max_context_chars:
            # Do not truncate exact source text. Skip oversized individual chunks.
            continue
        if context_length + addition_length > max_context_chars:
            break
        context_parts.append(block)
        context_length += addition_length
        sources.append({
            "chunk_id": result["chunk_id"],
            "document_id": result["document_id"],
            "filename": result.get("filename"),
            "score": result["score"],
            "text": result["text"],
        })

    return {"query": query, "context": "\n\n".join(context_parts), "sources": sources}
