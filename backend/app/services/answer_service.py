from app.core.config import RETRIEVAL_TOP_K
from app.services.llm_service import generate_answer
from app.services.retrieval_service import retrieve_context


NOT_FOUND_ANSWER = "The information could not be found in the available company documents."


def answer_question(
    chunks,
    *,
    query: str,
    current_user: dict,
    is_admin: bool,
) -> dict:
    retrieval = retrieve_context(
        chunks,
        query=query,
        top_k=RETRIEVAL_TOP_K,
        current_user=current_user,
        is_admin=is_admin,
    )
    sources = [
        {
            "chunk_id": source["chunk_id"],
            "document_id": source["document_id"],
            "filename": source.get("filename"),
            "score": source["score"],
        }
        for source in retrieval["sources"]
    ]
    if not retrieval["context"] or not sources:
        answer = NOT_FOUND_ANSWER
    else:
        answer = generate_answer(query, retrieval["context"])
    return {"query": query, "answer": answer, "sources": sources}
