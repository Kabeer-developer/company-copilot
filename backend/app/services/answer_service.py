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
    history: list[dict] | None = None,
) -> dict:
    previous_questions = [
        message["content"]
        for message in (history or [])
        if message.get("role") == "user"
        and isinstance(message.get("content"), str)
        and message["content"].strip()
    ][-3:]
    retrieval_query = query
    if previous_questions:
        retrieval_query = (
            "Previous user questions for topic resolution:\n"
            + "\n".join(previous_questions)
            + "\nCurrent follow-up question:\n"
            + query
        )
    retrieval = retrieve_context(
        chunks,
        query=retrieval_query,
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
    elif history:
        answer = generate_answer(query, retrieval["context"], history=history)
    else:
        answer = generate_answer(query, retrieval["context"])
    return {"query": query, "answer": answer, "sources": sources}
