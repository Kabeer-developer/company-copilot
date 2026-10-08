import os
import sys
import time

from bson import ObjectId
from pymongo import MongoClient
from pymongo.operations import SearchIndexModel

from app.core.config import MONGODB_DATABASE, VECTOR_SEARCH_INDEX_NAME


INDEX_DEFINITION = {
    "fields": [
        {
            "type": "vector",
            "path": "embedding",
            "numDimensions": 384,
            "similarity": "cosine",
        },
        {"type": "filter", "path": "uploaded_by"},
    ]
}
INDEX_WAIT_SECONDS = 600


def backfill_chunk_owners(documents, chunks) -> tuple[int, int]:
    updated = 0
    missing_parent = 0
    cursor = chunks.find(
        {"uploaded_by": {"$exists": False}},
        {"_id": 1, "document_id": 1},
    )
    for chunk in cursor:
        parent = documents.find_one(
            {"_id": chunk.get("document_id")},
            {"uploaded_by": 1},
        )
        owner_id = parent.get("uploaded_by") if parent else None
        if not isinstance(owner_id, ObjectId):
            missing_parent += 1
            continue
        result = chunks.update_one(
            {"_id": chunk["_id"], "uploaded_by": {"$exists": False}},
            {"$set": {"uploaded_by": owner_id}},
        )
        updated += result.modified_count
    return updated, missing_parent


def _definition_matches(index: dict) -> bool:
    definition = index.get("latestDefinition") or index.get("definition") or {}
    fields = definition.get("fields", [])
    vector_matches = any(
        field.get("type") == "vector"
        and field.get("path") == "embedding"
        and field.get("numDimensions") == 384
        and field.get("similarity") == "cosine"
        for field in fields
    )
    owner_filter_matches = any(
        field.get("type") == "filter" and field.get("path") == "uploaded_by"
        for field in fields
    )
    return vector_matches and owner_filter_matches


def ensure_vector_index(chunks) -> None:
    indexes = list(chunks.list_search_indexes())
    existing = next((index for index in indexes if index.get("name") == VECTOR_SEARCH_INDEX_NAME), None)
    if existing is not None:
        if not _definition_matches(existing):
            raise RuntimeError(
                f"Atlas index '{VECTOR_SEARCH_INDEX_NAME}' exists with a different definition; inspect or replace it manually."
            )
    else:
        chunks.create_search_index(
            model=SearchIndexModel(
                definition=INDEX_DEFINITION,
                name=VECTOR_SEARCH_INDEX_NAME,
                type="vectorSearch",
            )
        )

    deadline = time.monotonic() + INDEX_WAIT_SECONDS
    while time.monotonic() < deadline:
        state = next(
            (index for index in chunks.list_search_indexes() if index.get("name") == VECTOR_SEARCH_INDEX_NAME),
            None,
        )
        if state and state.get("status") == "READY" and state.get("queryable"):
            return
        if state and state.get("status") == "FAILED":
            raise RuntimeError(f"Atlas index '{VECTOR_SEARCH_INDEX_NAME}' failed to build; inspect the Atlas index error.")
        time.sleep(5)
    raise TimeoutError(f"Atlas index '{VECTOR_SEARCH_INDEX_NAME}' did not become ready within 10 minutes.")


def main() -> None:
    uri = os.getenv("MONGODB_URI", "").strip()
    if not uri:
        raise RuntimeError("MONGODB_URI is not configured.")
    client = MongoClient(uri, serverSelectionTimeoutMS=10000)
    try:
        client.admin.command("ping")
        database = client[MONGODB_DATABASE]
        chunks = database["document_chunks"]
        updated, missing_parent = backfill_chunk_owners(database["documents"], chunks)
        print(f"Backfilled uploaded_by on {updated} chunks; chunks without valid parent ownership: {missing_parent}.")
        ensure_vector_index(chunks)
        print(f"Atlas Vector Search index '{VECTOR_SEARCH_INDEX_NAME}' is ready.")
    finally:
        client.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Vector Search setup failed: {exc}", file=sys.stderr)
        raise
