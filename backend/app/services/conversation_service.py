import math
from datetime import datetime, timezone

from bson import ObjectId
from pymongo.collection import Collection
from pymongo.database import Database
from pymongo.errors import PyMongoError


DEFAULT_CONVERSATION_TITLE = "New conversation"
MAX_TITLE_LENGTH = 80


class InvalidConversationIdError(Exception):
    pass


class ConversationNotFoundError(Exception):
    pass


class ConversationPersistenceError(Exception):
    pass


def ensure_conversation_indexes(database: Database) -> None:
    try:
        database["conversations"].create_index(
            [("user_id", 1), ("updated_at", -1)],
            name="conversation_owner_updated_at",
        )
        database["messages"].create_index(
            [("conversation_id", 1), ("user_id", 1), ("created_at", 1), ("_id", 1)],
            name="message_conversation_owner_created_at",
        )
    except PyMongoError as exc:
        raise ConversationPersistenceError from exc


def _owner_id(current_user: dict) -> ObjectId:
    try:
        return ObjectId(str(current_user["_id"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ConversationPersistenceError from exc


def _conversation_id(value: str) -> ObjectId:
    if not ObjectId.is_valid(value):
        raise InvalidConversationIdError
    return ObjectId(value)


def _public_conversation(record: dict) -> dict:
    return {
        "id": str(record["_id"]),
        "title": record["title"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
    }


def _public_message(record: dict) -> dict:
    result = {
        "id": str(record["_id"]),
        "conversation_id": str(record["conversation_id"]),
        "role": record["role"],
        "content": record["content"],
        "created_at": record["created_at"],
    }
    if record["role"] == "assistant":
        result["sources"] = record.get("sources", [])
    return result


def create_conversation(conversations: Collection, current_user: dict) -> dict:
    now = datetime.now(timezone.utc)
    record = {
        "user_id": _owner_id(current_user),
        "title": DEFAULT_CONVERSATION_TITLE,
        "created_at": now,
        "updated_at": now,
    }
    try:
        result = conversations.insert_one(record)
    except PyMongoError as exc:
        raise ConversationPersistenceError from exc
    record["_id"] = result.inserted_id
    return _public_conversation(record)


def list_conversations(
    conversations: Collection,
    current_user: dict,
    *,
    page: int,
    page_size: int,
) -> dict:
    query = {"user_id": _owner_id(current_user)}
    try:
        total = conversations.count_documents(query)
        cursor = (
            conversations.find(query)
            .sort([("updated_at", -1), ("_id", -1)])
            .skip((page - 1) * page_size)
            .limit(page_size)
        )
        items = [_public_conversation(record) for record in cursor]
    except PyMongoError as exc:
        raise ConversationPersistenceError from exc
    return {
        "items": items,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": math.ceil(total / page_size),
    }


def get_conversation(
    conversations: Collection,
    conversation_id: str,
    current_user: dict,
) -> dict:
    object_id = _conversation_id(conversation_id)
    query = {"_id": object_id, "user_id": _owner_id(current_user)}
    try:
        record = conversations.find_one(query)
    except PyMongoError as exc:
        raise ConversationPersistenceError from exc
    if record is None:
        raise ConversationNotFoundError
    return record


def get_conversation_detail(
    conversations: Collection,
    messages: Collection,
    conversation_id: str,
    current_user: dict,
) -> dict:
    record = get_conversation(conversations, conversation_id, current_user)
    query = {
        "conversation_id": record["_id"],
        "user_id": _owner_id(current_user),
    }
    try:
        cursor = messages.find(query).sort([("created_at", 1), ("_id", 1)])
        history = [_public_message(message) for message in cursor]
    except PyMongoError as exc:
        raise ConversationPersistenceError from exc
    return {**_public_conversation(record), "messages": history}


def add_user_message(
    messages: Collection,
    *,
    conversation_id: ObjectId,
    current_user: dict,
    content: str,
) -> dict:
    record = {
        "conversation_id": conversation_id,
        "user_id": _owner_id(current_user),
        "role": "user",
        "content": content,
        "created_at": datetime.now(timezone.utc),
    }
    try:
        result = messages.insert_one(record)
    except PyMongoError as exc:
        raise ConversationPersistenceError from exc
    record["_id"] = result.inserted_id
    return _public_message(record)


def load_recent_messages(
    messages: Collection,
    *,
    conversation_id: ObjectId,
    current_user: dict,
    limit: int,
    exclude_message_id: str,
) -> list[dict]:
    if limit == 0:
        return []
    query = {
        "conversation_id": conversation_id,
        "user_id": _owner_id(current_user),
        "_id": {"$ne": ObjectId(exclude_message_id)},
    }
    try:
        records = list(messages.find(query).sort([("created_at", -1), ("_id", -1)]).limit(limit))
    except PyMongoError as exc:
        raise ConversationPersistenceError from exc
    records.reverse()
    return [{"role": record["role"], "content": record["content"]} for record in records]


def update_conversation_activity(
    conversations: Collection,
    *,
    conversation_id: ObjectId,
    current_user: dict,
    first_question: str,
) -> None:
    query = {"_id": conversation_id, "user_id": _owner_id(current_user)}
    now = datetime.now(timezone.utc)
    title = first_question.strip()
    if len(title) > MAX_TITLE_LENGTH:
        title = title[: MAX_TITLE_LENGTH - 3].rstrip() + "..."
    try:
        title_result = conversations.update_one(
            {**query, "title": DEFAULT_CONVERSATION_TITLE},
            {"$set": {"title": title or DEFAULT_CONVERSATION_TITLE, "updated_at": now}},
        )
        if title_result.matched_count == 0:
            touch_result = conversations.update_one(query, {"$set": {"updated_at": now}})
            if touch_result.matched_count == 0:
                raise ConversationNotFoundError
    except ConversationNotFoundError:
        raise
    except PyMongoError as exc:
        raise ConversationPersistenceError from exc


def add_assistant_message(
    messages: Collection,
    *,
    conversation_id: ObjectId,
    current_user: dict,
    content: str,
    sources: list[dict],
) -> dict:
    record = {
        "conversation_id": conversation_id,
        "user_id": _owner_id(current_user),
        "role": "assistant",
        "content": content,
        "sources": sources,
        "created_at": datetime.now(timezone.utc),
    }
    try:
        result = messages.insert_one(record)
    except PyMongoError as exc:
        raise ConversationPersistenceError from exc
    record["_id"] = result.inserted_id
    return _public_message(record)


def delete_conversation(
    conversations: Collection,
    messages: Collection,
    conversation_id: str,
    current_user: dict,
) -> None:
    object_id = _conversation_id(conversation_id)
    owner_id = _owner_id(current_user)
    query = {"_id": object_id, "user_id": owner_id}
    try:
        if conversations.find_one(query) is None:
            raise ConversationNotFoundError
        # Delete messages first so a message-delete error leaves the conversation
        # visible. MongoDB does not make these two collection writes atomic here.
        messages.delete_many({"conversation_id": object_id})
        result = conversations.delete_one(query)
    except ConversationNotFoundError:
        raise
    except PyMongoError as exc:
        raise ConversationPersistenceError from exc
    if result.deleted_count == 0:
        raise ConversationNotFoundError
