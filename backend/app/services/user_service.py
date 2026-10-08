from bson import ObjectId
from pymongo import ReturnDocument
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from app.models.user import UserRole
from app.schemas.user import UserRoleUpdateRequest, UserStatusUpdateRequest, UserUpdateRequest
from app.services.auth_service import get_user_by_id, public_user


class InvalidUserIdError(Exception):
    pass


class UserNotFoundError(Exception):
    pass


class DuplicateUserEmailError(Exception):
    pass


class LastAdminError(Exception):
    pass


class SelfDeletionError(Exception):
    pass


def list_users(users: Collection) -> list[dict]:
    return [public_user(user) for user in users.find().sort("created_at", 1)]


def require_user(users: Collection, user_id: str) -> tuple[ObjectId, dict]:
    if not ObjectId.is_valid(user_id):
        raise InvalidUserIdError
    object_id = ObjectId(user_id)
    user = get_user_by_id(users, user_id)
    if user is None:
        raise UserNotFoundError
    return object_id, user


def get_user(users: Collection, user_id: str) -> dict:
    _, user = require_user(users, user_id)
    return public_user(user)


def update_user(users: Collection, user_id: str, data: UserUpdateRequest) -> dict:
    object_id, _ = require_user(users, user_id)
    updates = data.model_dump(exclude_unset=True)
    if "email" in updates:
        updates["email"] = str(updates["email"]).strip().lower()
        users.create_index("email", unique=True)
    try:
        user = users.find_one_and_update(
            {"_id": object_id},
            {"$set": updates},
            return_document=ReturnDocument.AFTER,
        )
    except DuplicateKeyError as exc:
        raise DuplicateUserEmailError from exc
    if user is None:
        raise UserNotFoundError
    return public_user(user)


def _active_admin_count(users: Collection, *, session=None) -> int:
    return users.count_documents(
        {"role": UserRole.ADMIN.value, "is_active": True}, session=session
    )


def _admin_count(users: Collection, *, session=None) -> int:
    return users.count_documents({"role": UserRole.ADMIN.value}, session=session)


def _run_serialized_admin_mutation(users: Collection, operation):
    """Serialize guarded admin mutations so concurrent requests cannot remove all admins."""
    locks = users.database["user_management_locks"]
    locks.update_one(
        {"_id": "admin-mutations"},
        {"$setOnInsert": {"revision": 0}},
        upsert=True,
    )

    def transaction(session):
        locks.update_one(
            {"_id": "admin-mutations"},
            {"$inc": {"revision": 1}},
            session=session,
        )
        return operation(session)

    with users.database.client.start_session() as session:
        return session.with_transaction(transaction)


def change_user_role(users: Collection, user_id: str, data: UserRoleUpdateRequest) -> dict:
    if not ObjectId.is_valid(user_id):
        raise InvalidUserIdError
    object_id = ObjectId(user_id)
    new_role = data.role.value

    def update(session):
        existing = users.find_one({"_id": object_id}, session=session)
        if existing is None:
            raise UserNotFoundError
        if existing["role"] == UserRole.ADMIN.value and new_role != UserRole.ADMIN.value:
            if _admin_count(users, session=session) <= 1 or (
                existing.get("is_active") and _active_admin_count(users, session=session) <= 1
            ):
                raise LastAdminError
        user = users.find_one_and_update(
            {"_id": object_id},
            {"$set": {"role": new_role}},
            return_document=ReturnDocument.AFTER,
            session=session,
        )
        if user is None:
            raise UserNotFoundError
        return public_user(user)

    return _run_serialized_admin_mutation(users, update)


def change_user_status(users: Collection, user_id: str, data: UserStatusUpdateRequest) -> dict:
    if not ObjectId.is_valid(user_id):
        raise InvalidUserIdError
    object_id = ObjectId(user_id)

    def update(session):
        existing = users.find_one({"_id": object_id}, session=session)
        if existing is None:
            raise UserNotFoundError
        if (
            existing["role"] == UserRole.ADMIN.value
            and existing.get("is_active")
            and not data.is_active
            and _active_admin_count(users, session=session) <= 1
        ):
            raise LastAdminError
        user = users.find_one_and_update(
            {"_id": object_id},
            {"$set": {"is_active": data.is_active}},
            return_document=ReturnDocument.AFTER,
            session=session,
        )
        if user is None:
            raise UserNotFoundError
        return public_user(user)

    return _run_serialized_admin_mutation(users, update)


def delete_user(users: Collection, user_id: str, current_admin_id: str) -> None:
    if not ObjectId.is_valid(user_id):
        raise InvalidUserIdError
    object_id = ObjectId(user_id)
    if str(object_id) == current_admin_id:
        raise SelfDeletionError

    def delete(session):
        existing = users.find_one({"_id": object_id}, session=session)
        if existing is None:
            raise UserNotFoundError
        if existing["role"] == UserRole.ADMIN.value and (
            _admin_count(users, session=session) <= 1
            or (existing.get("is_active") and _active_admin_count(users, session=session) <= 1)
        ):
            raise LastAdminError
        result = users.delete_one({"_id": object_id}, session=session)
        if result.deleted_count == 0:
            raise UserNotFoundError

    _run_serialized_admin_mutation(users, delete)
