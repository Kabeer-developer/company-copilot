from datetime import datetime, timezone

from bson import ObjectId
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError

from app.core.security import hash_password, verify_password
from app.models.user import UserRole
from app.schemas.auth import LoginRequest, RegisterRequest


class DuplicateEmailError(Exception):
    pass


class InvalidCredentialsError(Exception):
    pass


class InactiveUserError(Exception):
    pass


def public_user(user: dict) -> dict:
    return {
        "id": str(user["_id"]),
        "name": user["name"],
        "email": user["email"],
        "role": user["role"],
        "is_active": user["is_active"],
        "created_at": user["created_at"],
    }


def register_user(users: Collection, data: RegisterRequest) -> dict:
    users.create_index("email", unique=True)
    normalized_email = str(data.email).strip().lower()
    if users.find_one({"email": normalized_email}):
        raise DuplicateEmailError

    user = {
        "name": data.name,
        "email": normalized_email,
        "password_hash": hash_password(data.password),
        "role": UserRole.EMPLOYEE.value,
        "is_active": True,
        "created_at": datetime.now(timezone.utc),
    }
    try:
        result = users.insert_one(user)
    except DuplicateKeyError as exc:
        raise DuplicateEmailError from exc
    user["_id"] = result.inserted_id
    return public_user(user)


def authenticate_user(users: Collection, data: LoginRequest) -> dict:
    normalized_email = str(data.email).strip().lower()
    user = users.find_one({"email": normalized_email})
    if user is None or not verify_password(data.password, user["password_hash"]):
        raise InvalidCredentialsError
    if not user.get("is_active", False):
        raise InactiveUserError
    return user


def get_user_by_id(users: Collection, user_id: str) -> dict | None:
    if not ObjectId.is_valid(user_id):
        return None
    return users.find_one({"_id": ObjectId(user_id)})
