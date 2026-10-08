from fastapi import APIRouter, Depends, HTTPException, Response, status
from pymongo.collection import Collection
from pymongo.database import Database

from app.models.user import UserRole
from app.routes.auth import get_current_user, get_database
from app.schemas.auth import UserResponse
from app.schemas.user import UserRoleUpdateRequest, UserStatusUpdateRequest, UserUpdateRequest
from app.services.user_service import (
    DuplicateUserEmailError,
    InvalidUserIdError,
    LastAdminError,
    SelfDeletionError,
    UserNotFoundError,
    change_user_role,
    change_user_status,
    delete_user,
    get_user,
    list_users,
    update_user,
)


router = APIRouter()


def require_admin(user: dict = Depends(get_current_user)) -> dict:
    if user.get("role") != UserRole.ADMIN.value:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Administrator access required")
    return user


def users_collection(database: Database = Depends(get_database)) -> Collection:
    return database["users"]


def raise_user_error(error: Exception) -> None:
    if isinstance(error, InvalidUserIdError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid user ID") from error
    if isinstance(error, UserNotFoundError):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found") from error
    if isinstance(error, DuplicateUserEmailError):
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email is already in use") from error
    if isinstance(error, LastAdminError):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Operation would leave the system without an administrator",
        ) from error
    if isinstance(error, SelfDeletionError):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="You cannot delete your own account") from error
    raise error


@router.get("", response_model=list[UserResponse])
def read_users(
    _admin: dict = Depends(require_admin),
    users: Collection = Depends(users_collection),
) -> list[dict]:
    return list_users(users)


@router.get("/{user_id}", response_model=UserResponse)
def read_user(
    user_id: str,
    _admin: dict = Depends(require_admin),
    users: Collection = Depends(users_collection),
) -> dict:
    try:
        return get_user(users, user_id)
    except (InvalidUserIdError, UserNotFoundError) as exc:
        raise_user_error(exc)


@router.put("/{user_id}", response_model=UserResponse)
def edit_user(
    user_id: str,
    data: UserUpdateRequest,
    _admin: dict = Depends(require_admin),
    users: Collection = Depends(users_collection),
) -> dict:
    try:
        return update_user(users, user_id, data)
    except (InvalidUserIdError, UserNotFoundError, DuplicateUserEmailError) as exc:
        raise_user_error(exc)


@router.patch("/{user_id}/role", response_model=UserResponse)
def edit_user_role(
    user_id: str,
    data: UserRoleUpdateRequest,
    _admin: dict = Depends(require_admin),
    users: Collection = Depends(users_collection),
) -> dict:
    try:
        return change_user_role(users, user_id, data)
    except (InvalidUserIdError, UserNotFoundError, LastAdminError) as exc:
        raise_user_error(exc)


@router.patch("/{user_id}/status", response_model=UserResponse)
def edit_user_status(
    user_id: str,
    data: UserStatusUpdateRequest,
    _admin: dict = Depends(require_admin),
    users: Collection = Depends(users_collection),
) -> dict:
    try:
        return change_user_status(users, user_id, data)
    except (InvalidUserIdError, UserNotFoundError, LastAdminError) as exc:
        raise_user_error(exc)


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_user(
    user_id: str,
    admin: dict = Depends(require_admin),
    users: Collection = Depends(users_collection),
) -> Response:
    try:
        delete_user(users, user_id, str(admin["_id"]))
    except (InvalidUserIdError, UserNotFoundError, LastAdminError, SelfDeletionError) as exc:
        raise_user_error(exc)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
