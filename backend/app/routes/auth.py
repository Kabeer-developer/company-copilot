from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
import jwt
from pymongo.database import Database

from app.core.config import JWT_ALGORITHM, JWT_SECRET_KEY, MONGODB_DATABASE
from app.core.security import create_access_token
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse, UserResponse
from app.services.auth_service import (
    DuplicateEmailError,
    InactiveUserError,
    InvalidCredentialsError,
    authenticate_user,
    get_user_by_id,
    public_user,
    register_user,
)


router = APIRouter()
bearer_scheme = HTTPBearer(auto_error=False)


def get_database(request: Request) -> Database:
    client = request.app.state.mongodb_client
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MongoDB is not configured",
        )
    return client[MONGODB_DATABASE]


def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired authentication token",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if credentials is None or not JWT_SECRET_KEY:
        raise unauthorized

    try:
        payload = jwt.decode(
            credentials.credentials,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            options={"require": ["exp", "sub"]},
        )
        user_id = payload.get("sub")
        if not user_id:
            raise unauthorized
    except jwt.PyJWTError as exc:
        raise unauthorized from exc

    database = get_database(request)
    user = get_user_by_id(database["users"], user_id)
    if user is None:
        raise unauthorized
    if not user.get("is_active", False):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is inactive")
    return user


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
def register(data: RegisterRequest, database: Database = Depends(get_database)) -> dict:
    try:
        return register_user(database["users"], data)
    except DuplicateEmailError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Email is already registered") from exc


@router.post("/login", response_model=TokenResponse)
def login(data: LoginRequest, database: Database = Depends(get_database)) -> dict:
    try:
        user = authenticate_user(database["users"], data)
        token = create_access_token(str(user["_id"]), user["role"])
    except InvalidCredentialsError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password") from exc
    except InactiveUserError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="User account is inactive") from exc
    return {"access_token": token, "user": public_user(user)}


@router.get("/me", response_model=UserResponse)
def read_current_user(user: dict = Depends(get_current_user)) -> dict:
    return public_user(user)
