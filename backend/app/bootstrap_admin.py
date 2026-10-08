import os
from datetime import datetime, timezone
from getpass import getpass

from dotenv import load_dotenv
from pydantic import EmailStr, TypeAdapter, ValidationError
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError, PyMongoError

from app.core.config import MONGODB_DATABASE
from app.core.security import hash_password
from app.models.user import UserRole


def main() -> None:
    load_dotenv()
    mongodb_uri = os.getenv("MONGODB_URI", "").strip()
    if not mongodb_uri:
        raise SystemExit("MONGODB_URI is not configured in the environment or .env")

    try:
        name = input("Initial administrator name: ").strip()
        if not name or len(name) > 100:
            raise SystemExit("Name must contain between 1 and 100 characters")
        email = str(TypeAdapter(EmailStr).validate_python(input("Administrator email: "))).lower()
        password = getpass("Administrator password (8–128 characters): ")
        if not 8 <= len(password) <= 128:
            raise SystemExit("Password must contain between 8 and 128 characters")
        if password != getpass("Confirm password: "):
            raise SystemExit("Passwords do not match")

        with MongoClient(mongodb_uri, serverSelectionTimeoutMS=5000) as client:
            users = client[MONGODB_DATABASE]["users"]
            users.create_index("email", unique=True)
            if users.count_documents({"role": UserRole.ADMIN.value}) > 0:
                raise SystemExit("An administrator already exists; bootstrap is only for the first administrator")
            if users.find_one({"email": email}):
                raise SystemExit("That email is already registered")
            users.insert_one(
                {
                    "name": name,
                    "email": email,
                    "password_hash": hash_password(password),
                    "role": UserRole.ADMIN.value,
                    "is_active": True,
                    "created_at": datetime.now(timezone.utc),
                }
            )
        print("Initial administrator created successfully.")
    except ValidationError:
        raise SystemExit("Enter a valid email address") from None
    except DuplicateKeyError:
        raise SystemExit("That email is already registered") from None
    except PyMongoError:
        raise SystemExit("MongoDB operation failed; check the Atlas connection and database access") from None


if __name__ == "__main__":
    main()
