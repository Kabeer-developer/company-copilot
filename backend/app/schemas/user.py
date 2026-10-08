from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator

from app.models.user import UserRole


class UserUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=100)
    email: EmailStr | None = None

    @field_validator("name")
    @classmethod
    def name_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            raise ValueError("Name cannot be null")
        value = value.strip()
        if not value:
            raise ValueError("Name must not be blank")
        return value

    @field_validator("email")
    @classmethod
    def email_must_not_be_null(cls, value: EmailStr | None) -> EmailStr | None:
        if value is None:
            raise ValueError("Email cannot be null")
        return value

    @model_validator(mode="after")
    def require_a_profile_field(self) -> "UserUpdateRequest":
        if not self.model_fields_set:
            raise ValueError("At least one of name or email must be provided")
        return self


class UserRoleUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: UserRole


class UserStatusUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_active: bool
