"""
User Models for MongoDB

This module defines the User model and related schemas
for authentication and user management.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from passlib.context import CryptContext
import hashlib

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class UserInDB(BaseModel):
    """User model as stored in MongoDB"""
    id: Optional[str] = Field(None, alias="_id")
    email: EmailStr
    hashed_password: str
    full_name: str
    google_id: Optional[str] = None
    avatar_url: Optional[str] = None
    role: str = "user"  # "user" or "admin"
    is_active: bool = True
    is_verified: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "email": "user@example.com",
                "full_name": "John Doe",
                "is_active": True,
            }
        }


class UserCreate(BaseModel):
    """Schema for user registration"""
    email: EmailStr
    password: str  # No length restrictions
    full_name: str = Field(..., min_length=2)
    role: str = Field(default="user")  # "user" or "admin", defaults to "user"


class UserLogin(BaseModel):
    """Schema for user login"""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Schema for user response (without password)"""
    id: str = Field(..., alias="_id")
    email: EmailStr
    full_name: str
    avatar_url: Optional[str] = None
    role: str  # "user" or "admin", no default - use value from database
    is_active: bool
    created_at: datetime

    class Config:
        populate_by_name = True


class TokenResponse(BaseModel):
    """Schema for JWT token response"""
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


class PasswordResetRequest(BaseModel):
    """Schema for password reset request"""
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    """Schema for password reset confirmation"""
    token: str
    new_password: str  # No length restrictions


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    TEMPORARY: Plain text password verification (INSECURE - for development only)
    """
    return plain_password == hashed_password


def get_password_hash(password: str) -> str:
    """
    TEMPORARY: No hashing - stores plain text (INSECURE - for development only)
    """
    return password



