"""
User Models for MongoDB

This module defines the User model and related schemas
for authentication and user management.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
import bcrypt
import hashlib


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


class PasswordReset(BaseModel):
    """Schema for direct password reset (without token)"""
    email: EmailStr
    new_password: str  # No length restrictions


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify plain password against bcrypt hash.

    Handles backward compatibility:
    - Tries bcrypt verification first (new users)
    - Falls back to plain-text comparison (old users during migration)
    """
    try:
        # Convert password to bytes if needed
        password_bytes = plain_password.encode('utf-8')
        hash_bytes = hashed_password.encode('utf-8') if isinstance(hashed_password, str) else hashed_password
        return bcrypt.checkpw(password_bytes, hash_bytes)
    except Exception:
        # Fallback for plain-text passwords (migration period)
        return plain_password == hashed_password


def get_password_hash(password: str) -> str:
    """
    Hash password using bcrypt.

    Uses bcrypt default cost factor (12 rounds) - industry standard.
    """
    password_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt(rounds=12)  # 12 rounds = industry standard
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')  # Return as string for MongoDB storage


def is_password_hashed(hashed_password: str) -> bool:
    """
    Detect if password is bcrypt-hashed or plain text.

    Bcrypt hashes start with "$2a$", "$2b$", or "$2y$".
    """
    if not hashed_password:
        return False
    return hashed_password.startswith(('$2a$', '$2b$', '$2y$'))



