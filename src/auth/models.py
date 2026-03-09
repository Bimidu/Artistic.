"""
User Models for MongoDB

This module defines the User model and related schemas
for authentication and user management.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
import bcrypt


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
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")
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
    role: str
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
    new_password: str = Field(..., min_length=8, description="Password must be at least 8 characters")


class PasswordReset(BaseModel):
    """Schema for direct password reset (without token)"""
    email: EmailStr
    new_password: str = Field(..., min_length=8, description="Password must be at least 8 characters")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain text password against a hashed password using bcrypt

    Args:
        plain_password: The plain text password to verify
        hashed_password: The bcrypt hashed password from database

    Returns:
        True if password matches, False otherwise

    Note:
        Bcrypt has a 72-byte password limit. Passwords longer than 72 bytes
        are automatically truncated to match the hashing behavior.
    """
    # Bcrypt has a 72-byte limit, truncate if necessary
    password_bytes = plain_password.encode('utf-8')[:72]
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)


def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt

    Args:
        password: Plain text password to hash

    Returns:
        Bcrypt hashed password string

    Note:
        Bcrypt has a 72-byte password limit. Passwords longer than 72 bytes
        are automatically truncated to prevent errors.
    """
    # Bcrypt has a 72-byte limit, truncate if necessary
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode('utf-8')
