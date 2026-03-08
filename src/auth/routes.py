"""
Authentication Routes

FastAPI routes for user registration, login, and authentication.
"""

from fastapi import APIRouter, HTTPException, status, Depends
from src.auth.models import (
    UserCreate, UserLogin, TokenResponse, UserResponse,
    PasswordResetRequest, PasswordResetConfirm, PasswordReset,
    get_password_hash, verify_password
)
from src.auth.jwt import create_access_token
from src.auth.dependencies import get_current_active_user
from src.database import get_database
from datetime import datetime
import uuid

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate):
    """
    Register a new user

    - **email**: Valid email address
    - **password**: Minimum 8 characters (will be securely hashed with bcrypt)
    - **full_name**: User's full name
    - **role**: User role (defaults to 'user')
    """
    db = get_database()

    # Check if user already exists
    existing_user = await db.users.find_one({"email": user_data.email})
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )

    # Create user document with SECURELY HASHED password
    user_id = str(uuid.uuid4())
    user_doc = {
        "_id": user_id,
        "email": user_data.email,
        "hashed_password": get_password_hash(user_data.password),  # Bcrypt hash
        "full_name": user_data.full_name,
        "role": user_data.role,
        "is_active": True,
        "is_verified": False,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }

    # Insert into database
    await db.users.insert_one(user_doc)

    # Create access token
    access_token = create_access_token(data={"sub": user_id, "email": user_data.email, "role": user_data.role})

    # Return token and user info
    user_response = UserResponse(
        _id=user_id,
        email=user_data.email,
        full_name=user_data.full_name,
        role=user_data.role,
        is_active=True,
        created_at=user_doc["created_at"]
    )

    return TokenResponse(access_token=access_token, user=user_response)


@router.post("/login", response_model=TokenResponse)
async def login(credentials: UserLogin):
    """
    Login with email and password

    - **email**: User's email address
    - **password**: User's password (verified against bcrypt hash)
    """
    db = get_database()

    # Find user by email
    user = await db.users.find_one({"email": credentials.email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    # Verify password using bcrypt
    if not verify_password(credentials.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
        )

    # Check if user is active
    if not user.get("is_active", True):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user account"
        )

    # Create access token
    access_token = create_access_token(
        data={"sub": user["_id"], "email": user["email"], "role": user.get("role", "user")}
    )

    # Return token and user info
    user_response = UserResponse(**user)

    return TokenResponse(access_token=access_token, user=user_response)


@router.post("/logout")
async def logout(current_user: UserResponse = Depends(get_current_active_user)):
    """
    Logout user (client should delete token)

    Note: JWT tokens are stateless, so logout is handled client-side by deleting the token.
    """
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: UserResponse = Depends(get_current_active_user)
):
    """
    Get current user information from JWT token
    """
    return current_user


@router.post("/password-reset-request")
async def request_password_reset(request: PasswordResetRequest):
    """
    Request password reset (sends email with reset link)

    Note: Email functionality not yet implemented.
    In production, this would generate a secure reset token and send it via email.
    """
    db = get_database()

    # Find user
    user = await db.users.find_one({"email": request.email})
    if not user:
        # Don't reveal if email exists or not (security best practice)
        return {"message": "If the email exists, a reset link has been sent"}

    # TODO: Generate secure reset token and send email
    # For now, just return success

    return {"message": "If the email exists, a reset link has been sent"}


@router.post("/password-reset-confirm")
async def confirm_password_reset(reset_data: PasswordResetConfirm):
    """
    Confirm password reset with token

    Note: Token validation not yet implemented.
    In production, this would validate the reset token before allowing password change.
    """
    # TODO: Validate reset token and update password
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Password reset with token not yet implemented. Use /auth/reset-password for direct reset."
    )


@router.post("/reset-password")
async def reset_password(reset_data: PasswordReset):
    """
    Reset password directly with email and new password

    - **email**: User's email address
    - **new_password**: New password (will be securely hashed with bcrypt)

    Note: This endpoint allows direct password reset without token validation.
    In production, you may want to require the old password or use token-based reset.
    """
    db = get_database()

    # Find user by email
    user = await db.users.find_one({"email": reset_data.email})
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    # Update password with SECURE bcrypt hash
    hashed_password = get_password_hash(reset_data.new_password)
    await db.users.update_one(
        {"email": reset_data.email},
        {"$set": {
            "hashed_password": hashed_password,
            "updated_at": datetime.utcnow()
        }}
    )

    return {"message": "Password reset successful"}
