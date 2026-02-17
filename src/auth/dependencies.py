"""
Authentication Dependencies

FastAPI dependencies for authentication and user extraction.
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional
from src.auth.jwt import verify_token
from src.auth.models import UserResponse
from src.database import get_database

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> UserResponse:
    """
    Extract and validate the current user from JWT token

    Args:
        credentials: HTTP Bearer token from request header

    Returns:
        UserResponse object

    Raises:
        HTTPException: If token is invalid or user not found
    """
    from src.utils.logger import get_logger
    logger = get_logger(__name__)

    token = credentials.credentials

    # Verify token
    payload = verify_token(token)
    if payload is None:
        logger.error("Token verification failed - invalid or expired token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id: str = payload.get("sub")
    if user_id is None:
        logger.error("Token payload missing 'sub' field")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get user from database
    db = get_database()
    user = await db.users.find_one({"_id": user_id})

    if user is None:
        logger.error(f"User not found in database: {user_id}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
        )

    logger.info(f"User authenticated: {user.get('email')}")
    return UserResponse(**user)


async def get_current_active_user(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """
    Ensure the current user is active
    
    Args:
        current_user: User from get_current_user dependency
        
    Returns:
        UserResponse object
        
    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user


async def get_admin_user(
    current_user: UserResponse = Depends(get_current_active_user)
) -> UserResponse:
    """
    Ensure the current user has admin role
    
    Args:
        current_user: User from get_current_active_user dependency
        
    Returns:
        UserResponse object
        
    Raises:
        HTTPException: If user is not an admin
    """
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user
