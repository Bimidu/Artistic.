"""
Google OAuth Routes

Handles Google OAuth 2.0 authentication flow.
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import RedirectResponse
import os
from src.auth.models import get_password_hash
from src.auth.jwt import create_access_token
from src.database import get_database
import uuid
from datetime import datetime
from src.utils.logger import get_logger
import httpx
from jose import jwt, JWTError
import time

logger = get_logger(__name__)

router = APIRouter(prefix="/auth/google", tags=["Google OAuth"])

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")

async def verify_google_token_async(token: str, client_id: str) -> dict:
    """
    Verify Google ID token asynchronously using httpx.

    Args:
        token: The Google ID token to verify
        client_id: Expected Google OAuth client ID

    Returns:
        dict: Decoded token payload with user info

    Raises:
        ValueError: If token verification fails
    """
    try:
        unverified = jwt.decode(
            token,
            "",
            options={
                "verify_signature": False,
                "verify_aud": False,
                "verify_iat": False,
                "verify_exp": False,
                "verify_nbf": False,
                "verify_iss": False,
                "verify_sub": False,
                "verify_jti": False,
                "verify_at_hash": False
            }
        )

        logger.info(f"Token issuer: {unverified.get('iss')}")
        logger.info(f"Token audience: {unverified.get('aud')}")
        logger.info(f"Expected client_id: {client_id}")

        # Verify basic claims manually
        if unverified.get("iss") not in ["https://accounts.google.com", "accounts.google.com"]:
            raise ValueError("Invalid issuer")

        if unverified.get("aud") != client_id:
            raise ValueError(f"Invalid audience: expected {client_id}, got {unverified.get('aud')}")

        # Check expiration
        if unverified.get("exp", 0) < time.time():
            raise ValueError("Token expired")

        return unverified

    except JWTError as e:
        logger.error(f"Failed to decode Google token: {e}")
        raise ValueError(f"Invalid token: {e}")
    except Exception as e:
        logger.error(f"Unexpected error verifying Google token: {e}")
        raise ValueError(f"Token verification failed: {e}")

@router.get("/login")
async def google_login():
    """
    Redirect to Google OAuth login page

    Returns redirect URL to Google's OAuth consent screen
    """
    if not GOOGLE_CLIENT_ID:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Google OAuth is not configured. Please set GOOGLE_CLIENT_ID in environment variables."
        )

    google_auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        "&response_type=code"
        "&scope=openid email profile"
        "&access_type=offline"
        "&prompt=consent"
    )

    logger.info(f"Redirecting to Google OAuth: {google_auth_url[:100]}...")
    logger.info(f"GOOGLE_CLIENT_ID: {GOOGLE_CLIENT_ID[:20]}...")
    logger.info(f"GOOGLE_REDIRECT_URI: {GOOGLE_REDIRECT_URI}")

    return RedirectResponse(url=google_auth_url, status_code=302)


@router.get("/callback")
async def google_callback(code: str = None, error: str = None):
    """
    Handle Google OAuth callback

    Exchanges authorization code for tokens, verifies user,
    and creates/updates user in database.
    """
    logger.info(f"Google OAuth callback received - code: {code is not None}, error: {error}")

    if error:
        logger.error(f"Google OAuth returned error: {error}")
        return RedirectResponse(url=f"{FRONTEND_URL}/?error=google_auth_{error}")

    if not code:
        logger.error("No authorization code received from Google")
        return RedirectResponse(url=f"{FRONTEND_URL}/?error=no_auth_code")

    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Google OAuth is not configured"
        )

    start_time = time.time()

    # Exchange authorization code for tokens
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    try:
        logger.info("Starting token exchange with Google...")
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(token_url, data=data)
            tokens = response.json()
        logger.info(f"Token exchange completed in {time.time() - start_time:.2f}s")

        if "error" in tokens:
            logger.error(f"Google OAuth token exchange failed: {tokens}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Google OAuth error: {tokens.get('error_description', 'Unknown error')}"
            )

        # Verify ID token and extract user info
        logger.info("Starting ID token verification...")
        verify_start = time.time()
        idinfo = await verify_google_token_async(tokens["id_token"], GOOGLE_CLIENT_ID)
        logger.info(f"ID token verification completed in {time.time() - verify_start:.2f}s")

        email = idinfo["email"]
        name = idinfo.get("name", email.split("@")[0])
        google_id = idinfo["sub"]
        avatar_url = idinfo.get("picture")

        logger.info(f"Google OAuth successful for email: {email}")

    except ValueError as e:
        logger.error(f"Invalid Google token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid token: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Google OAuth error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )

    # Find or create user in database
    logger.info(f"Looking up user in database: {email}")
    db_start = time.time()
    db = get_database()
    user = await db.users.find_one({"email": email})
    logger.info(f"Database lookup completed in {time.time() - db_start:.2f}s")

    if user:
        # Update Google ID and avatar if not set
        if not user.get("google_id"):
            await db.users.update_one(
                {"_id": user["_id"]},
                {
                    "$set": {
                        "google_id": google_id,
                        "avatar_url": avatar_url,
                        "is_verified": True,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            user["google_id"] = google_id
            user["avatar_url"] = avatar_url

        logger.info(f"Existing user logged in via Google: {user['_id']}")
    else:
        # Create new user with random hashed password (user can't use it directly)
        user_id = str(uuid.uuid4())
        # Generate a short random password for OAuth users (they won't use it)
        random_password = str(uuid.uuid4())[:16]  # Keep it short, under 72 bytes
        user = {
            "_id": user_id,
            "email": email,
            "hashed_password": get_password_hash(random_password),  # Random bcrypt hash
            "full_name": name,
            "google_id": google_id,
            "avatar_url": avatar_url,
            "role": "user",
            "is_active": True,
            "is_verified": True,  # Google verified email
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }

        await db.users.insert_one(user)
        logger.info(f"New user created via Google: {user_id}")

    # Create JWT access token
    logger.info("Creating JWT access token...")
    jwt_start = time.time()
    access_token = create_access_token(
        data={"sub": user["_id"], "email": user["email"], "role": user.get("role", "user")}
    )
    logger.info(f"JWT token created in {time.time() - jwt_start:.2f}s")

    # Redirect to frontend with token
    redirect_url = f"{FRONTEND_URL}/auth/google/callback?token={access_token}"
    logger.info(f"Total OAuth callback processing time: {time.time() - start_time:.2f}s")
    logger.info(f"Redirecting to: {redirect_url[:60]}...")
    return RedirectResponse(url=redirect_url, status_code=302)
