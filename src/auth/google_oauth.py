"""
Google OAuth Routes

Handles Google OAuth 2.0 authentication flow.
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import RedirectResponse
from google.oauth2 import id_token
from google.auth.transport import requests
import os
from src.auth.models import get_password_hash
from src.auth.jwt import create_access_token
from src.database import get_database
import uuid
from datetime import datetime
from src.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/auth/google", tags=["Google OAuth"])

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/google/callback")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")


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
    
    return RedirectResponse(url=google_auth_url)


@router.get("/callback")
async def google_callback(code: str):
    """
    Handle Google OAuth callback
    
    Exchanges authorization code for tokens, verifies user,
    and creates/updates user in database.
    """
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="Google OAuth is not configured"
        )
    
    import httpx
    
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
        async with httpx.AsyncClient() as client:
            response = await client.post(token_url, data=data)
            tokens = response.json()
        
        if "error" in tokens:
            logger.error(f"Google OAuth token exchange failed: {tokens}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Google OAuth error: {tokens.get('error_description', 'Unknown error')}"
            )
        
        # Verify ID token and extract user info
        idinfo = id_token.verify_oauth2_token(
            tokens["id_token"],
            requests.Request(),
            GOOGLE_CLIENT_ID
        )
        
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
    db = get_database()
    user = await db.users.find_one({"email": email})
    
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
        # Create new user
        user_id = str(uuid.uuid4())
        user = {
            "_id": user_id,
            "email": email,
            "hashed_password": get_password_hash(str(uuid.uuid4())),  # Random password (user can't use it)
            "full_name": name,
            "google_id": google_id,
            "avatar_url": avatar_url,
            "is_active": True,
            "is_verified": True,  # Google verified email
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        
        await db.users.insert_one(user)
        logger.info(f"New user created via Google: {user_id}")
    
    # Create JWT access token
    access_token = create_access_token(
        data={"sub": user["_id"], "email": user["email"]}
    )
    
    # Redirect to frontend with token
    redirect_url = f"{FRONTEND_URL}/auth/google/callback?token={access_token}"
    return RedirectResponse(url=redirect_url)
