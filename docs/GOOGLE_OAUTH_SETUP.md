# Google OAuth Setup Guide

## Step 1: Create Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click "Select a project" → "New Project"
3. Name it "ASD Detection" and click "Create"

## Step 2: Configure OAuth Consent Screen

1. In the left sidebar, go to **APIs & Services** → **OAuth consent screen**
2. Choose **External** (for testing) or **Internal** (if you have Google Workspace)
3. Fill in the required fields:
   - **App name**: ASD Detection System
   - **User support email**: Your email
   - **Developer contact**: Your email
4. Click **Save and Continue**
5. **Scopes**: Click "Add or Remove Scopes"
   - Add: `userinfo.email`
   - Add: `userinfo.profile`
   - Add: `openid`
6. Click **Save and Continue**
7. **Test users** (for External apps): Add your email for testing
8. Click **Save and Continue** → **Back to Dashboard**

## Step 3: Create OAuth Credentials

1. Go to **APIs & Services** → **Credentials**
2. Click **+ Create Credentials** → **OAuth client ID**
3. Choose **Web application**
4. Configure:
   - **Name**: ASD Detection Web Client
   - **Authorized JavaScript origins**:
     - `http://localhost:5173` (React dev server)
     - `http://localhost:8000` (FastAPI backend)
   - **Authorized redirect URIs**:
     - `http://localhost:8000/auth/google/callback`
     - `http://localhost:5173/auth/google/callback`
5. Click **Create**
6. **SAVE** your Client ID and Client Secret!

## Step 4: Add to Environment Variables

Update your backend `.env` file:

```bash
# Google OAuth
GOOGLE_CLIENT_ID=your-client-id-from-step-3.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret-from-step-3
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
```

Update your frontend `.env` file:

```bash
VITE_GOOGLE_CLIENT_ID=your-client-id-from-step-3.apps.googleusercontent.com
```

## Step 5: Implement Google OAuth Backend Routes

Create the file `/Users/user/Desktop/Research/Artistic./src/auth/google_oauth.py`:

```python
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import RedirectResponse
from google.oauth2 import id_token
from google.auth.transport import requests
import os
from src.auth.models import UserInDB, get_password_hash
from src.auth.jwt import create_access_token
from src.database import get_database
import uuid
from datetime import datetime

router = APIRouter(prefix="/auth/google", tags=["Google OAuth"])

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:5173")


@router.get("/login")
async def google_login():
    """Redirect to Google OAuth login"""
    google_auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={os.getenv('GOOGLE_REDIRECT_URI')}"
        "&response_type=code"
        "&scope=openid email profile"
    )
    return RedirectResponse(url=google_auth_url)


@router.get("/callback")
async def google_callback(code: str):
    """Handle Google OAuth callback"""
    import httpx
    
    # Exchange code for tokens
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": os.getenv("GOOGLE_REDIRECT_URI"),
        "grant_type": "authorization_code",
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(token_url, data=data)
        tokens = response.json()
    
    if "error" in tokens:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Google OAuth error: {tokens.get('error_description', 'Unknown error')}"
        )
    
    # Verify ID token
    try:
        idinfo = id_token.verify_oauth2_token(
            tokens["id_token"],
            requests.Request(),
            GOOGLE_CLIENT_ID
        )
        
        email = idinfo["email"]
        name = idinfo.get("name", email.split("@")[0])
        google_id = idinfo["sub"]
        avatar_url = idinfo.get("picture")
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid token: {str(e)}"
        )
    
    # Find or create user
    db = get_database()
    user = await db.users.find_one({"email": email})
    
    if user:
        # Update Google ID if not set
        if not user.get("google_id"):
            await db.users.update_one(
                {"_id": user["_id"]},
                {"$set": {"google_id": google_id, "avatar_url": avatar_url}}
            )
            user["google_id"] = google_id
            user["avatar_url"] = avatar_url
    else:
        # Create new user
        user_id = str(uuid.uuid4())
        user = {
            "_id": user_id,
            "email": email,
            "hashed_password": get_password_hash(str(uuid.uuid4())),  # Random password
            "full_name": name,
            "google_id": google_id,
            "avatar_url": avatar_url,
            "is_active": True,
            "is_verified": True,  # Google verified
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        await db.users.insert_one(user)
    
    # Create JWT token
    access_token = create_access_token(
        data={"sub": user["_id"], "email": user["email"]}
    )
    
    # Redirect to frontend with token
    return RedirectResponse(
        url=f"{FRONTEND_URL}/auth/google/callback?token={access_token}"
    )
```

## Step 6: Update FastAPI App

Add to `src/api/app.py`:

```python
from src.auth.google_oauth import router as google_router

# After including auth_router
app.include_router(google_router)
```

## Step 7: Update React Google Callback Handler

Create `/Users/user/Desktop/Research/Artistic./frontend-react/src/pages/Auth/GoogleCallbackPage.jsx`:

```javascript
import React, { useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { setToken } from '@utils/storage';
import { useAuth } from '@hooks/useAuth';

export const GoogleCallbackPage = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { refreshUser } = useAuth();

  useEffect(() => {
    const token = searchParams.get('token');
    
    if (token) {
      setToken(token);
      refreshUser().then(() => {
        navigate('/');
      });
    } else {
      navigate('/login');
    }
  }, [searchParams, navigate, refreshUser]);

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="text-center">
        <div className="spinner w-12 h-12 mx-auto mb-4"></div>
        <p className="text-lime-700">Completing Google sign-in...</p>
      </div>
    </div>
  );
};
```

## Step 8: Add Route in Router

Update `src/router.jsx`:

```javascript
import { GoogleCallbackPage } from '@pages/Auth/GoogleCallbackPage';

// Add this route
{
  path: '/auth/google/callback',
  element: <GoogleCallbackPage />,
},
```

## Step 9: Install Dependencies

```bash
# Backend
pip install httpx google-auth google-auth-oauthlib

# Frontend - already installed
```

## Testing

1. Start MongoDB
2. Start backend: `python run_api.py`
3. Start frontend: `cd frontend-react && npm run dev`
4. Click "Continue with Google" button
5. Sign in with Google
6. You should be redirected back to the app and logged in!

## Production Considerations

- Use HTTPS URLs for redirect URIs
- Store secrets in environment variables (never commit)
- Update authorized origins to your production domain
- Consider using refresh tokens for long-lived sessions
