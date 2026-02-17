# 🔐 Setup Instructions - Add Your Credentials

## Quick Setup Checklist

### 1. MongoDB Atlas (Cloud Database)

☐ **Sign up** at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas/register)  
☐ **Create cluster** (Free M0 tier)  
☐ **Create database user** with password  
☐ **Whitelist IP**: `0.0.0.0/0` (for development)  
☐ **Get connection string** from "Connect" button  

**Add to `.env` file:**
```bash
MONGODB_URL=mongodb+srv://username:password@cluster.xxxxx.mongodb.net/?retryWrites=true&w=majority
DATABASE_NAME=asd_detection
```

---

### 2. Google OAuth (Optional - "Continue with Google" button)

☐ **Go to** [console.cloud.google.com](https://console.cloud.google.com)  
☐ **Create new project** (e.g., "ASD Detection")  
☐ **Enable** APIs & Services → OAuth consent screen  
☐ **Create** OAuth 2.0 Client ID (Web application)  
☐ **Add redirect URIs**:
   - `http://localhost:8000/auth/google/callback`
   - `http://localhost:5173/auth/google/callback`

**Add to `.env` file:**
```bash
GOOGLE_CLIENT_ID=your-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-secret-here
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
```

**Add to `frontend-react/.env` file:**
```bash
VITE_GOOGLE_CLIENT_ID=your-id.apps.googleusercontent.com
```

---

## Installation Commands

### Backend Dependencies

```bash
# Activate virtual environment
source venv/bin/activate

# Install ALL auth dependencies at once
pip install pymongo motor pydantic-settings 'python-jose[cryptography]' 'passlib[bcrypt]' python-multipart httpx google-auth google-auth-oauthlib

# Or use the requirements files
pip install -r requirements-auth.txt
pip install -r requirements-google-oauth.txt
```

---

## Test Your Setup

### 1. **Start MongoDB**
```bash
# If using MongoDB Atlas - just update .env with connection string
# If using local MongoDB:
docker run -d -p 27017:27017 --name mongodb mongo:latest
# OR
mongod
```

### 2. **Start Backend**
```bash
python run_api.py
```

Look for:
- ✓ Connected to MongoDB at mongodb+srv://... (or mongodb://localhost:27017)
- Server running at http://localhost:8000

### 3. **Start Frontend**
```bash
cd frontend-react
npm run dev
```

Open: http://localhost:5173

### 4. **Test Features**

**Register/Login:**
- Regular email/password registration works immediately
- MongoDB stores users

**Google OAuth (if configured):**
- Click "Continue with Google"
- Sign in with Google
- Should redirect back and auto-login

---

## Detailed Setup Guides

- 📘 **[GOOGLE_OAUTH_SETUP.md](file:///Users/user/Desktop/Research/Artistic./GOOGLE_OAUTH_SETUP.md)** - Complete Google OAuth setup
- 📗 **[MONGODB_ATLAS_SETUP.md](file:///Users/user/Desktop/Research/Artistic./MONGODB_ATLAS_SETUP.md)** - Complete MongoDB Atlas setup

---

## Environment Files Summary

### Backend `.env`
```bash
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true
DATABASE_NAME=asd_detection
JWT_SECRET_KEY=your-secret-key-here
GOOGLE_CLIENT_ID=your-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-secret
GOOGLE_REDIRECT_URI=http://localhost:8000/auth/google/callback
FRONTEND_URL=http://localhost:5173
```

### Frontend `frontend-react/.env`
```bash
VITE_API_URL=http://localhost:8000
VITE_GOOGLE_CLIENT_ID=your-id.apps.googleusercontent.com
```

---

## ✅ What's Been Implemented

### Backend
- ✅ Google OAuth routes (`/auth/google/login`, `/auth/google/callback`)
- ✅ MongoDB Atlas support (just add connection string)
- ✅ User creation/update on Google sign-in
- ✅ JWT token generation after OAuth

### Frontend  
- ✅ Google callback page handler
- ✅ "Continue with Google" button (already in LoginForm/RegisterForm)
- ✅ Token storage and redirect after OAuth

---

## 🚀 You're Ready!

1. **Add your MongoDB Atlas connection string** to `.env`
2. **Add Google credentials** to `.env` (optional)
3. **Install dependencies** (see commands above)
4. **Start the app** and test!

See detailed guides for step-by-step instructions on getting credentials.
