# React Frontend Migration - Quick Start Guide

## 🎯 What Was Built

A complete React application with MongoDB authentication, replacing the vanilla JS frontend.

## ⚡ Quick Setup

Run the automated setup script:

```bash
chmod +x setup-react-auth.sh
./setup-react-auth.sh
```

Or manually:

```bash
# 1. Install backend deps
source venv/bin/activate
pip install pymongo motor pydantic-settings 'python-jose[cryptography]' 'passlib[bcrypt]' python-multipart

# 2. Copy fonts
cp -r frontend/fonts frontend-react/src/assets/

# 3. Install React deps
cd frontend-react && npm install

# 4. Start MongoDB
docker run -d -p 27017:27017 --name mongodb mongo:latest

# 5. Start backend (terminal 1)
python run_api.py

# 6. Start frontend (terminal 2)
cd frontend-react && npm run dev
```

## 🔗 URLs

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

## 📱 Features

### Authentication
- ✅ User registration with validation
- ✅ Login with JWT tokens
- ✅ Protected routes
- ✅ Auto-redirect on auth

### Prediction (User Mode)
- ✅ Audio file upload
- ✅ Text input
- ✅ CHAT file upload
- ✅ Model selection
- ✅ Multi-component fusion
- ✅ Results display

### Training Mode
- ✅ Dataset selection
- ✅ Feature extraction
- ✅ Model training
- ✅ Component configuration
- ✅ Results visualization

## 📚 Documentation

See [walkthrough.md](file:///Users/user/.gemini/antigravity/brain/362fb9c7-4b12-41ba-87da-472526db9404/walkthrough.md) for complete documentation.
