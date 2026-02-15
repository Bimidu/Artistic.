# ASD Detection System

Multimodal autism spectrum disorder (ASD) detection from conversational features using machine learning.

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 20.19+ or 22.12+
- MongoDB (local) OR MongoDB Atlas (cloud)

### Installation

```bash
# 1. Clone and setup Python environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install all dependencies
pip install -r requirements.txt

# 3. Download spaCy language model
python -m spacy download en_core_web_lg

# 4. Setup React frontend
cd frontend-react
npm install
cd ..

# 5. Configure environment variables
cp .env.example .env
# Edit .env with your MongoDB and Google OAuth credentials
```

### Environment Configuration

Edit the `.env` file in the project root with your credentials:

**Required fields:**
```bash
# MongoDB - choose one:
MONGODB_URL=mongodb://localhost:27017  # Local MongoDB
# OR
MONGODB_URL=mongodb+srv://user:pass@cluster.mongodb.net/?retryWrites=true  # MongoDB Atlas

# JWT Secret - generate a secure key
JWT_SECRET_KEY=your-super-secret-key-here
```

**Optional (for Google OAuth):**
```bash
GOOGLE_CLIENT_ID=your-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-secret
```

See the `.env` file for detailed instructions and `docs/` folder for setup guides.

### Running the Application

```bash
# Terminal 1: Start MongoDB (if using local)
mongod
# OR use MongoDB Atlas (cloud) - just update .env

# Terminal 2: Start backend
python run_api.py

# Terminal 3: Start frontend
cd frontend-react
npm run dev
```

Access the application at: **http://localhost:5173**

---

## 📚 Documentation

- **[System Architecture](./SYSTEM_ARCHITECTURE.md)** - Complete technical documentation
- **[Google OAuth Setup](./docs/GOOGLE_OAUTH_SETUP.md)** - Setup "Continue with Google"
- **[MongoDB Atlas Setup](./docs/MONGODB_ATLAS_SETUP.md)** - Cloud database setup
- **[Setup Credentials](./docs/SETUP_CREDENTIALS.md)** - Quick credential checklist

---

## 🎯 Features

### Authentication
- ✅ Email/password registration and login
- ✅ Google OAuth ("Continue with Google")
- ✅ JWT token-based authentication
- ✅ Protected routes and API endpoints

### Prediction (User Mode)
- ✅ Audio file upload (.wav, .mp3, .flac)
- ✅ Text transcript input
- ✅ CHAT file upload (.cha format)
- ✅ Model selection or auto-best model
- ✅ Multi-component fusion
- ✅ Confidence scores and class probabilities

### Training Mode
- ✅ Dataset selection and management
- ✅ Feature extraction (audio, text, chat)
- ✅ Model training (XGBoost, LightGBM, etc.)
- ✅ Component-specific configuration
- ✅ Training progress and results

---

## 🛠️ Technology Stack

### Backend
- **FastAPI** - Modern Python web framework
- **MongoDB** - NoSQL database (local or Atlas cloud)
- **JWT** - Secure authentication
- **XGBoost/LightGBM** - Machine learning models
- **spaCy** - Natural language processing
- **librosa** - Audio feature extraction

### Frontend
- **React** - UI framework
- **Vite** - Build tool and dev server
- **React Router** - Client-side routing
- **Tailwind CSS** - Styling
- **Context API** - State management

---

## 📁 Project Structure

```
.
├── src/                    # Backend source code
│   ├── api/               # FastAPI application and routes
│   ├── auth/              # Authentication (JWT, Google OAuth)
│   ├── models/            # ML model definitions
│   └── utils/             # Utilities and helpers
├── frontend-react/        # React frontend application
│   ├── src/
│   │   ├── components/   # Reusable React components
│   │   ├── pages/        # Page components
│   │   ├── services/     # API service layer
│   │   └── utils/        # Frontend utilities
├── data/                  # Datasets
├── models/                # Trained model files
├── docs/                  # Documentation and setup guides
├── requirements.txt       # Python dependencies
└── SYSTEM_ARCHITECTURE.md # Complete technical documentation
```

---

## 🔧 Development

### Backend API Documentation

Once the backend is running, visit:
- **API Docs**: http://localhost:8000/docs
- **Alternative**: http://localhost:8000/redoc

### Code Quality

```bash
# Format code
black src/

# Lint
flake8 src/

# Type checking
mypy src/

# Run tests
pytest
```

---

## 🐛 Troubleshooting

### Frontend Build Issues

If you see Tailwind CSS errors:
```bash
cd frontend-react
npm install -D @tailwindcss/postcss
npm run dev
```

### MongoDB Connection Issues

- **Local MongoDB**: Ensure `mongod` is running
- **MongoDB Atlas**: Check connection string in `.env` and IP whitelist

### Google OAuth Issues

- Verify client ID and secret in `.env`
- Check redirect URIs in Google Cloud Console
- See `docs/GOOGLE_OAUTH_SETUP.md` for detailed setup

---

## 📝 License

This project is for research and educational purposes.

---

## 👥 Contributing

This is an academic research project. For questions or contributions, please refer to the documentation.

---

## 🙏 Acknowledgments

Built for autism spectrum disorder detection research using multimodal conversational features.
