#!/bin/bash

echo "🚀 Setting up ASD Detection System - React Frontend + MongoDB Authentication"
echo "================================================================================"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Creating one...${NC}"
    python -m venv venv
fi

# Step 2: Activate virtual environment
echo -e "${GREEN}✓${NC} Activating virtual environment..."
source venv/bin/activate

# Step 3: Install ALL backend dependencies from consolidated requirements.txt
echo -e "${GREEN}✓${NC} Installing all backend dependencies (this may take a few minutes)..."
pip install -q -r requirements.txt

# Step 4: Copy fonts if they exist
if [ -d "frontend/fonts" ]; then
    echo -e "${GREEN}✓${NC} Copying fonts to React project..."
    mkdir -p frontend-react/src/assets
    cp -r frontend/fonts frontend-react/src/assets/
else
    echo -e "${YELLOW}⚠️  Fonts directory not found - skipping${NC}"
fi

# Step 5: Install React dependencies
echo -e "${GREEN}✓${NC} Installing React dependencies..."
cd frontend-react
npm install

# Step 6: Create .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${GREEN}✓${NC} Creating .env file..."
    echo "VITE_API_URL=http://localhost:8000" > .env
fi

cd ..

# Step 7: Create backend .env if it doesn't exist
if [ ! -f ".env" ]; then
    echo -e "${GREEN}✓${NC} Creating backend .env file..."
    cat > .env << EOF
# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=asd_detection

# JWT Configuration
JWT_SECRET_KEY=$(openssl rand -hex 32)
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=30
EOF
else
    # Append MongoDB config if not present
    if ! grep -q "MONGODB_URL" .env; then
        echo -e "${GREEN}✓${NC} Adding MongoDB configuration to .env..."
        cat >> .env << EOF

# MongoDB Configuration
MONGODB_URL=mongodb://localhost:27017
DATABASE_NAME=asd_detection

# JWT Configuration
JWT_SECRET_KEY=$(openssl rand -hex 32)
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=30
EOF
    fi
fi

echo ""
echo "================================================================================"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "================================================================================"
echo ""
echo "📚 Documentation:"
echo "   - Quick Start: README.md"
echo "   - Google OAuth: docs/GOOGLE_OAUTH_SETUP.md"
echo "   - MongoDB Atlas: docs/MONGODB_ATLAS_SETUP.md"
echo "   - Credentials Checklist: docs/SETUP_CREDENTIALS.md"
echo ""
echo "Next steps:"
echo ""
echo "1. Configure .env file with your credentials"
echo "   See docs/SETUP_CREDENTIALS.md for help"
echo ""
echo "2. Start MongoDB:"
echo "   docker run -d -p 27017:27017 --name mongodb mongo:latest"
echo "   OR use MongoDB Atlas (see docs/MONGODB_ATLAS_SETUP.md)"
echo ""
echo "3. Start the backend (in terminal 1):"
echo "   source venv/bin/activate"
echo "   python run_api.py"
echo ""
echo "4. Start the frontend (in terminal 2):"
echo "   cd frontend-react"
echo "   npm run dev"
echo ""
echo "5. Open http://localhost:5173 and create an account!"
echo ""
echo "================================================================================"
