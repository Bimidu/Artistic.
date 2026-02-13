#!/bin/bash

# Quick Setup Script for Cloud Storage
# Author: Bimidu Gunathilake
# Date: 2026-02-13

set -e  # Exit on error

echo ""
echo "======================================================================"
echo "  ARTISTIC ASD Detection - Cloud Storage Quick Setup"
echo "======================================================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if .env exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}Creating .env from template...${NC}"
    cp .env.example .env
    echo -e "${GREEN}✓${NC} Created .env file"
    echo ""
    echo -e "${YELLOW}⚠️  Please update the following in .env:${NC}"
    echo "   HF_DATASET_REPO=your-username/artistic-asd-datasets"
    echo "   HF_MODEL_REPO=your-username/artistic-asd-models"
    echo ""
    read -p "Press Enter after updating .env file..."
fi

# Check if huggingface-hub is installed
if ! python3 -c "import huggingface_hub" &> /dev/null; then
    echo -e "${RED}✗${NC} huggingface-hub not found"
    echo "   Installing..."
    pip3 install --upgrade huggingface-hub
    echo -e "${GREEN}✓${NC} Installed huggingface-hub"
fi

# Check authentication
echo ""
echo "Checking HuggingFace authentication..."
if python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" &> /dev/null; then
    echo -e "${GREEN}✓${NC} Already authenticated with HuggingFace Hub"
    USERNAME=$(python3 -c "from huggingface_hub import HfApi; print(HfApi().whoami().get('name', 'Unknown'))" 2>/dev/null || echo "Unknown")
    echo "   Logged in as: $USERNAME"
else
    echo -e "${YELLOW}⚠️  Not authenticated with HuggingFace Hub${NC}"
    echo ""
    echo "Steps to authenticate:"
    echo "1. Go to: https://huggingface.co/settings/tokens"
    echo "2. Create a token with 'write' permission"
    echo "3. Copy the token"
    echo ""
    read -p "Ready to login? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python3 scripts/hf_login.py
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓${NC} Successfully authenticated"
        else
            echo -e "${RED}✗${NC} Authentication failed"
            exit 1
        fi
    else
        echo -e "${RED}✗${NC} Authentication skipped"
        echo "   Run 'python3 scripts/hf_login.py' later to authenticate"
        exit 1
    fi
fi

# Run tests
echo ""
echo "Running cloud storage tests..."
python tests/test_cloud_storage.py

# Check test result
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}======================================================================"
    echo "  ✓ Cloud Storage Setup Complete!"
    echo "======================================================================${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Upload your models:"
    echo "     python scripts/cloud_sync.py upload-models"
    echo ""
    echo "  2. (Optional) Upload datasets:"
    echo "     python scripts/cloud_sync.py upload-datasets"
    echo ""
    echo "  3. Check status:"
    echo "     python scripts/cloud_sync.py status"
    echo ""
    echo "  4. Start using cloud storage:"
    echo "     python run_api.py"
    echo ""
    echo "Documentation:"
    echo "  - Quick Start: MIGRATION_GUIDE.md"
    echo "  - Full Docs: docs/CLOUD_SETUP.md"
    echo ""
else
    echo ""
    echo -e "${RED}======================================================================"
    echo "  ✗ Cloud Storage Setup Failed"
    echo "======================================================================${NC}"
    echo ""
    echo "Common fixes:"
    echo "  1. Ensure you're authenticated: huggingface-cli login"
    echo "  2. Update .env with your HuggingFace repository names"
    echo "  3. Create repositories on huggingface.co"
    echo ""
    echo "For help, see: docs/CLOUD_SETUP.md"
    echo ""
    exit 1
fi
