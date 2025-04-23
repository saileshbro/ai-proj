#!/bin/bash

# Colors for pretty output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Create virtual environment
echo -e "${BLUE}Creating virtual environment...${NC}"
python -m venv venv

# Activate virtual environment
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    echo -e "${BLUE}Activating virtual environment (Windows)...${NC}"
    source venv/Scripts/activate
else
    echo -e "${BLUE}Activating virtual environment (Unix)...${NC}"
    source venv/bin/activate
fi

# Install requirements
echo -e "${BLUE}Installing requirements...${NC}"
pip install -r requirements.txt

# Check environment setup
echo -e "${BLUE}Checking environment setup...${NC}"
python check_setup.py

# Make predict.py executable
chmod +x predict.py

echo -e "${GREEN}Setup completed!${NC}"
echo -e "\nTo start using the project:"
echo -e "${BLUE}1. Train the model:${NC} python -m jupyter notebook model_training.ipynb"
echo -e "${BLUE}2. Test predictions:${NC} ./predict.py \"तपाईंको फिल्म राम्रो छ\""
echo -e "${BLUE}3. Start API server:${NC} uvicorn app:app --reload"