#!/bin/bash
# Setup script for Open Source Web Scraper Agent
# This script installs dependencies and pulls required Ollama models

set -e  # Exit on error

echo "=============================================="
echo "Open Source Web Scraper Agent - Setup"
echo "=============================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored status
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Check Python version
echo "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

    if [ "$MAJOR" -ge 3 ] && [ "$MINOR" -ge 9 ]; then
        print_status "Python $PYTHON_VERSION detected"
    else
        print_error "Python 3.9+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.10+"
    exit 1
fi

# Check if Ollama is installed
echo ""
echo "Checking Ollama installation..."
if command -v ollama &> /dev/null; then
    print_status "Ollama is installed"

    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_status "Ollama is running"
    else
        print_warning "Ollama is not running. Starting Ollama..."
        ollama serve &
        sleep 3
    fi
else
    print_error "Ollama is not installed"
    echo ""
    echo "Please install Ollama from: https://ollama.com/download"
    echo "Or run: curl -fsSL https://ollama.com/install.sh | sh"
    exit 1
fi

# Create virtual environment if it doesn't exist
echo ""
echo "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_status "Virtual environment activated"

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1
print_status "pip upgraded"

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."
pip install -r requirements.txt
print_status "Python dependencies installed"

# Install Playwright for Crawl4AI (JavaScript rendering)
echo ""
echo "Installing Playwright browsers for Crawl4AI..."
python -m playwright install chromium > /dev/null 2>&1 || print_warning "Playwright install skipped (optional)"
print_status "Playwright setup complete"

# Pull Ollama models
echo ""
echo "=============================================="
echo "Pulling Ollama models (this may take a while)..."
echo "=============================================="
echo ""

# Primary model - Qwen 2.5 14B (best for reasoning)
echo "Pulling qwen2.5:14b (primary model, ~8GB)..."
if ollama pull qwen2.5:14b; then
    print_status "qwen2.5:14b pulled successfully"
else
    print_warning "Failed to pull qwen2.5:14b, trying smaller version..."
    if ollama pull qwen2.5:7b; then
        print_status "qwen2.5:7b pulled successfully (fallback)"
    else
        print_error "Failed to pull Qwen model"
    fi
fi

# Fallback model - Mistral 7B
echo ""
echo "Pulling mistral:7b-instruct (fallback model, ~4GB)..."
if ollama pull mistral:7b-instruct; then
    print_status "mistral:7b-instruct pulled successfully"
else
    print_warning "Failed to pull mistral:7b-instruct"
fi

# Optional - DeepSeek R1 for reasoning
echo ""
echo "Pulling deepseek-r1:8b (reasoning model, ~5GB)..."
if ollama pull deepseek-r1:8b; then
    print_status "deepseek-r1:8b pulled successfully"
else
    print_warning "Failed to pull deepseek-r1:8b (optional)"
fi

# Create .env file if it doesn't exist
echo ""
echo "Checking environment configuration..."
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        cp .env.example .env
        print_status ".env file created from .env.example"
    else
        echo "# Environment variables for Open Source Web Scraper Agent" > .env
        echo "" >> .env
        echo "# Optional: HuggingFace token for fallback LLM" >> .env
        echo "# Get your free token at: https://huggingface.co/settings/tokens" >> .env
        echo "# HF_TOKEN=your_token_here" >> .env
        print_status ".env file created"
    fi
else
    print_status ".env file already exists"
fi

# Verify installation
echo ""
echo "=============================================="
echo "Verifying installation..."
echo "=============================================="
echo ""

# Test import
python3 -c "
from src.config import get_config
from src.llm.client import health_check
print('Module imports successful')
" && print_status "Python modules import correctly"

# Check Ollama models
echo ""
echo "Installed Ollama models:"
ollama list

# Print success message
echo ""
echo "=============================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=============================================="
echo ""
echo "To start the server:"
echo "  1. Activate virtual environment: source venv/bin/activate"
echo "  2. Start server: uvicorn app.main:app --reload"
echo ""
echo "The API will be available at:"
echo "  - http://localhost:8000"
echo "  - API docs: http://localhost:8000/docs"
echo ""
echo "Test with:"
echo '  curl -X POST http://localhost:8000/research \'
echo '    -H "Content-Type: application/json" \'
echo '    -d '\''{"prompt": "What are the benefits of renewable energy?"}'\'''
echo ""
echo "Optional: Set HF_TOKEN in .env for HuggingFace fallback"
echo ""
