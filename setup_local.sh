#!/bin/bash
set -e

echo "=============================================="
echo "  MLOps Project - Local Setup Script"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

# Step 1: Check prerequisites
echo ""
echo "Step 1: Checking prerequisites..."

command -v python3 >/dev/null 2>&1 || { print_error "Python3 is required but not installed."; exit 1; }
command -v docker >/dev/null 2>&1 || { print_warning "Docker is not installed. Container features will be unavailable."; }
command -v git >/dev/null 2>&1 || { print_error "Git is required but not installed."; exit 1; }

print_status "Prerequisites check completed"

# Step 2: Create virtual environment
echo ""
echo "Step 2: Setting up Python virtual environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Step 3: Install dependencies
echo ""
echo "Step 3: Installing Python dependencies..."

pip install --upgrade pip -q
pip install -r requirements-dev.txt -q
print_status "Dependencies installed"

# Step 4: Initialize DVC
echo ""
echo "Step 4: Initializing DVC..."

if [ ! -d ".dvc" ]; then
    dvc init
    print_status "DVC initialized"
else
    print_status "DVC already initialized"
fi

# Step 5: Create directory structure
echo ""
echo "Step 5: Creating directory structure..."

mkdir -p data/{raw,processed}
mkdir -p artifacts
mkdir -p logs
mkdir -p mlruns

print_status "Directory structure created"

# Step 6: Create .env file
echo ""
echo "Step 6: Creating environment configuration..."

if [ ! -f ".env" ]; then
    cp .env.example .env 2>/dev/null || cat > .env << 'ENVEOF'
LOG_LEVEL=INFO
MODEL_PATH=artifacts/model.pt
MLFLOW_TRACKING_URI=mlruns
ENVEOF
    print_status ".env file created"
else
    print_status ".env file already exists"
fi

echo ""
echo "=============================================="
echo "  Setup completed successfully!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Activate venv: source venv/bin/activate"
echo "  2. Download data: ./scripts/download_data.sh"
echo "  3. Train model:   ./scripts/train_model.sh"
echo "  4. Run API:       ./scripts/run_api.sh"
echo ""
