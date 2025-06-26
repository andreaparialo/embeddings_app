#!/bin/bash

# Startup script for running the app with conda-installed faiss-gpu

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}✅ Starting Hybrid Product Search Engine with GPU acceleration...${NC}"

# Check if conda is available
if [ ! -d "$HOME/miniconda3" ]; then
    echo -e "${RED}❌ Miniconda not found at $HOME/miniconda3${NC}"
    echo -e "${YELLOW}Please install miniconda first${NC}"
    exit 1
fi

# Initialize conda
source ~/miniconda3/etc/profile.d/conda.sh

# Activate the faiss environment
echo -e "${GREEN}🔄 Activating conda environment 'faiss_env'...${NC}"
conda activate faiss_env

# Check if activation was successful
if [ "$CONDA_DEFAULT_ENV" != "faiss_env" ]; then
    echo -e "${RED}❌ Failed to activate conda environment${NC}"
    exit 1
fi

# Display GPU and FAISS status
echo -e "${GREEN}🎮 Checking GPU and FAISS status...${NC}"
python -c "
import faiss
import torch
print(f'✅ FAISS version: {faiss.__version__ if hasattr(faiss, \"__version__\") else \"unknown\"}')
print(f'✅ FAISS GPU support: {faiss.get_num_gpus()} GPUs detected')
print(f'✅ PyTorch CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✅ PyTorch GPU count: {torch.cuda.device_count()}')
    for i in range(min(torch.cuda.device_count(), 2)):  # Show first 2 GPUs
        print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
"

# Set environment variables
export USE_FAISS_GPU=true
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all 4 GPUs

# Fix MKL threading conflict with OpenMP
export MKL_THREADING_LAYER=GNU
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo -e "${GREEN}📊 This may take a few minutes to load the model and data...${NC}"
echo -e "${GREEN}🌐 Web interface will be available at: http://127.0.0.1:8080${NC}"

# Run the application from parent directory
cd "$(dirname "$0")/.."
python -m uvicorn old_app.app:app --host 127.0.0.1 --port 8080 --reload 