#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting FAST Delta Indexing for IAN/resized${NC}"
echo -e "${GREEN}================================================${NC}"

# Activate conda environment
echo -e "${YELLOW}📦 Activating conda environment...${NC}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate faiss_env

# Check if activation was successful
if [ "$CONDA_DEFAULT_ENV" != "faiss_env" ]; then
    echo -e "${RED}❌ Failed to activate conda environment${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Conda environment activated: $CONDA_DEFAULT_ENV${NC}"

# Check dependencies
echo -e "${YELLOW}🔍 Checking dependencies...${NC}"
python -c "import transformers; print(f'✅ Transformers version: {transformers.__version__}')" 2>/dev/null || {
    echo -e "${YELLOW}Installing transformers==4.51.3...${NC}"
    pip install transformers==4.51.3
}

python -c "import accelerate; print(f'✅ Accelerate version: {accelerate.__version__}')" 2>/dev/null || {
    echo -e "${YELLOW}Installing accelerate...${NC}"
    pip install accelerate
}

python -c "from peft import PeftModel; print('✅ PEFT installed')" 2>/dev/null || {
    echo -e "${YELLOW}Installing peft...${NC}"
    pip install peft
}

# Set environment variables for optimal performance
echo -e "${YELLOW}⚙️  Setting performance environment variables...${NC}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Use all 8 GPUs
export TOKENIZERS_PARALLELISM=false
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo -e "${GREEN}🎮 GPU Configuration:${NC}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -4

# Run the FAST delta indexing script
echo -e "${GREEN}🏃 Running FAST delta indexing script...${NC}"
echo -e "${YELLOW}⚡ Expected performance: 10-20 images/second${NC}"
python delta_indexing_ian_fast.py

echo -e "${GREEN}✨ Script completed!${NC}" 