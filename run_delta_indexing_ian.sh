#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting Smart Delta Indexing for IAN/resized${NC}"
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

# Check if transformers is installed
echo -e "${YELLOW}🔍 Checking dependencies...${NC}"
python -c "import transformers; print(f'✅ Transformers version: {transformers.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ transformers module not found${NC}"
    echo -e "${YELLOW}Installing transformers==4.51.3...${NC}"
    pip install transformers==4.51.3
fi

# Check if accelerate is installed
python -c "import accelerate; print(f'✅ Accelerate version: {accelerate.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ accelerate module not found${NC}"
    echo -e "${YELLOW}Installing accelerate...${NC}"
    pip install accelerate
fi

# Check other dependencies
echo -e "${YELLOW}🔍 Checking other key dependencies...${NC}"
python -c "import torch; print(f'✅ PyTorch version: {torch.__version__}')"
python -c "import faiss; print(f'✅ FAISS version: {faiss.__version__ if hasattr(faiss, \"__version__\") else \"unknown\"}')"
python -c "from peft import PeftModel; print('✅ PEFT installed')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}❌ peft module not found${NC}"
    echo -e "${YELLOW}Installing peft...${NC}"
    pip install peft
fi

# Set environment variables for better GPU utilization
echo -e "${YELLOW}⚙️  Setting GPU environment variables...${NC}"
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  # Use all 8 GPUs
export TOKENIZERS_PARALLELISM=false

echo -e "${GREEN}🎮 GPU Configuration:${NC}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# Run the delta indexing script
echo -e "${GREEN}🏃 Running delta indexing script...${NC}"
python delta_indexing_ian_smart.py

echo -e "${GREEN}✨ Script completed!${NC}" 