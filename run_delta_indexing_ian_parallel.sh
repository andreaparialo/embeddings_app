#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Starting PARALLEL Delta Indexing for IAN/resized (8 GPUs)${NC}"
echo -e "${GREEN}=======================================================${NC}"

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

# Check critical dependencies
echo -e "${YELLOW}🔍 Checking dependencies...${NC}"
python -c "import transformers; print(f'✅ Transformers: {transformers.__version__}')" || exit 1
python -c "import torch; print(f'✅ PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" || exit 1
python -c "import faiss; print('✅ FAISS installed')" || exit 1
python -c "from peft import PeftModel; print('✅ PEFT installed')" || exit 1

# Show GPU status
echo -e "${GREEN}🎮 GPU Status:${NC}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv

echo -e "${YELLOW}⚡ This will use all 8 GPUs in parallel for maximum speed!${NC}"
echo -e "${YELLOW}⚡ Expected performance: 10-20 images/second total across all GPUs${NC}"

# Run the parallel delta indexing script
echo -e "${GREEN}🏃 Running parallel delta indexing script...${NC}"
python delta_indexing_ian_parallel.py

echo -e "${GREEN}✨ Script completed!${NC}" 