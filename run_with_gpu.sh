#!/bin/bash

# Wrapper script to run the application on specific GPUs
# Helps avoid overloading GPUs that are already in use

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Default GPU (avoiding GPU 0 which is often in use)
DEFAULT_GPU=1

# Parse command line arguments
GPU_ID=$1
CHECKPOINT=$2

# Show usage if no arguments
if [ -z "$GPU_ID" ]; then
    echo -e "${BLUE}Usage: ./run_with_gpu.sh [GPU_ID] [CHECKPOINT]${NC}"
    echo -e "${BLUE}Examples:${NC}"
    echo -e "  ${GREEN}./run_with_gpu.sh 1${NC}        # Run on GPU 1"
    echo -e "  ${GREEN}./run_with_gpu.sh 2 1095${NC}   # Run on GPU 2 with checkpoint 1095"
    echo -e "  ${GREEN}./run_with_gpu.sh cpu${NC}      # Run in CPU mode (no GPU)"
    echo ""
    echo -e "${YELLOW}Available GPUs:${NC}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader | while read line; do
        gpu_id=$(echo $line | cut -d',' -f1 | tr -d ' ')
        gpu_name=$(echo $line | cut -d',' -f2 | tr -d ' ')
        mem_used=$(echo $line | cut -d',' -f3 | tr -d ' ')
        mem_total=$(echo $line | cut -d',' -f4 | tr -d ' ')
        
        # Calculate percentage
        mem_used_mb=$(echo $mem_used | sed 's/MiB//')
        mem_total_mb=$(echo $mem_total | sed 's/MiB//')
        if [ "$mem_total_mb" -gt 0 ]; then
            mem_percent=$((mem_used_mb * 100 / mem_total_mb))
        else
            mem_percent=0
        fi
        
        # Color code based on usage
        if [ $mem_percent -gt 80 ]; then
            color=$RED
            status="BUSY"
        elif [ $mem_percent -gt 50 ]; then
            color=$YELLOW
            status="IN USE"
        else
            color=$GREEN
            status="AVAILABLE"
        fi
        
        echo -e "  GPU $gpu_id: ${color}$mem_used/$mem_total ($mem_percent%) - $status${NC}"
    done
    echo ""
    echo -e "${YELLOW}Using default GPU $DEFAULT_GPU. Press Ctrl+C to cancel or Enter to continue...${NC}"
    read -t 5
    GPU_ID=$DEFAULT_GPU
fi

# Activate conda environment
echo -e "${YELLOW}📦 Activating conda environment...${NC}"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate faiss_env

if [ "$CONDA_DEFAULT_ENV" != "faiss_env" ]; then
    echo -e "${RED}❌ Failed to activate conda environment${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Conda environment activated: $CONDA_DEFAULT_ENV${NC}"

# Run the application
if [ "$GPU_ID" = "cpu" ]; then
    echo -e "${BLUE}🖥️  Starting in CPU mode...${NC}"
    python run.py --no-gpu
elif [ -n "$CHECKPOINT" ]; then
    echo -e "${BLUE}🎮 Starting on GPU $GPU_ID with checkpoint $CHECKPOINT...${NC}"
    python run.py --gpu $GPU_ID --checkpoint $CHECKPOINT
else
    echo -e "${BLUE}🎮 Starting on GPU $GPU_ID...${NC}"
    python run.py --gpu $GPU_ID
fi 