#!/bin/bash
# Start script for OpenCLIP app with GPU support

# Setup environment variables for optimal performance
export CUDA_VISIBLE_DEVICES=0,1,2,3  # Use all 4 A100 GPUs
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TORCH_CUDA_ARCH_LIST="8.0"  # A100 architecture

# FAISS/CUDA cache persistence
export CUDA_CACHE_DISABLE=0
export CUDA_CACHE_MAXSIZE=2147483648  # 2GB
export CUDA_CACHE_PATH=/tmp/cuda_cache
mkdir -p $CUDA_CACHE_PATH

# Suppress warnings
export TF_CPP_MIN_LOG_LEVEL=2
export PYTHONWARNINGS="ignore"

# MKL/OpenMP settings
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1

# Activate conda environment
echo "Activating conda environment faiss_env..."
# Use miniconda3 instead of anaconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate faiss_env

# Change to the app directory (if not already there)
if [ "$PWD" != "/home/ubuntu/SPEEDINGTHEPROCESS/old_app" ]; then
    cd /home/ubuntu/SPEEDINGTHEPROCESS/old_app
fi

# Check if port is already in use
PORT=8001
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null ; then
    echo "Port $PORT is already in use. Killing existing process..."
    kill $(lsof -Pi :$PORT -sTCP:LISTEN -t)
    sleep 2
fi

# Start the OpenCLIP app with GPU support
echo "Starting OpenCLIP app with GPU support on port $PORT..."
python -m uvicorn app_openclip:app --host 0.0.0.0 --port $PORT --workers 1 --log-level info 