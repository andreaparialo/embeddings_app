#!/bin/bash

# Quick GME-Qwen GPU Indexing Script
# Optimized for 4x A100 GPUs

echo "🚀 Quick GME-Qwen GPU Indexing for A100s"
echo "========================================"

# Activate virtual environment if it exists
if [ -d "venv_app" ]; then
    echo "📦 Activating virtual environment..."
    source venv_app/bin/activate
fi

# Set optimal environment variables for A100s
export CUDA_VISIBLE_DEVICES=0,1,2,3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export CUDA_LAUNCH_BLOCKING=0

# Disable tokenizers parallelism warning (common with multi-GPU)
export TOKENIZERS_PARALLELISM=false

# Disable transformers warnings
export TRANSFORMERS_VERBOSITY=error

# Check if we have the required files
if [ ! -d "pictures" ]; then
    echo "❌ Error: pictures/ directory not found!"
    exit 1
fi

if [ ! -f "database_results/final_with_aws_shapes_20250625_155822.csv" ]; then
    echo "❌ Error: CSV database file not found!"
    exit 1
fi

# Check available checkpoints
echo "📁 Available GME LoRA checkpoints:"
if [ -d "loras/v11-20250620-105815" ]; then
    if [ -d "loras/v11-20250620-105815/checkpoint-680" ]; then
        echo "   ✅ 680 (fast)"
    fi
    if [ -d "loras/v11-20250620-105815/checkpoint-1020" ]; then
        echo "   ✅ 1020 (balanced)"
    fi
    if [ -d "loras/v11-20250620-105815/checkpoint-1095" ]; then
        echo "   ✅ 1095 (highest quality - DEFAULT)"
    fi
fi

# Get checkpoint choice (default to 1095 for best quality)
CHECKPOINT=${1:-1095}
echo "🎯 Using LoRA checkpoint: $CHECKPOINT"

# Run the indexing with optimal settings
echo "⚡ Starting GPU-accelerated indexing..."
echo "💡 This will use all 4 A100 GPUs with optimized batch sizes"

python3 gme_fast_gpu_indexing.py \
    --pictures-dir pictures \
    --csv-path database_results/final_with_aws_shapes_20250625_155822.csv \
    --output-dir indexes \
    --checkpoint "$CHECKPOINT" \
    --batch-size 8

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Your GPU-accelerated index is ready!"
    echo "📁 Check the indexes/ directory for the new files"
    echo "🔍 You can now use this index for ultra-fast search"
else
    echo ""
    echo "❌ FAILED! Check the logs above for errors"
    exit 1
fi 