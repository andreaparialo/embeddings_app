#!/bin/bash

# Optimized GME-Qwen GPU Indexing Script
# Uses native batch processing for maximum speed

echo "🚀 OPTIMIZED GME-Qwen GPU Batch Indexing"
echo "======================================="

# Activate virtual environment
if [ -d "venv_app" ]; then
    echo "📦 Activating virtual environment..."
    source venv_app/bin/activate
fi

# Set optimal environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Remove expandable_segments due to PyTorch bug
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

# Check GPUs
echo "🖥️  Checking GPU status..."
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Get parameters
CHECKPOINT=${1:-1095}
BATCH_SIZE=${2:-32}

echo ""
echo "Configuration:"
echo "  LoRA Checkpoint: $CHECKPOINT"
echo "  Batch Size: $BATCH_SIZE"
echo ""

# Run the optimized indexing
echo "⚡ Starting optimized batch processing..."
python3 gme_optimized_batch_indexing.py \
    --pictures-dir pictures \
    --csv-path database_results/final_with_aws_shapes_20250625_155822.csv \
    --output-dir indexes \
    --checkpoint "$CHECKPOINT" \
    --batch-size "$BATCH_SIZE"

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Optimized index created!"
    echo "📁 Check the indexes/ directory for your new files"
else
    echo ""
    echo "❌ Failed! Check the logs above for errors"
    exit 1
fi 