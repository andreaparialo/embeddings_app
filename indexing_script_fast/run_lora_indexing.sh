#!/bin/bash

echo "🚀 Starting LoRA Indexing with v11 checkpoint-1095"
echo "=================================================="

# Set environment variables for optimal performance
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment active: $VIRTUAL_ENV"
else
    echo "⚠️  No virtual environment detected"
    echo "   Consider activating venv_app: source venv_app/bin/activate"
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    echo "🔍 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    echo ""
else
    echo "⚠️  nvidia-smi not found - GPU status unknown"
fi

# Check required files
echo "🔍 Checking required files..."
if [[ ! -d "gme-Qwen2-VL-7B-Instruct" ]]; then
    echo "❌ Base model not found: gme-Qwen2-VL-7B-Instruct"
    exit 1
fi

if [[ ! -d "loras/v11-20250620-105815/checkpoint-1095" ]]; then
    echo "❌ LoRA checkpoint not found: loras/v11-20250620-105815/checkpoint-1095"
    exit 1
fi

if [[ ! -d "pictures" ]]; then
    echo "❌ Pictures directory not found: pictures"
    exit 1
fi

echo "✅ All required files found"
echo ""

# Count images
IMAGE_COUNT=$(find pictures -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)
echo "📁 Found $IMAGE_COUNT images to process"
echo ""

# Run the indexing
echo "🚀 Starting LoRA indexing..."
python lora_adapted_indexing.py

# Check if indexing completed successfully
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ LoRA indexing completed successfully!"
    echo "📁 Check the 'indexes' directory for the generated index files"
    echo "🎯 Index name: lora_v11_pictures"
    echo ""
    echo "Files created:"
    ls -la indexes/lora_v11_pictures* 2>/dev/null || echo "   No index files found"
else
    echo ""
    echo "❌ LoRA indexing failed!"
    echo "Check the error messages above for troubleshooting"
fi 