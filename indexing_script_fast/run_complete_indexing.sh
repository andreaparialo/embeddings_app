#!/bin/bash

echo "🚀 Complete LoRA Indexing Pipeline"
echo "=================================="
echo "Step 1: Prepare images (resize to 512px)"
echo "Step 2: Index with LoRA v11 checkpoint-1095"
echo ""

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
    echo ""
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

# Count original images
ORIGINAL_COUNT=$(find pictures -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" -o -iname "*.gif" -o -iname "*.bmp" -o -iname "*.tiff" -o -iname "*.webp" \) | wc -l)
echo "📁 Found $ORIGINAL_COUNT original images"
echo ""

# Step 1: Image Preparation
echo "🖼️  STEP 1: IMAGE PREPARATION"
echo "============================="
echo "Resizing images to 512px max dimension..."
echo ""

python prepare_images.py

# Check if preparation was successful
if [[ $? -ne 0 ]]; then
    echo ""
    echo "❌ Image preparation failed!"
    exit 1
fi

# Count prepared images
if [[ -d "pictures_prepared" ]]; then
    PREPARED_COUNT=$(find pictures_prepared -type f -name "*.jpg" | wc -l)
    echo ""
    echo "✅ Image preparation completed!"
    echo "   📊 Original images: $ORIGINAL_COUNT"
    echo "   📊 Prepared images: $PREPARED_COUNT"
    echo ""
else
    echo "❌ Prepared images directory not found!"
    exit 1
fi

# Step 2: LoRA Indexing
echo "🔥 STEP 2: LoRA INDEXING"
echo "========================"
echo "Indexing prepared images with LoRA model..."
echo ""

python lora_optimized_indexing.py

# Check if indexing was successful
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ Complete indexing pipeline completed successfully!"
    echo ""
    echo "📊 Final Results:"
    echo "   📁 Original images: $ORIGINAL_COUNT"
    echo "   📁 Prepared images: $PREPARED_COUNT"
    echo "   📂 Prepared images location: pictures_prepared/"
    echo "   🎯 Index name: lora_v11_prepared"
    echo ""
    echo "📁 Generated files:"
    echo "   indexes/lora_v11_prepared.faiss"
    echo "   indexes/lora_v11_prepared_metadata.json"
    echo "   indexes/lora_v11_prepared_embeddings.npy"
    echo ""
    echo "🎉 Ready for LoRA-enhanced image search!"
    echo ""
    echo "💡 Benefits achieved:"
    echo "   • Faster processing due to smaller images"
    echo "   • Lower memory usage"
    echo "   • Multi-GPU utilization"
    echo "   • Worker-based parallel processing"
    
    # Show final index files
    if [[ -d "indexes" ]]; then
        echo ""
        echo "📋 Index files created:"
        ls -lah indexes/lora_v11_prepared* 2>/dev/null || echo "   No index files found"
    fi
    
else
    echo ""
    echo "❌ LoRA indexing failed!"
    echo "Check the error messages above for troubleshooting"
    exit 1
fi 