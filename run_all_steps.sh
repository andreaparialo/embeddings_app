#!/bin/bash
# Run all steps to create db_pictures_512 and index it

echo "🚀 Starting Complete DB Pictures Pipeline"
echo "========================================"

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate faiss_env

# Step 1: Create db_pictures folder
echo ""
echo "1️⃣ Creating db_pictures folder..."
python3 create_db_pictures_folder.py
if [ $? -ne 0 ]; then
    echo "❌ Failed to create db_pictures folder"
    exit 1
fi

# Step 2: Resize images to 512x512
echo ""
echo "2️⃣ Resizing images to 512x512..."
python3 resize_db_pictures_512.py
if [ $? -ne 0 ]; then
    echo "❌ Failed to resize images"
    exit 1
fi

# Step 3: Index the resized images
echo ""
echo "3️⃣ Indexing resized images..."
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=0,1,2,3

python3 indexing_script_fast/lora_max_performance_indexing_custom.py \
    loras/v11-20250620-105815/checkpoint-1095 \
    db_pictures_512 \
    v11_1095_db_pictures_512 \
    indexes

if [ $? -ne 0 ]; then
    echo "❌ Failed to index images"
    exit 1
fi

# Step 4: Create index configuration
echo ""
echo "4️⃣ Creating index configuration..."
python3 add_multi_index_support.py

echo ""
echo "✅ All steps completed successfully!"
echo ""
echo "📊 Summary:"
echo "  - Created db_pictures folder with database-referenced images"
echo "  - Resized all images to 512x512 with white padding"
echo "  - Indexed resized images with GME + LoRA v11-1095"
echo "  - Created index configuration for multi-index support"
echo ""
echo "📋 Next steps:"
echo "  - Update app.py and data_loader.py to support multiple indexes"
echo "  - Test the new 512x512 index performance"
echo "  - Enjoy faster, more consistent image search!" 