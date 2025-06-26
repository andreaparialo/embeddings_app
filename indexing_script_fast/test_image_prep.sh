#!/bin/bash

echo "🖼️  Testing Image Preparation"
echo "============================="
echo "This will resize a subset of images to test the preparation process"
echo ""

# Activate virtual environment if available
if [[ -f "venv_app/bin/activate" ]]; then
    source venv_app/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  No virtual environment found"
fi

# Check if pictures directory exists
if [[ ! -d "pictures" ]]; then
    echo "❌ Pictures directory not found"
    exit 1
fi

# Count images
IMAGE_COUNT=$(find pictures -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | head -100 | wc -l)
echo "📁 Will test with first 100 images (found $IMAGE_COUNT)"
echo ""

# Create a test subset
if [[ ! -d "pictures_test" ]]; then
    echo "📂 Creating test subset..."
    mkdir -p pictures_test
    find pictures -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | head -100 | while read file; do
        cp "$file" pictures_test/
    done
    echo "✅ Test subset created with $(ls pictures_test | wc -l) images"
fi

# Test image preparation
echo ""
echo "🚀 Testing image preparation..."
python -c "
from prepare_images import ImagePreparer
import sys

try:
    preparer = ImagePreparer(max_size=512, quality=95, num_workers=4)
    result = preparer.prepare_images('pictures_test', 'pictures_test_prepared')
    print(f'✅ Test completed successfully! Processed {result} images')
except Exception as e:
    print(f'❌ Test failed: {e}')
    sys.exit(1)
"

if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ Image preparation test successful!"
    echo ""
    echo "📊 Results:"
    echo "   Original test images: $(ls pictures_test 2>/dev/null | wc -l)"
    echo "   Prepared test images: $(ls pictures_test_prepared 2>/dev/null | wc -l)"
    echo ""
    echo "🔍 Sample size comparison:"
    if [[ -d "pictures_test" && -d "pictures_test_prepared" ]]; then
        ORIG_SIZE=$(du -sh pictures_test | cut -f1)
        PREP_SIZE=$(du -sh pictures_test_prepared | cut -f1)
        echo "   Original size: $ORIG_SIZE"
        echo "   Prepared size: $PREP_SIZE"
    fi
    echo ""
    echo "💡 Ready to run full preparation on all images!"
    echo "   Run: python prepare_images.py"
else
    echo "❌ Image preparation test failed!"
    exit 1
fi 