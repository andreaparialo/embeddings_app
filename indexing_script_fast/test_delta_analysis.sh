#!/bin/bash

# Quick test script - Just run delta analysis to see what we're dealing with

echo "🔍 DELTA ANALYSIS TEST"
echo "======================"
echo "Quick check to see how many new images need indexing"
echo ""

# Activate virtual environment
if [ -d "venv_app" ]; then
    echo "🔧 Activating virtual environment..."
    source venv_app/bin/activate
else
    echo "❌ Virtual environment not found: venv_app"
    exit 1
fi

# Check requirements
echo "🔍 Checking requirements..."

if [ ! -f "indexes/v11_o00_index_1095.faiss" ]; then
    echo "❌ Existing index not found: indexes/v11_o00_index_1095.faiss"
    exit 1
fi

if [ ! -d "pictures" ]; then
    echo "❌ Pictures directory not found"
    exit 1
fi

echo "✅ Requirements check passed"
echo ""

# Run analysis
echo "🔍 Running delta analysis..."
python analyze_delta.py

echo ""
echo "📋 Analysis complete!"
echo ""

# Show quick summary if analysis file exists
if [ -f "delta_analysis.json" ]; then
    echo "📊 QUICK SUMMARY:"
    echo "=================="
    
    NEW_COUNT=$(python -c "import json; data=json.load(open('delta_analysis.json')); print(data['statistics']['new_images_count'])")
    TOTAL_COUNT=$(python -c "import json; data=json.load(open('delta_analysis.json')); print(data['statistics']['total_current'])")
    EXISTING_COUNT=$(python -c "import json; data=json.load(open('delta_analysis.json')); print(data['statistics']['common_images_count'])")
    
    echo "🆕 New images to index: $NEW_COUNT"
    echo "🔄 Already indexed: $EXISTING_COUNT"
    echo "📁 Total current images: $TOTAL_COUNT"
    
    if [ "$NEW_COUNT" -gt 0 ]; then
        PERCENTAGE=$(python -c "print(f'{($NEW_COUNT/$TOTAL_COUNT)*100:.1f}')")
        TIME_SAVINGS=$(python -c "print(f'{100-($NEW_COUNT/$TOTAL_COUNT)*100:.1f}')")
        
        echo ""
        echo "💡 Delta Indexing Benefits:"
        echo "   📈 Only need to process: $PERCENTAGE% of images"
        echo "   ⏱️  Estimated time savings: $TIME_SAVINGS%"
        echo "   🚀 Ready for: ./run_delta_indexing.sh"
    else
        echo ""
        echo "✅ All images already indexed - no delta processing needed!"
    fi
fi 