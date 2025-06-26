#!/bin/bash

# Delta Indexing Pipeline - Complete automation
# Analyzes, prepares, indexes, and merges only new images

set -e  # Exit on any error

echo "🚀 DELTA INDEXING PIPELINE"
echo "=========================================="
echo "Complete incremental indexing with existing v11_o00_index_1095"
echo ""

# Configuration
PYTHON_ENV="venv_app"
LOG_FILE="delta_indexing_$(date +%Y%m%d_%H%M%S).log"

# Function to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        log "❌ Command $1 not found"
        exit 1
    fi
}

# Function to activate virtual environment
activate_venv() {
    if [ -d "$PYTHON_ENV" ]; then
        log "🔧 Activating virtual environment: $PYTHON_ENV"
        source "$PYTHON_ENV/bin/activate"
    else
        log "❌ Virtual environment not found: $PYTHON_ENV"
        exit 1
    fi
}

# Function to check GPU availability
check_gpu() {
    log "🔍 Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
        log "✅ Found $GPU_COUNT GPU(s)"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read line; do
            log "   🎯 $line"
        done
    else
        log "❌ nvidia-smi not found - GPU required for indexing"
        exit 1
    fi
}

# Function to check required files
check_requirements() {
    log "🔍 Checking requirements..."
    
    # Check existing index
    if [ ! -f "indexes/v11_o00_index_1095.faiss" ]; then
        log "❌ Existing index not found: indexes/v11_o00_index_1095.faiss"
        exit 1
    fi
    log "✅ Existing index found"
    
    # Check pictures directory
    if [ ! -d "pictures" ]; then
        log "❌ Pictures directory not found"
        exit 1
    fi
    
    PICTURE_COUNT=$(find pictures -name "*.jpg" -o -name "*.JPG" -o -name "*.jpeg" -o -name "*.JPEG" -o -name "*.png" -o -name "*.PNG" | wc -l)
    log "✅ Found $PICTURE_COUNT images in pictures/"
    
    # Check LoRA model
    if [ ! -d "loras/v11-20250620-105815/checkpoint-1095" ]; then
        log "❌ LoRA checkpoint not found: loras/v11-20250620-105815/checkpoint-1095"
        exit 1
    fi
    log "✅ LoRA checkpoint found"
    
    # Check LoRA indexing components
    if [ ! -d "lora_indexing_lightweight_20250625_100732" ]; then
        log "❌ LoRA indexing components not found"
        exit 1
    fi
    log "✅ LoRA indexing components found"
}

# Function to run delta analysis
run_analysis() {
    log "🔍 STEP 1: Delta Analysis"
    log "----------------------------------------"
    
    if python analyze_delta.py; then
        log "✅ Delta analysis completed"
        
        # Show analysis results
        if [ -f "delta_analysis.json" ]; then
            NEW_COUNT=$(python -c "import json; data=json.load(open('delta_analysis.json')); print(data['statistics']['new_images_count'])")
            TOTAL_COUNT=$(python -c "import json; data=json.load(open('delta_analysis.json')); print(data['statistics']['total_current'])")
            
            log "📊 Analysis Results:"
            log "   🆕 New images to index: $NEW_COUNT"
            log "   📁 Total current images: $TOTAL_COUNT"
            
            if [ "$NEW_COUNT" -eq 0 ]; then
                log "✅ All images already indexed - no delta processing needed!"
                exit 0
            fi
            
            PERCENTAGE=$(python -c "print(f'{($NEW_COUNT/$TOTAL_COUNT)*100:.1f}')")
            log "   📈 Processing only: $PERCENTAGE% of total images"
        fi
    else
        log "❌ Delta analysis failed"
        exit 1
    fi
}

# Function to prepare delta images
run_preparation() {
    log "🔄 STEP 2: Delta Image Preparation"
    log "----------------------------------------"
    
    if python prepare_delta_images.py; then
        log "✅ Delta image preparation completed"
        
        # Check prepared images
        if [ -d "pictures_delta_prepared" ]; then
            PREPARED_COUNT=$(find pictures_delta_prepared -name "*.jpg" | wc -l)
            log "📁 Prepared $PREPARED_COUNT delta images"
        fi
    else
        log "❌ Delta image preparation failed"
        exit 1
    fi
}

# Function to run delta indexing
run_indexing() {
    log "🚀 STEP 3: Delta Indexing"
    log "----------------------------------------"
    
    # Set GPU environment
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    export TOKENIZERS_PARALLELISM=false
    export TRANSFORMERS_VERBOSITY=error
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    
    log "🔧 GPU Environment configured"
    
    if python index_delta_only.py; then
        log "✅ Delta indexing completed"
        
        # Check delta index files
        DELTA_INDEX=$(find indexes -name "v11_delta_*.faiss" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -n "$DELTA_INDEX" ]; then
            DELTA_SIZE=$(du -h "$DELTA_INDEX" | cut -f1)
            log "📁 Delta index created: $(basename $DELTA_INDEX) ($DELTA_SIZE)"
        fi
    else
        log "❌ Delta indexing failed"
        exit 1
    fi
}

# Function to merge indexes
run_merging() {
    log "🔗 STEP 4: Index Merging"
    log "----------------------------------------"
    
    if python merge_indexes.py; then
        log "✅ Index merging completed"
        
        # Check merged index files
        MERGED_INDEX=$(find indexes -name "v11_complete_merged_*.faiss" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
        if [ -n "$MERGED_INDEX" ]; then
            MERGED_SIZE=$(du -h "$MERGED_INDEX" | cut -f1)
            log "📁 Merged index created: $(basename $MERGED_INDEX) ($MERGED_SIZE)"
            
            # Get final statistics
            MERGED_METADATA="${MERGED_INDEX%.*}_metadata.json"
            if [ -f "$MERGED_METADATA" ]; then
                TOTAL_INDEXED=$(python -c "import json; data=json.load(open('$MERGED_METADATA')); print(data['total_embeddings'])")
                log "📊 Total images in merged index: $TOTAL_INDEXED"
            fi
        fi
    else
        log "❌ Index merging failed"
        exit 1
    fi
}

# Function to cleanup intermediate files
cleanup_files() {
    log "🧹 STEP 5: Cleanup (Optional)"
    log "----------------------------------------"
    
    read -p "🗑️  Remove intermediate delta files? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "🗑️  Cleaning up intermediate files..."
        
        # Remove delta preparation directory
        if [ -d "pictures_delta_prepared" ]; then
            rm -rf pictures_delta_prepared
            log "   ✅ Removed pictures_delta_prepared/"
        fi
        
        # Remove delta analysis files
        if [ -f "delta_analysis.json" ]; then
            rm delta_analysis.json
            log "   ✅ Removed delta_analysis.json"
        fi
        
        if [ -f "delta_images_list.txt" ]; then
            rm delta_images_list.txt
            log "   ✅ Removed delta_images_list.txt"
        fi
        
        # Remove delta index files (keep merged only)
        find indexes -name "v11_delta_*" -type f -delete 2>/dev/null || true
        log "   ✅ Removed delta index files"
        
        log "✅ Cleanup completed"
    else
        log "📁 Intermediate files preserved for debugging"
    fi
}

# Function to show final summary
show_summary() {
    log "🎉 DELTA INDEXING PIPELINE COMPLETED"
    log "=========================================="
    
    END_TIME=$(date)
    DURATION=$(($(date +%s) - START_TIME))
    HOURS=$((DURATION / 3600))
    MINUTES=$(((DURATION % 3600) / 60))
    SECONDS=$((DURATION % 60))
    
    log "⏱️  Total pipeline duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
    
    # Show final index information
    MERGED_INDEX=$(find indexes -name "v11_complete_merged_*.faiss" -type f -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)
    if [ -n "$MERGED_INDEX" ]; then
        log "📁 Final merged index: $(basename $MERGED_INDEX)"
        log "📊 Index size: $(du -h $MERGED_INDEX | cut -f1)"
        
        MERGED_METADATA="${MERGED_INDEX%.*}_metadata.json"
        if [ -f "$MERGED_METADATA" ]; then
            TOTAL_IMAGES=$(python -c "import json; data=json.load(open('$MERGED_METADATA')); print(data['total_embeddings'])")
            NEW_IMAGES=$(python -c "import json; data=json.load(open('$MERGED_METADATA')); print(data['merge_info']['delta_count'])")
            log "📈 Total images indexed: $TOTAL_IMAGES"
            log "🆕 New images added: $NEW_IMAGES"
        fi
    fi
    
    log ""
    log "💡 Next Steps:"
    log "   1. Test the merged index with search queries"
    log "   2. Update applications to use: $(basename $MERGED_INDEX)"
    log "   3. Consider backing up the original v11_o00_index_1095"
    log "   4. Monitor performance improvements"
    log ""
    log "📋 Full log saved to: $LOG_FILE"
}

# Main execution
main() {
    START_TIME=$(date +%s)
    
    log "🚀 Starting Delta Indexing Pipeline"
    log "📅 Started at: $(date)"
    log ""
    
    # Pre-flight checks
    check_command python
    check_command nvidia-smi
    activate_venv
    check_gpu
    check_requirements
    
    log ""
    log "✅ All pre-flight checks passed"
    log ""
    
    # Execute pipeline steps
    run_analysis
    log ""
    
    run_preparation
    log ""
    
    run_indexing
    log ""
    
    run_merging
    log ""
    
    cleanup_files
    log ""
    
    show_summary
}

# Handle interruption
trap 'log "❌ Pipeline interrupted by user"; exit 130' INT

# Run main function
main "$@" 