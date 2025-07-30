# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a GPU-accelerated visual product search system using AI models (GME-Qwen2-VL and OpenCLIP) with FAISS for similarity search. The system processes 29,136 product images and supports image-based, SKU-based, and filtered searches with batch processing capabilities.

## Development Commands

### Environment Setup
```bash
# CRITICAL: Use conda for FAISS GPU (pip installation will fail)
conda create -n faiss_env python=3.10 -y
conda activate faiss_env
conda install -c pytorch -c nvidia faiss-gpu=1.8.0 -y
pip install -r requirements.txt
```

### Running the Application
```bash
# Start with GPU acceleration (default)
./start_gpu.sh

# Start without GPU
export USE_FAISS_GPU=false
./start.sh

# Start OpenCLIP variant
./start_openclip_gpu.sh

# Run complete indexing pipeline
./run_all_steps.sh
```

### Testing and Validation
```bash
# Test GPU performance
python test_faiss_gpu_performance.py

# Check GPU status
python check_gpu_status.py

# Test batch processing
python test_batch_fix.py
```

## Architecture

### Core Components

1. **Main Applications**
   - `app.py` - Full-featured FastAPI application
   - `app_openclip.py` - OpenCLIP variant
   - `app_minimal.py` - Minimal version

2. **Search Infrastructure**
   - `search_engine.py` - Hybrid search implementation
   - `data_loader.py` - FAISS index and CSV data management
   - `optimized_faiss_search.py` - Pre-filtering FAISS search
   - `batch_processor_optimized.py` - Batch processing with pre-filtering
   - `dual_engine.py` - Multi-checkpoint search combination

3. **Model Wrappers**
   - `gme_model.py` - GME-Qwen2-VL with LoRA fine-tuning
   - `openclip_model.py` - OpenCLIP model implementation

4. **GPU Utilities**
   - `faiss_gpu_utils.py` - Multi-GPU management
   - GPU configuration: 4x NVIDIA A100 (40GB each)

### Data Architecture

- **Primary Database**: `database_results/DB_ACTIVE.csv` (34,431 rows)
- **FAISS Indexes**: Located in `faiss_indexes/` with 512x512 standardized images
- **Index Config**: `index_config.json` defines model and index parameters
- **Embedding Dimension**: 3584 (GME-Qwen2-VL-7B)

### Frontend Structure

Located in `/static/`:
- Modular JavaScript architecture (ES6+) in `/js/modules/`
- No frontend framework dependencies
- Clean separation between UI components and business logic

## Key Implementation Details

1. **GPU Optimization**
   - Uses float16 precision for memory efficiency
   - Optimized GPU transfers (2-5 seconds vs minutes)
   - Pre-filtering reduces search space before FAISS
   - Batch processing achieves 100+ images/second

2. **Search Types**
   - Image similarity search (AI embeddings)
   - SKU search (exact and partial matching)
   - Filter-based search with attribute pre-filtering
   - Batch processing via Excel upload

3. **Error Handling**
   - GPU fallback to CPU when unavailable
   - Graceful handling of missing images/data
   - Comprehensive logging throughout

## Important Notes

- **FAISS Installation**: Must use conda, not pip, for GPU support
- **GPU Memory**: System requires ~20GB GPU memory for full operation
- **Index Updates**: Use delta indexing for incremental updates
- **Performance**: Pre-filtering is critical for sub-second response times with filters