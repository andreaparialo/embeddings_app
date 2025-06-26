# Indexing Scripts Collection

This folder contains all indexing-related scripts extracted from the SPEEDINGTHEPROCESS project. The scripts are organized by their functionality and indexing approach.

## 📋 Script Categories

### 1. Delta Indexing Scripts
Scripts for incremental indexing of new images without re-indexing existing ones:

- **`analyze_delta.py`** - Analyzes existing indexes vs current images to find delta
- **`analyze_delta_fast.py`** - Fast version of delta analysis
- **`prepare_delta_images.py`** - Prepares only new images for indexing (resize to 512px)
- **`index_delta_only.py`** - Indexes only the delta images using LoRA
- **`merge_indexes.py`** - Merges delta index with existing index
- **`create_delta_index.py`** - Alternative delta index creation script
- **`prepare_new_images_for_indexing.py`** - Prepares new images with symlinks

### 2. LoRA Indexing Scripts
Scripts using LoRA (Low-Rank Adaptation) models for enhanced indexing:

- **`lora_max_performance_indexing.py`** - High-performance LoRA indexing with adaptive batching
- **`lora_max_performance_indexing_custom.py`** - Customized version with profiling
- **`lora_adapted_indexing.py`** - Simple LoRA indexing implementation
- **`lora_optimized_indexing.py`** - Multi-GPU optimized LoRA indexing with workers
- **`lora_incremental_indexing.py`** - Incremental LoRA indexing helper

### 3. GME Model Indexing Scripts
Scripts using GME-Qwen2-VL model for indexing:

- **`gme_optimized_batch_indexing.py`** - Optimized batch processing for GME
- **`gme_fast_gpu_indexing.py`** - Fast multi-GPU GME indexing
- **`gme_model.py`** - GME model wrapper and utilities

### 4. OpenCLIP Indexing
- **`openclip_create_embeddings.py`** - Creates embeddings using OpenCLIP model

### 5. Image Preparation
- **`prepare_images.py`** - Resizes images to 512px for faster processing

### 6. Shell Scripts
Automation scripts for running complete pipelines:

- **`run_delta_indexing.sh`** - Complete delta indexing pipeline
- **`run_lora_indexing.sh`** - Run LoRA indexing
- **`run_complete_indexing.sh`** - Complete indexing pipeline (prep + index)
- **`run_optimized_indexing.sh`** - Run optimized indexing
- **`quick_index.sh`** - Quick indexing script
- **`test_image_prep.sh`** - Test image preparation with 100 images
- **`test_delta_analysis.sh`** - Test delta analysis

### 7. Supporting Utilities
- **`lora_similarity_engine.py`** - LoRA model engine for similarity computation
- **`lora_model_utils.py`** - Utilities for LoRA model discovery and loading
- **`data_loader.py`** - Data loading utilities for batch processing

### 8. Documentation
- **`DELTA_INDEXING_ANALYSIS.md`** - Detailed analysis of delta indexing approach
- **`COMPLETE_INDEXING_PIPELINE.md`** - Complete pipeline documentation
- **`WORKER_ANALYSIS.md`** - Analysis of worker patterns and performance

## 🚀 Quick Start Guide

### For Delta Indexing (Incremental Updates)
```bash
# 1. Analyze what's new
python analyze_delta.py

# 2. Prepare delta images
python prepare_delta_images.py

# 3. Index delta
python index_delta_only.py

# 4. Merge with existing
python merge_indexes.py
```

### For Full LoRA Indexing
```bash
# Option 1: High performance
python lora_max_performance_indexing.py

# Option 2: Multi-GPU optimized
python lora_optimized_indexing.py

# Option 3: Run complete pipeline
./run_complete_indexing.sh
```

### For GME Indexing
```bash
# Fast GPU indexing
python gme_fast_gpu_indexing.py

# Optimized batch processing
python gme_optimized_batch_indexing.py
```

## 📊 Performance Comparison

| Method | GPUs Used | Est. Time (29K images) | Best For |
|--------|-----------|----------------------|----------|
| Delta Indexing | 1-4 | 20-30 min (2K new) | Incremental updates |
| LoRA Optimized | 4 | 2-3 hours | Full re-indexing |
| GME Fast GPU | 4 | 2-3 hours | Alternative model |
| Sequential | 1 | 13+ hours | Not recommended |

## 🔧 Environment Requirements

- **Python**: 3.8+
- **GPU**: CUDA-capable (4x A100 GPUs available)
- **Memory**: 16GB+ RAM
- **Dependencies**: See individual script requirements

## 💡 Recommendations

1. **For incremental updates**: Use delta indexing approach (85% time savings)
2. **For full indexing**: Use `lora_optimized_indexing.py` with multi-GPU
3. **For testing**: Start with test scripts before full runs
4. **Image prep**: Always resize to 512px for optimal performance

## ⚠️ Important Notes

- The `old_app/` folder is excluded as it's already organized
- All scripts assume the base GME model is downloaded
- LoRA checkpoints are in `loras/` directory
- Indexes are saved to `indexes/` directory 