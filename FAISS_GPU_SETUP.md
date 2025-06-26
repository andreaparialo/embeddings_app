# FAISS GPU Setup and Optimization Guide

## Overview
This guide helps you set up and optimize FAISS GPU acceleration for the search engine, addressing the common issue of slow GPU transfers.

## Key Features Implemented

### 1. **GPU Auto-Detection and Fallback**
- Automatically detects available GPUs (your 4 A100s)
- Falls back to CPU if GPU is not available or fails
- Logs detailed GPU information on startup

### 2. **Optimized GPU Transfer**
- CUDA cache persistence for faster subsequent loads
- GPU pre-warming to trigger JIT compilation early
- Float16 precision for 2x memory efficiency
- Disabled temporary memory for faster startup
- Multi-GPU support for distributing load across all 4 A100s

### 3. **Performance Monitoring**
- Detailed timing logs for GPU transfer
- Memory usage tracking
- Benchmark script to compare CPU vs GPU performance

## Usage

### Environment Variables

```bash
# Enable/disable GPU (default: true)
export USE_FAISS_GPU=true

# Start the app
./start.sh
```

### Testing GPU Performance

Run the benchmark script to see GPU vs CPU performance:

```bash
cd old_app
python test_faiss_gpu_performance.py
```

Expected output:
- GPU should be 5-20x faster than CPU for large batches
- Initial GPU transfer might take 10-30 seconds (only on first run)
- Subsequent runs should be much faster due to CUDA cache

## Troubleshooting Slow GPU Transfers

### 1. **Pip vs Conda Installation Issue**

The most common cause of slow GPU transfers is using pip-installed faiss-gpu. 

**Check your installation:**
```bash
# The app will warn you on startup if pip-installed
# Look for: "⚠️ FAISS appears to be pip-installed"
```

**Solution - Reinstall with Conda:**
```bash
# Remove pip version
pip uninstall faiss-gpu faiss-cpu

# Install conda version (for CUDA 12.x)
conda install -c pytorch -c nvidia faiss-gpu=1.8.0

# Or for CUDA 11.x
conda install -c pytorch faiss-gpu=1.7.4 cudatoolkit=11.8
```

### 2. **First-Run JIT Compilation**

The first GPU transfer triggers CUDA kernel compilation, which can take several minutes.

**Solutions implemented:**
- GPU pre-warming with dummy operations
- CUDA cache persistence between runs
- The cache is stored in `~/.nv/ComputeCache`

### 3. **Memory Configuration**

For faster startup, temporary memory is disabled by default. If you need it for specific operations:

```python
# In faiss_gpu_utils.py, modify:
self.create_gpu_resources(temp_memory_mb=512)  # Add 512MB temp memory
```

### 4. **Multi-GPU Usage**

With 4 A100s, the system will automatically distribute the index across all GPUs if available.

To force single GPU mode:
```python
# In data_loader.py, change:
if faiss.get_num_gpus() >= 4:
    self.index = faiss_gpu_manager.transfer_index_to_all_gpus(self.cpu_index)
# To:
self.index = faiss_gpu_manager.transfer_index_to_gpu(self.cpu_index)
```

## Performance Tips

1. **Batch Queries**: GPU performs best with batched queries. The batch search feature automatically leverages this.

2. **Float16 Precision**: Enabled by default for 2x memory savings with minimal accuracy loss.

3. **Index Type**: Some index types transfer faster than others. IVF indexes generally transfer faster than flat indexes.

4. **Persistent Process**: Keep the app running to avoid repeated GPU transfers. The index stays on GPU between requests.

## Monitoring

The app provides detailed logs:
- `🎮 Found 4 GPU(s) available` - GPU detection
- `🔄 Transferring to GPU 0...` - Transfer progress  
- `✅ GPU transfer completed in X.XXs` - Transfer timing
- `GPU 0 Memory: X.XGB allocated` - Memory usage

## Fallback to CPU

If GPU fails or is too slow, disable it:
```bash
export USE_FAISS_GPU=false
./start.sh
```

The app will use CPU-only mode, which is still quite fast for smaller datasets.

## Expected Performance

With proper setup on your 4 A100s:
- Initial GPU transfer: 10-30 seconds (first run only)
- Subsequent transfers: 2-5 seconds (with CUDA cache)
- Search speedup: 10-50x faster than CPU
- Batch processing: Can handle thousands of queries per second

## Additional Resources

- [FAISS GPU Documentation](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU)
- [Troubleshooting Slow Transfers](https://github.com/facebookresearch/faiss/issues/2710)
- [Multi-GPU Setup](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU#multi-gpu-search) 