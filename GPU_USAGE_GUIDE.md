# GPU Usage Guide

## Overview
The application now supports GPU selection to avoid overloading GPUs that are already in use (e.g., for indexing).

## Quick Start

### Using the Wrapper Script (Recommended)
```bash
# Show GPU status and run on default GPU (1)
./run_with_gpu.sh

# Run on specific GPU
./run_with_gpu.sh 2          # Use GPU 2
./run_with_gpu.sh 3 1095     # Use GPU 3 with checkpoint 1095
./run_with_gpu.sh cpu        # Force CPU mode
```

### Direct Python Command
```bash
# Activate environment first
source ~/miniconda3/etc/profile.d/conda.sh && conda activate faiss_env

# Run with specific GPU
python run.py --gpu 2                    # Use GPU 2
python run.py --gpu 3 --checkpoint 1095  # Use GPU 3 with checkpoint 1095
python run.py --no-gpu                   # Force CPU mode
```

### Using Environment Variables
```bash
# Set GPU before running
export CUDA_VISIBLE_DEVICES=2
python run.py

# Use multiple GPUs
export CUDA_VISIBLE_DEVICES=2,3
python run.py
```

## GPU Selection Strategy

Based on your current GPU status:
- **GPU 0**: BUSY (15GB/16GB) - Used for indexing, avoid!
- **GPU 1-7**: Available (4MB/16GB) - Safe to use

### Recommended GPUs:
1. **GPU 1** - Default choice (automatically selected if not specified)
2. **GPU 2-7** - All available for use
3. **CPU Mode** - If all GPUs are busy or for testing

## Features

### 1. Automatic GPU Status Check
The wrapper script shows real-time GPU usage:
```
Available GPUs:
  GPU 0: 15258MiB/16384MiB (93%) - BUSY
  GPU 1: 4MiB/16384MiB (0%) - AVAILABLE
  GPU 2: 4MiB/16384MiB (0%) - AVAILABLE
  ...
```

### 2. Smart Defaults
- Defaults to GPU 1 (not GPU 0) to avoid conflicts
- 5-second timeout allows quick start with default

### 3. Checkpoint Selection
Specify which model checkpoint to use:
- `680` - Earlier checkpoint
- `1095` - Latest checkpoint (default)
- `1020` - Alternative checkpoint

## Examples

### Running While Indexing
```bash
# GPU 0 is busy with indexing, use GPU 2
./run_with_gpu.sh 2

# Or use environment variable
CUDA_VISIBLE_DEVICES=2 python run.py
```

### Testing Different Checkpoints
```bash
# Test checkpoint 680 on GPU 3
./run_with_gpu.sh 3 680

# Test checkpoint 1095 on GPU 4
./run_with_gpu.sh 4 1095
```

### Multi-GPU Setup
```bash
# Use GPUs 2 and 3 for FAISS
export CUDA_VISIBLE_DEVICES=2,3
python run.py
```

## Troubleshooting

### Out of Memory Error
If you get OOM errors, try:
1. Use a different GPU: `./run_with_gpu.sh 3`
2. Use CPU mode: `./run_with_gpu.sh cpu`
3. Check GPU status: `nvidia-smi`

### Slow Performance
- FAISS GPU operations might be slow with pip-installed faiss-gpu
- Consider using conda-installed version for better performance
- CPU mode might be faster for small queries

### Model Loading Issues
- Each GPU needs ~15GB for the full model
- Ensure the selected GPU has enough free memory
- Use `nvidia-smi` to check available memory

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `CUDA_VISIBLE_DEVICES` | GPU(s) to use | `2` or `2,3` |
| `DEFAULT_CHECKPOINT` | Model checkpoint | `1095` |
| `FORCE_CPU_FAISS` | Force CPU mode | `true` |
| `USE_FAISS_GPU` | Enable GPU FAISS | `false` to disable 