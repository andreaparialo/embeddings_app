#!/usr/bin/env python3
"""
Quick script to check GPU and FAISS GPU status
"""

import torch
import faiss
import subprocess
import sys

print("=" * 60)
print("🔍 GPU and FAISS Status Check")
print("=" * 60)

# Check PyTorch CUDA
print("\n📊 PyTorch CUDA Status:")
if torch.cuda.is_available():
    print(f"✅ CUDA is available")
    print(f"   - GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"   - GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"     Memory: {props.total_memory / 1024**3:.1f} GB")
else:
    print("❌ CUDA is NOT available")

# Check FAISS GPU
print("\n📊 FAISS GPU Status:")
try:
    ngpus = faiss.get_num_gpus()
    print(f"✅ FAISS detects {ngpus} GPU(s)")
    
    # Try to create GPU resources
    res = faiss.StandardGpuResources()
    print("✅ FAISS GPU resources can be created")
    
    # Test with a small index
    print("\n🧪 Testing FAISS GPU transfer...")
    import numpy as np
    d = 64
    index = faiss.IndexFlatL2(d)
    index.add(np.random.random((100, d)).astype('float32'))
    
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    print("✅ Successfully transferred index to GPU")
    
except Exception as e:
    print(f"❌ FAISS GPU error: {e}")
    print("\n💡 This might be due to pip-installed faiss-gpu")
    print("   Try: conda install -c pytorch faiss-gpu")

# Check nvidia-smi
print("\n📊 NVIDIA-SMI Output:")
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ nvidia-smi is working")
        # Show GPU utilization
        lines = result.stdout.split('\n')
        for line in lines:
            if 'MiB' in line and '%' in line:
                print(f"   {line.strip()}")
    else:
        print("❌ nvidia-smi failed")
except Exception as e:
    print(f"❌ Could not run nvidia-smi: {e}")

# Check FAISS installation type
print("\n📦 FAISS Installation Check:")
try:
    import faiss
    if hasattr(faiss, '__version__'):
        print(f"   FAISS version: {faiss.__version__}")
    
    # Check if conda or pip
    import site
    site_packages = site.getsitepackages()
    for path in site_packages:
        if 'faiss' in path:
            if 'conda' in path.lower():
                print("✅ FAISS appears to be conda-installed (good!)")
            else:
                print("⚠️  FAISS appears to be pip-installed")
                print("   This may cause slow GPU transfers!")
                print("   Recommended: conda install -c pytorch faiss-gpu")
            break
except Exception as e:
    print(f"Could not check installation: {e}")

print("\n" + "=" * 60) 