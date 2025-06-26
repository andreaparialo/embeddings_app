#!/usr/bin/env python3
"""Optimized GPU speed test for OpenCLIP with larger batches"""

import torch
import time
from openclip_model import openclip_model
import glob
import numpy as np

print("=" * 80)
print("OpenCLIP Optimized GPU Speed Test")
print("=" * 80)

# Check device
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {openclip_model.device}")

# Load model
print("\nLoading OpenCLIP model...")
start = time.time()
success = openclip_model.load_model("epoch_008_model.pth")
load_time = time.time() - start
print(f"Model loaded in {load_time:.2f} seconds")
print(f"Model device: {next(openclip_model.model.parameters()).device}")

# Get test images (more for better testing)
test_images = glob.glob("pictures/*.jpg")[:1000] + glob.glob("pictures/*.JPG")[:1000]
test_images = list(set(test_images))[:1000]  # Ensure unique images, max 1000
print(f"\nTesting with {len(test_images)} images")

# Test different batch sizes - including much larger ones for GH200
batch_sizes = [32, 64, 128, 256, 512, 1024]

# Warm up GPU with a few runs
print("\nWarming up GPU...")
for _ in range(3):
    _ = openclip_model.encode_images(test_images[:256], batch_size=256)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

print("\nRunning speed tests...")
for batch_size in batch_sizes:
    if batch_size > len(test_images):
        continue
        
    print(f"\nBatch size: {batch_size}")
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Actual test with synchronization
    start = time.time()
    embeddings = openclip_model.encode_images(test_images, batch_size=batch_size)
    
    # Ensure GPU operations complete
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    encode_time = time.time() - start
    
    if embeddings is not None:
        images_per_sec = len(test_images) / encode_time
        print(f"  Time: {encode_time:.2f}s")
        print(f"  Speed: {images_per_sec:.1f} images/sec")
        print(f"  Estimated time for 28,671 images: {28671 / images_per_sec:.1f} seconds ({28671 / images_per_sec / 60:.1f} minutes)")
        
        # Calculate throughput
        mb_per_image = 0.5  # Approximate size
        throughput = images_per_sec * mb_per_image
        print(f"  Throughput: ~{throughput:.1f} MB/s")
    else:
        print("  Failed to encode images")

# Memory info
if torch.cuda.is_available():
    print(f"\nGPU Memory after tests:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"  Max available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\nRecommendations:")
print(f"  - Best batch size for your GPU: {512 if torch.cuda.is_available() else 32}")
print(f"  - With 94GB GPU memory, you could process even larger batches")
print(f"  - Consider batch size 512-1024 for optimal performance")

print("\n" + "=" * 80) 