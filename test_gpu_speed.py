#!/usr/bin/env python3
"""Quick GPU speed test for OpenCLIP"""

import torch
import time
from openclip_model import openclip_model
import glob
import numpy as np

print("=" * 80)
print("OpenCLIP GPU Speed Test")
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

# Get a small batch of test images
test_images = glob.glob("pictures/*.jpg")[:100] + glob.glob("pictures/*.JPG")[:100]
test_images = test_images[:100]  # Ensure we have at most 100 images
print(f"\nTesting with {len(test_images)} images")

# Test different batch sizes
batch_sizes = [16, 32, 64, 128, 256]
for batch_size in batch_sizes:
    print(f"\nBatch size: {batch_size}")
    
    # Warm up
    _ = openclip_model.encode_images(test_images[:batch_size], batch_size=batch_size)
    
    # Actual test
    start = time.time()
    embeddings = openclip_model.encode_images(test_images, batch_size=batch_size)
    encode_time = time.time() - start
    
    if embeddings is not None:
        images_per_sec = len(test_images) / encode_time
        print(f"  Time: {encode_time:.2f}s")
        print(f"  Speed: {images_per_sec:.1f} images/sec")
        print(f"  Estimated time for 28,671 images: {28671 / images_per_sec / 60:.1f} minutes")
    else:
        print("  Failed to encode images")

# Memory info
if torch.cuda.is_available():
    print(f"\nGPU Memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"  Cached: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    print(f"  Max available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

print("\n" + "=" * 80) 