#!/usr/bin/env python3
"""Check GME model files location"""

import os

print("🔍 Checking for GME model files...")
print("=" * 60)

# Check for model directories
model_names = [
    "gme-Qwen2-VL-7B-Instruct",
    "GME-Qwen2-VL-7B-Instruct",
    "gme-qwen2-vl-7b-instruct",
    "GME-QWEN2-VL-7B-INSTRUCT"
]

found = False
for name in model_names:
    if os.path.exists(name):
        print(f"✅ Found model directory: {name}")
        found = True
        # List contents
        print("   Contents:")
        for item in os.listdir(name)[:10]:  # Show first 10 items
            print(f"   - {item}")
        break

if not found:
    print("❌ GME model directory not found in current directory")

print("\n🔍 Checking for LoRA directories...")
lora_paths = [
    "loras/v11-20250620-105815/checkpoint-1095",
    "loras/v11-20250620-105815"
]

lora_found = False
for path in lora_paths:
    if os.path.exists(path):
        print(f"✅ Found LoRA path: {path}")
        lora_found = True
        if os.path.isdir(path):
            print("   Contents:")
            for item in os.listdir(path)[:5]:
                print(f"   - {item}")
        break

if not lora_found:
    print("❌ LoRA checkpoint not found")

print("\n" + "=" * 60)
print("📝 Summary:")
if not found:
    print("⚠️  GME model needs to be downloaded or linked")
    print("   Expected location: ./gme-Qwen2-VL-7B-Instruct")
    print("   You can:")
    print("   1. Download the model from HuggingFace")
    print("   2. Create a symlink to existing model location")
    print("   3. Update the model path in gme_model.py")
else:
    print("✅ GME model found")

if not lora_found:
    print("\n⚠️  LoRA checkpoint not found")
    print("   Expected: loras/v11-20250620-105815/checkpoint-1095")
else:
    print("✅ LoRA checkpoint found")

print("\n💡 To proceed:")
print("1. Stop the current process (Ctrl+C)")
print("2. Ensure model files are in place")
print("3. Run: ./start_gpu.sh") 