#!/usr/bin/env python3
"""
Test Delta Indexing with Small Batch
Tests the indexing process with a small number of images first
"""

import json
import shutil
from pathlib import Path
from datetime import datetime
import subprocess
import sys


def create_test_delta_directory(num_images=10):
    """Create a test delta directory with a small number of new images"""
    
    # Get already indexed images
    all_indexed = set()
    for metadata_file in Path("indexes").glob("*_metadata.json"):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        for path in metadata.get('image_paths', []):
            filename = Path(path).name
            all_indexed.add(filename)
    
    print(f"📊 Found {len(all_indexed)} already indexed images")
    
    # Find new images in pictures directory
    pictures_path = Path("pictures")
    all_images = list(pictures_path.glob("*.jpg")) + list(pictures_path.glob("*.JPG"))
    
    new_images = []
    for img_path in all_images:
        if img_path.name not in all_indexed:
            new_images.append(img_path)
    
    print(f"✨ Found {len(new_images)} new images in pictures/")
    
    # Create test delta directory
    test_dir = Path("test_delta_images")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    # Copy only a small number of images for testing
    test_images = new_images[:num_images]
    
    print(f"\n📋 Copying {num_images} images for testing...")
    for img in test_images:
        dest = test_dir / img.name
        shutil.copy2(img, dest)
        print(f"   ✓ {img.name}")
    
    return test_dir, len(test_images)


def main():
    print("🧪 TEST DELTA INDEXING WITH SMALL BATCH")
    print("=" * 60)
    
    # Create test directory with 10 images
    test_dir, num_images = create_test_delta_directory(10)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_index_name = f"v11_delta_test_{timestamp}"
    
    print(f"\n🚀 Testing indexing with {num_images} images")
    print(f"📁 Test directory: {test_dir}")
    print(f"📝 Test index name: {test_index_name}")
    
    # Run indexing command
    cmd = [
        sys.executable,
        "lora_max_performance_indexing_custom.py",
        "loras/v11-20250620-105815/checkpoint-1095",
        str(test_dir),
        test_index_name,
        "indexes"
    ]
    
    print("\n🔧 Running test indexing...")
    print(" ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("\n✅ Test indexing successful!")
            print("\nCleaning up test directory...")
            shutil.rmtree(test_dir)
            
            print(f"\n💡 Test index created: {test_index_name}")
            print("   You can now proceed with full delta indexing")
            
        else:
            print(f"\n❌ Test indexing failed!")
            print(f"Error: {result.stderr}")
            
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        print("Cleaning up...")
        if test_dir.exists():
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    main() 