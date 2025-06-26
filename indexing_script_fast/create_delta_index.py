#!/usr/bin/env python3
"""
Create Delta Index for LoRA
Indexes only the new images, creating a separate delta index
"""

import json
import os
import shutil
from pathlib import Path
from datetime import datetime
import subprocess


def get_already_indexed_images(index_dir="indexes"):
    """Get set of all already indexed image filenames"""
    all_indexed = set()
    
    for metadata_file in Path(index_dir).glob("*_metadata.json"):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        for path in metadata.get('image_paths', []):
            filename = Path(path).name
            all_indexed.add(filename)
    
    return all_indexed


def create_delta_image_directory(existing_images, pictures_dir="pictures", delta_dir="delta_images_to_index"):
    """Create directory with only new images from pictures folder"""
    
    pictures_path = Path(pictures_dir)
    delta_path = Path(delta_dir)
    
    # Remove delta directory if it exists
    if delta_path.exists():
        print(f"⚠️  Removing existing directory: {delta_dir}")
        shutil.rmtree(delta_path)
    
    # Create fresh delta directory
    delta_path.mkdir(parents=True)
    print(f"📁 Created delta directory: {delta_dir}")
    
    # Find all images in pictures directory
    all_images = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        all_images.extend(pictures_path.glob(f"*{ext}"))
    
    # Copy new images only
    new_count = 0
    skip_count = 0
    
    print(f"\n📋 Copying delta images...")
    
    for img_path in all_images:
        filename = img_path.name
        
        if filename in existing_images:
            skip_count += 1
        else:
            # Copy file to delta directory
            dest_path = delta_path / filename
            shutil.copy2(img_path, dest_path)
            new_count += 1
            
            if new_count <= 5:
                print(f"   ✓ Copied: {filename}")
    
    if new_count > 5:
        print(f"   ... and {new_count - 5} more")
    
    print(f"\n📊 Summary:")
    print(f"   New images for delta index: {new_count}")
    print(f"   Already indexed (skipped): {skip_count}")
    
    return delta_path, new_count


def main():
    print("🔧 CREATE DELTA INDEX FOR LoRA")
    print("=" * 80)
    print("This will create an index containing ONLY new images")
    print()
    
    # Configuration
    pictures_dir = "pictures"
    delta_dir = "delta_images_to_index"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Recommended LoRA checkpoint
    lora_checkpoint = "loras/v11-20250620-105815/checkpoint-1095"
    delta_index_name = f"v11_delta_{timestamp}"
    
    print(f"📁 Source: {pictures_dir} (all images)")
    print(f"📁 Delta directory: {delta_dir} (only new images)")
    print(f"🎯 LoRA checkpoint: {lora_checkpoint}")
    print(f"📝 Delta index name: {delta_index_name}")
    
    # Get already indexed images
    print("\n🔍 Loading existing index information...")
    existing_images = get_already_indexed_images()
    print(f"   Found {len(existing_images)} already indexed images")
    
    # Create delta directory with new images only
    delta_path, new_count = create_delta_image_directory(
        existing_images, pictures_dir, delta_dir
    )
    
    if new_count == 0:
        print("\n❌ No new images to index!")
        return
    
    # Generate indexing command for delta
    print(f"\n🚀 DELTA INDEXING COMMAND")
    print("=" * 60)
    print(f"This will create an index with ONLY the {new_count} new images:")
    print()
    
    delta_cmd = f"""python3 lora_max_performance_indexing_custom.py \\
    "{lora_checkpoint}" \\
    "{delta_dir}" \\
    "{delta_index_name}" \\
    "indexes"
"""
    
    print(delta_cmd)
    
    print("\n📋 WORKFLOW OPTIONS:")
    print("=" * 60)
    print("\nOption 1: DELTA INDEX (Recommended for your use case)")
    print(f"  - Creates index with only {new_count} new images")
    print(f"  - Index name: {delta_index_name}")
    print("  - Can be used alongside existing indexes")
    print("  - Faster to create and update")
    
    print("\nOption 2: COMPLETE INDEX")
    print("  - Would index all ~29,000 images")
    print("  - Single unified index")
    print("  - Takes longer to create")
    
    print(f"\n✅ Ready to create delta index with {new_count} new images!")
    print("\n💡 After indexing, you'll have:")
    print("   - Existing index: v11_o00_index_1095 (27,018 images)")
    print(f"   - Delta index: {delta_index_name} ({new_count} images)")
    print("   - Total coverage: ~29,000 images across both indexes")
    
    # Ask user if they want to proceed
    print("\n" + "="*60)
    response = input("Proceed with delta indexing? (y/n): ").lower().strip()
    
    if response == 'y':
        print("\n🚀 Starting delta indexing...")
        try:
            subprocess.run(delta_cmd, shell=True, check=True)
            print("\n✅ Delta indexing completed!")
            
            # Cleanup
            print(f"\n🧹 Cleaning up {delta_dir}...")
            shutil.rmtree(delta_path)
            print("✅ Cleanup completed!")
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Indexing failed: {e}")
    else:
        print("\n❌ Delta indexing cancelled")
        print(f"💡 The delta directory '{delta_dir}' is ready if you want to run manually")


if __name__ == "__main__":
    main() 