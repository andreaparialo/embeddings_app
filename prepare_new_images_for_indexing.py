#!/usr/bin/env python3
"""
Prepare New Images for LoRA Indexing
Creates a directory with symlinks to only new (not already indexed) images
"""

import json
import os
import shutil
from pathlib import Path
import sys
from datetime import datetime


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


def prepare_new_images_directory(source_dir, existing_images, output_dir="temp_new_images_for_indexing"):
    """Create directory with symlinks to only new images"""
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Remove output directory if it exists
    if output_path.exists():
        print(f"⚠️  Removing existing directory: {output_dir}")
        shutil.rmtree(output_path)
    
    # Create fresh output directory
    output_path.mkdir(parents=True)
    print(f"📁 Created directory: {output_dir}")
    
    # Find all images
    supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    all_images = []
    
    for ext in supported_formats:
        all_images.extend(source_path.rglob(f"*{ext}"))
        all_images.extend(source_path.rglob(f"*{ext.upper()}"))
    
    # Copy new images only
    new_count = 0
    skip_count = 0
    
    print(f"\n📋 Copying new images...")
    
    for img_path in all_images:
        filename = img_path.name
        
        if filename in existing_images:
            skip_count += 1
        else:
            # Copy file
            dest_path = output_path / filename
            shutil.copy2(img_path, dest_path)
            new_count += 1
            
            if new_count <= 5:
                print(f"   ✓ Copied: {filename}")
    
    if new_count > 5:
        print(f"   ... and {new_count - 5} more")
    
    print(f"\n📊 Summary:")
    print(f"   New images (copied): {new_count}")
    print(f"   Already indexed (skipped): {skip_count}")
    print(f"   Total processed: {new_count + skip_count}")
    
    return output_path, new_count


def generate_indexing_commands(temp_dir, lora_checkpoint, index_name):
    """Generate the indexing commands"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n🚀 INDEXING COMMANDS")
    print("=" * 60)
    
    # First, resize and move images to pictures/ directory
    print("Step 1: Resize and move images to pictures/ directory")
    print(f"python3 image_resizer.py --source {temp_dir} --destination pictures/")
    
    print("\nStep 2: Index the new images with LoRA")
    print(f"""python3 lora_max_performance_indexing_custom.py \\
    "{lora_checkpoint}" \\
    "pictures" \\
    "{index_name}_{timestamp}" \\
    "indexes"
""")
    
    print("\n⚠️  IMPORTANT: The indexing will process ALL images in pictures/")
    print("   This is OK because the LoRA indexing creates a complete new index")
    print("   including both old and new images.")


def main():
    print("🔧 PREPARE NEW IMAGES FOR LoRA INDEXING")
    print("=" * 80)
    
    # Configuration
    source_dir = "rel20260101/pictures_O00_20260101"
    temp_dir = "temp_new_images_for_indexing"
    
    # Recommended LoRA checkpoint (matching existing indexes)
    lora_checkpoint = "loras/v11-20250620-105815/checkpoint-1095"
    index_name = "v11_o00_complete"
    
    print(f"📁 Source directory: {source_dir}")
    print(f"📁 Temp directory: {temp_dir}")
    print(f"🎯 LoRA checkpoint: {lora_checkpoint}")
    
    # Get already indexed images
    print("\n🔍 Loading existing index information...")
    existing_images = get_already_indexed_images()
    print(f"   Found {len(existing_images)} already indexed images")
    
    # Prepare directory with new images only
    output_path, new_count = prepare_new_images_directory(
        source_dir, existing_images, temp_dir
    )
    
    if new_count == 0:
        print("\n❌ No new images to process!")
        return
    
    # Generate commands
    generate_indexing_commands(temp_dir, lora_checkpoint, index_name)
    
    print("\n📋 COMPLETE WORKFLOW:")
    print("=" * 60)
    print("1. This script has created symlinks to ONLY new images")
    print("2. Run the image resizer to resize and move them to pictures/")
    print("3. Run the LoRA indexing on the complete pictures/ directory")
    print("4. The new index will contain ALL images (old + new)")
    
    print(f"\n✅ Ready to process {new_count} new images!")


if __name__ == "__main__":
    main() 