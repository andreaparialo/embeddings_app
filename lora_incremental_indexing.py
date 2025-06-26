#!/usr/bin/env python3
"""
Incremental LoRA Indexing Script
Helps to index new images without duplicating existing indexed images
"""

import json
import os
from pathlib import Path
import sys
import numpy as np


def analyze_existing_indexes(index_dir="indexes"):
    """Analyze existing indexes to understand what's already indexed"""
    index_path = Path(index_dir)
    
    print("🔍 ANALYZING EXISTING INDEXES")
    print("=" * 60)
    
    # Find all metadata files
    metadata_files = list(index_path.glob("*_metadata.json"))
    
    if not metadata_files:
        print("No existing indexes found!")
        return None
    
    all_indexed_images = set()
    index_info = []
    
    for metadata_file in sorted(metadata_files):
        index_name = metadata_file.stem.replace("_metadata", "")
        
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        image_paths = metadata.get('image_paths', [])
        lora_checkpoint = metadata.get('lora_checkpoint_path', 'Unknown')
        
        # Extract just the filenames (not full paths) for comparison
        image_filenames = set()
        for path in image_paths:
            filename = Path(path).name
            image_filenames.add(filename)
            all_indexed_images.add(filename)
        
        info = {
            'index_name': index_name,
            'total_images': len(image_paths),
            'lora_checkpoint': lora_checkpoint,
            'metadata_file': str(metadata_file),
            'image_filenames': image_filenames
        }
        index_info.append(info)
        
        print(f"\n📁 Index: {index_name}")
        print(f"   Images: {len(image_paths)}")
        print(f"   LoRA: {lora_checkpoint}")
        print(f"   Sample images:")
        for i, path in enumerate(image_paths[:3]):
            print(f"     - {Path(path).name}")
        if len(image_paths) > 3:
            print(f"     ... and {len(image_paths) - 3} more")
    
    print(f"\n📊 TOTAL UNIQUE IMAGES ALREADY INDEXED: {len(all_indexed_images)}")
    
    return index_info, all_indexed_images


def analyze_new_images(image_dir, existing_images=None):
    """Analyze new images to be indexed"""
    image_path = Path(image_dir)
    
    print(f"\n🖼️  ANALYZING NEW IMAGES IN: {image_dir}")
    print("=" * 60)
    
    if not image_path.exists():
        print(f"❌ Directory {image_dir} does not exist!")
        return []
    
    # Find all image files
    supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    all_images = []
    
    for ext in supported_formats:
        all_images.extend(image_path.rglob(f"*{ext}"))
        all_images.extend(image_path.rglob(f"*{ext.upper()}"))
    
    print(f"📁 Total images found: {len(all_images)}")
    
    if existing_images:
        # Filter out already indexed images
        new_images = []
        duplicates = []
        
        for img_path in all_images:
            filename = img_path.name
            if filename in existing_images:
                duplicates.append(img_path)
            else:
                new_images.append(img_path)
        
        print(f"✅ New images to index: {len(new_images)}")
        print(f"⚠️  Already indexed (will skip): {len(duplicates)}")
        
        if duplicates and len(duplicates) <= 10:
            print("\nDuplicate examples:")
            for dup in duplicates[:10]:
                print(f"   - {dup.name}")
        
        return new_images
    
    return all_images


def list_available_loras(lora_dir="loras"):
    """List all available LoRA checkpoints"""
    lora_path = Path(lora_dir)
    
    print(f"\n🎯 AVAILABLE LoRA CHECKPOINTS")
    print("=" * 60)
    
    # Find all checkpoint directories
    checkpoints = []
    
    for version_dir in sorted(lora_path.iterdir()):
        if version_dir.is_dir():
            # Look for checkpoint subdirectories
            checkpoint_dirs = list(version_dir.glob("checkpoint-*"))
            
            for checkpoint_dir in sorted(checkpoint_dirs):
                if checkpoint_dir.is_dir():
                    checkpoints.append({
                        'version': version_dir.name,
                        'checkpoint': checkpoint_dir.name,
                        'path': str(checkpoint_dir)
                    })
    
    if not checkpoints:
        print("No LoRA checkpoints found!")
        return []
    
    # Group by version
    by_version = {}
    for cp in checkpoints:
        version = cp['version']
        if version not in by_version:
            by_version[version] = []
        by_version[version].append(cp)
    
    # Print organized list
    for i, (version, cps) in enumerate(sorted(by_version.items(), reverse=True)):
        print(f"\n{i+1}. {version}")
        for cp in cps:
            print(f"   - {cp['checkpoint']} → {cp['path']}")
    
    return checkpoints


def generate_incremental_index_command(new_images_dir, lora_checkpoint, index_name, save_dir="indexes"):
    """Generate the command to run incremental indexing"""
    
    print(f"\n🚀 INCREMENTAL INDEXING COMMAND")
    print("=" * 60)
    
    # First, we need to prepare the new images
    # For now, we'll show the command structure
    
    command = f"""python3 lora_max_performance_indexing_custom.py \\
    "{lora_checkpoint}" \\
    "{new_images_dir}" \\
    "{index_name}" \\
    "{save_dir}"
"""
    
    print("Command to run:")
    print(command)
    
    return command


def suggest_indexing_strategy(index_info, new_images, checkpoints):
    """Suggest the best indexing strategy"""
    
    print(f"\n💡 INDEXING STRATEGY RECOMMENDATIONS")
    print("=" * 60)
    
    if not index_info:
        print("📌 No existing indexes found. You can start fresh with any LoRA checkpoint.")
        
        # Suggest latest checkpoint
        if checkpoints:
            latest = checkpoints[0]
            print(f"\n✅ Recommended checkpoint: {latest['path']}")
            print(f"   (Latest version: {latest['version']})")
    else:
        # Find the most recent index
        latest_index = index_info[0]
        
        print(f"📌 Most recent index: {latest_index['index_name']}")
        print(f"   Used LoRA: {latest_index['lora_checkpoint']}")
        print(f"   Contains: {latest_index['total_images']} images")
        
        # Check if we should use the same LoRA
        existing_lora_path = Path(latest_index['lora_checkpoint'])
        
        # Normalize paths for comparison
        if existing_lora_path.exists():
            print(f"\n✅ Recommended: Use the same LoRA checkpoint for consistency")
            print(f"   Path: {existing_lora_path}")
        else:
            # Try to find equivalent in current structure
            checkpoint_name = existing_lora_path.name
            version_match = None
            
            for cp in checkpoints:
                if checkpoint_name in cp['path']:
                    version_match = cp
                    break
            
            if version_match:
                print(f"\n✅ Found matching checkpoint: {version_match['path']}")
            else:
                print(f"\n⚠️  Original LoRA not found. Choose a similar checkpoint.")
    
    if new_images:
        print(f"\n📊 New images to index: {len(new_images)}")
        print("\n🎯 Suggested index name format:")
        print("   - For updating existing: v11_o00_updated_YYYYMMDD")
        print("   - For new version: v11_new_images_checkpoint_XXX")


def create_filtered_image_directory(new_images, output_dir="temp_new_images"):
    """Create a temporary directory with only new images"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"\n📁 Creating filtered image directory: {output_dir}")
    
    # We'll need to either:
    # 1. Create symlinks to new images only
    # 2. Copy new images only
    # 3. Modify the indexing script to accept a list of files
    
    # For now, let's show the approach
    print("Options for handling new images only:")
    print("1. Create symbolic links to new images in a temp directory")
    print("2. Modify indexing script to accept a file list")
    print("3. Copy new images to a temp directory (uses more disk space)")
    
    return output_path


def main():
    print("🔧 LoRA INCREMENTAL INDEXING HELPER")
    print("=" * 80)
    print("This tool helps you index new images without duplicating existing ones")
    print()
    
    # Step 1: Analyze existing indexes
    index_info, existing_images = analyze_existing_indexes()
    
    # Step 2: List available LoRA checkpoints
    checkpoints = list_available_loras()
    
    # Step 3: Analyze new images
    # Using the specific directory mentioned by the user
    new_images_dir = "rel20260101/pictures_O00_20260101"
    new_images = analyze_new_images(new_images_dir, existing_images)
    
    # Step 4: Provide recommendations
    suggest_indexing_strategy(index_info, new_images, checkpoints)
    
    # Step 5: Show how to proceed
    print("\n📋 NEXT STEPS:")
    print("=" * 60)
    
    if new_images:
        print(f"1. You have {len(new_images)} new images to index")
        print("2. Choose a LoRA checkpoint from the list above")
        print("3. Decide on handling approach:")
        print("   a) Create a temporary directory with only new images")
        print("   b) Modify the indexing script to skip duplicates")
        print("\nExample command (after preparing new images):")
        
        # Use v11 checkpoint as example (matching existing index)
        example_checkpoint = "loras/v11-20250620-105815/checkpoint-1095"
        example_index_name = "v11_new_batch_20260101"
        
        generate_incremental_index_command(
            "temp_new_images",  # Would contain only new images
            example_checkpoint,
            example_index_name
        )
    else:
        print("❌ No new images found to index!")
        print("All images in the directory are already indexed.")
    
    print("\n💡 TIP: To avoid re-indexing, we need to either:")
    print("   1. Create a filtered directory with only new images, OR")
    print("   2. Modify the indexing script to check against existing index")


if __name__ == "__main__":
    main() 