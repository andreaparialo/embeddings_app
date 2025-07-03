#!/usr/bin/env python3
"""
Prepare delta images for indexing
"""

import pandas as pd
import os
import shutil
from tqdm import tqdm

def prepare_delta_images():
    print("📁 Preparing Delta Images for Indexing")
    print("=" * 80)
    
    # Load the list of products with pictures
    missing_with_pics = pd.read_csv('missing_indexes_have_pictures.csv')
    print(f"\n📊 Products to prepare: {len(missing_with_pics):,}")
    
    # Create delta directory
    delta_dir = "delta_images_to_index"
    if os.path.exists(delta_dir):
        print(f"⚠️  Removing existing {delta_dir} directory...")
        shutil.rmtree(delta_dir)
    os.makedirs(delta_dir)
    
    # Copy/link images to delta directory
    pictures_dir = "pictures"
    copied_count = 0
    
    print("\n🔗 Creating symlinks for delta images...")
    for idx, row in tqdm(missing_with_pics.iterrows(), total=len(missing_with_pics), desc="Processing"):
        filename_root = row['filename_root']
        
        # Try different image path variations
        image_found = False
        for ext in ['.jpg', '.JPG']:
            for suffix in ['_O00', '']:
                source_path = os.path.join(pictures_dir, f"{filename_root}{suffix}{ext}")
                if os.path.exists(source_path):
                    # Create symlink in delta directory
                    dest_filename = f"{filename_root}{suffix}{ext}"
                    dest_path = os.path.join(delta_dir, dest_filename)
                    
                    try:
                        # Create symlink instead of copying (faster)
                        os.symlink(os.path.abspath(source_path), dest_path)
                        copied_count += 1
                        image_found = True
                    except Exception as e:
                        print(f"\n⚠️  Error linking {filename_root}: {e}")
                    
                    break
            if image_found:
                break
        
        if not image_found:
            print(f"\n❌ No image found for {filename_root}")
    
    print(f"\n✅ Prepared {copied_count:,} images in {delta_dir}")
    
    # Save info for later
    delta_info = {
        'delta_dir': delta_dir,
        'num_images': copied_count,
        'filename_roots': missing_with_pics['filename_root'].tolist()
    }
    
    import json
    with open('delta_preparation_info.json', 'w') as f:
        json.dump(delta_info, f, indent=2)
    
    print(f"💾 Saved preparation info to delta_preparation_info.json")
    
    return delta_dir, copied_count

if __name__ == "__main__":
    prepare_delta_images() 