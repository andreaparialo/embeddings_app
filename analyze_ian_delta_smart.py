#!/usr/bin/env python3
"""
Smart analysis of delta between IAN/resized and indexed images
Handles naming differences like _P02 suffix
"""

import json
from pathlib import Path

def get_base_filename(filename):
    """Extract base filename without _P02 suffix and extension"""
    # Remove extension
    name_no_ext = Path(filename).stem
    # Remove _P02 or similar suffixes
    if '_P' in name_no_ext:
        base = name_no_ext.split('_P')[0]
    else:
        base = name_no_ext
    return base

def analyze_delta_smart():
    # Load existing indexed images
    metadata_file = "indexes/v11_1095_db_pictures_512_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    # Build set of base filenames from existing index
    existing_bases = set()
    existing_full = {}  # base -> full filename
    for path in metadata.get('image_paths', []):
        filename = Path(path).name
        base = get_base_filename(filename)
        existing_bases.add(base)
        existing_full[base] = filename
    
    print(f"✅ Found {len(existing_bases)} unique base filenames in existing index")
    
    # Scan IAN/resized
    ian_path = Path("IAN/resized")
    ian_bases = {}  # base -> full filename
    for img_file in ian_path.glob("*"):
        if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            base = get_base_filename(img_file.name)
            ian_bases[base] = img_file.name
    
    print(f"📁 Found {len(ian_bases)} unique base filenames in IAN/resized")
    
    # Find delta based on base filenames
    ian_base_set = set(ian_bases.keys())
    new_bases = ian_base_set - existing_bases
    already_indexed_bases = ian_base_set & existing_bases
    
    # Convert back to full filenames
    new_images = [ian_bases[base] for base in new_bases]
    already_indexed = [ian_bases[base] for base in already_indexed_bases]
    
    print(f"\n📊 SMART DELTA ANALYSIS (comparing base filenames):")
    print(f"   🆕 New images (need indexing): {len(new_images)}")
    print(f"   ✅ Already indexed (different suffix): {len(already_indexed)}")
    print(f"   📊 Percentage new: {len(new_images)/len(ian_bases)*100:.1f}%")
    
    # Show some examples of matches
    if already_indexed_bases and len(already_indexed_bases) < 10:
        print("\n📝 Examples of already indexed (with different names):")
        for i, base in enumerate(list(already_indexed_bases)[:5], 1):
            print(f"   {i}. IAN: {ian_bases[base]} → DB: {existing_full.get(base, 'N/A')}")
    
    if len(new_images) < 20 and len(new_images) > 0:
        print("\n🆕 New images to index:")
        for i, img in enumerate(sorted(new_images)[:20], 1):
            print(f"   {i:2d}. {img}")
    
    # Save list of truly new images
    if new_images:
        with open("ian_delta_images_smart.txt", "w") as f:
            for img in sorted(new_images):
                f.write(f"{img}\n")
        print(f"\n💾 Saved list of {len(new_images)} truly new images to: ian_delta_images_smart.txt")
    
    return new_images, already_indexed

if __name__ == "__main__":
    new_images, already_indexed = analyze_delta_smart() 