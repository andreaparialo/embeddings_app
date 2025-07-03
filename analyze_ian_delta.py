#!/usr/bin/env python3
"""
Quick analysis of delta between IAN/resized and indexed images
"""

import json
from pathlib import Path

def analyze_delta():
    # Load existing indexed images
    metadata_file = "indexes/v11_1095_db_pictures_512_metadata.json"
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    existing_images = set()
    for path in metadata.get('image_paths', []):
        filename = Path(path).name
        existing_images.add(filename)
    
    print(f"✅ Found {len(existing_images)} images in existing index")
    
    # Scan IAN/resized
    ian_path = Path("IAN/resized")
    ian_images = set()
    for img_file in ian_path.glob("*"):
        if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            ian_images.add(img_file.name)
    
    print(f"📁 Found {len(ian_images)} images in IAN/resized")
    
    # Find delta
    new_images = ian_images - existing_images
    already_indexed = ian_images & existing_images
    
    print(f"\n📊 DELTA ANALYSIS:")
    print(f"   🆕 New images (need indexing): {len(new_images)}")
    print(f"   ✅ Already indexed: {len(already_indexed)}")
    print(f"   📊 Percentage new: {len(new_images)/len(ian_images)*100:.1f}%")
    
    if len(new_images) < 20 and len(new_images) > 0:
        print("\n🆕 New images to index:")
        for i, img in enumerate(sorted(new_images), 1):
            print(f"   {i:2d}. {img}")
    
    # Save list of new images
    if new_images:
        with open("ian_delta_images.txt", "w") as f:
            for img in sorted(new_images):
                f.write(f"{img}\n")
        print(f"\n💾 Saved list of {len(new_images)} new images to: ian_delta_images.txt")

if __name__ == "__main__":
    analyze_delta() 