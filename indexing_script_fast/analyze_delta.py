#!/usr/bin/env python3
"""
Fast Delta Analysis Script - Optimized version
Much faster file list creation using lookup tables
"""

import json
import sys
from pathlib import Path
from typing import Set, List, Dict
import os
from collections import defaultdict

def load_existing_index_images(metadata_file: str = "indexes/v11_o00_index_1095_metadata.json") -> Set[str]:
    """Load list of images already in the existing index"""
    
    if not Path(metadata_file).exists():
        print(f"❌ Existing index metadata not found: {metadata_file}")
        return set()
    
    print(f"📖 Loading existing index metadata: {metadata_file}")
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        existing_images = set()
        image_paths = metadata.get('image_paths', [])
        
        for path in image_paths:
            # Extract filename from path (handle ../o00_images/filename.jpg)
            filename = Path(path).name
            existing_images.add(filename)
        
        print(f"✅ Found {len(existing_images)} images in existing index")
        return existing_images
        
    except Exception as e:
        print(f"❌ Error loading existing index: {e}")
        return set()


def build_current_images_lookup(pictures_dir: str = "pictures") -> Dict[str, Path]:
    """Build a fast lookup table of current images"""
    
    pictures_path = Path(pictures_dir)
    if not pictures_path.exists():
        print(f"❌ Pictures directory not found: {pictures_dir}")
        return {}
    
    print(f"📁 Building fast lookup table for: {pictures_dir}")
    
    # Supported image formats
    supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', 
                        '.JPG', '.JPEG', '.PNG', '.GIF', '.BMP', '.TIFF', '.WEBP']
    
    # Build lookup table: filename -> full_path
    lookup_table = {}
    total_files = 0
    
    for ext in supported_formats:
        image_files = list(pictures_path.rglob(f"*{ext}"))
        for img_file in image_files:
            lookup_table[img_file.name] = img_file
            total_files += 1
    
    print(f"✅ Built lookup table with {len(lookup_table)} unique images ({total_files} total files)")
    return lookup_table


def find_delta_images_fast(existing_images: Set[str], current_lookup: Dict[str, Path]) -> Dict[str, List]:
    """Find the delta between existing and current images using fast lookup"""
    
    print("\n🔍 DELTA ANALYSIS (FAST)")
    print("=" * 50)
    
    current_images = set(current_lookup.keys())
    
    # Find new images (in current but not in existing)
    new_images = current_images - existing_images
    
    # Find missing images (in existing but not in current)
    missing_images = existing_images - current_images
    
    # Find common images
    common_images = existing_images & current_images
    
    delta_info = {
        'new_images': sorted(list(new_images)),
        'missing_images': sorted(list(missing_images)),
        'common_images': sorted(list(common_images))
    }
    
    # Print analysis
    print(f"📊 Analysis Results:")
    print(f"   🆕 New images (need indexing): {len(new_images)}")
    print(f"   🔄 Common images (already indexed): {len(common_images)}")
    print(f"   ❓ Missing images (in index but not current): {len(missing_images)}")
    print(f"   📈 Total current images: {len(current_images)}")
    print(f"   📉 Total existing images: {len(existing_images)}")
    
    return delta_info


def create_delta_file_list_fast(delta_info: Dict[str, List], 
                               current_lookup: Dict[str, Path],
                               output_file: str = "delta_images_list.txt") -> List[Path]:
    """Create list of full paths for delta images using fast lookup"""
    
    print(f"\n📝 Creating delta file list (FAST)...")
    
    delta_image_paths = []
    
    # Use lookup table for instant path resolution
    for filename in delta_info['new_images']:
        if filename in current_lookup:
            delta_image_paths.append(current_lookup[filename])
        else:
            print(f"⚠️  Warning: Could not find path for {filename}")
    
    # Save delta list to file
    with open(output_file, 'w') as f:
        for path in delta_image_paths:
            f.write(str(path) + '\n')
    
    print(f"✅ Delta file list created instantly: {output_file}")
    print(f"   📁 {len(delta_image_paths)} images ready for indexing")
    
    return delta_image_paths


def save_delta_analysis(delta_info: Dict[str, List], 
                       output_file: str = "delta_analysis.json"):
    """Save complete delta analysis to JSON file"""
    
    analysis_summary = {
        'timestamp': str(Path().absolute()),
        'existing_index': 'v11_o00_index_1095',
        'current_directory': 'pictures',
        'statistics': {
            'new_images_count': len(delta_info['new_images']),
            'common_images_count': len(delta_info['common_images']),
            'missing_images_count': len(delta_info['missing_images']),
            'total_current': len(delta_info['new_images']) + len(delta_info['common_images']),
            'total_existing': len(delta_info['common_images']) + len(delta_info['missing_images'])
        },
        'delta_data': delta_info
    }
    
    with open(output_file, 'w') as f:
        json.dump(analysis_summary, f, indent=2)
    
    print(f"✅ Complete analysis saved: {output_file}")


def show_sample_deltas(delta_info: Dict[str, List], sample_size: int = 10):
    """Show sample of delta images for verification"""
    
    print(f"\n🔍 SAMPLE DELTA IMAGES (first {sample_size}):")
    print("-" * 50)
    
    new_images = delta_info['new_images']
    
    if new_images:
        print("🆕 New images to index:")
        for i, filename in enumerate(new_images[:sample_size]):
            print(f"   {i+1:2d}. {filename}")
        
        if len(new_images) > sample_size:
            print(f"   ... and {len(new_images) - sample_size} more")
    else:
        print("✅ No new images found - all images are already indexed!")
    
    # Show missing images if any
    missing_images = delta_info['missing_images']
    if missing_images:
        print(f"\n❓ Sample missing images (in index but not current):")
        for i, filename in enumerate(missing_images[:sample_size]):
            print(f"   {i+1:2d}. {filename}")
        
        if len(missing_images) > sample_size:
            print(f"   ... and {len(missing_images) - sample_size} more")


def main():
    """Main fast delta analysis function"""
    
    print("🚀 FAST DELTA INDEXING ANALYSIS")
    print("=" * 60)
    print("Optimized version with instant file list creation")
    print()
    
    # Configuration
    existing_metadata = "indexes/v11_o00_index_1095_metadata.json"
    pictures_directory = "pictures"
    
    # Check if files exist
    if not Path(existing_metadata).exists():
        print(f"❌ Existing index metadata not found: {existing_metadata}")
        print("   Make sure the v11_o00_index_1095 files are in the indexes/ directory")
        sys.exit(1)
    
    if not Path(pictures_directory).exists():
        print(f"❌ Pictures directory not found: {pictures_directory}")
        sys.exit(1)
    
    # Step 1: Load existing index images
    existing_images = load_existing_index_images(existing_metadata)
    if not existing_images:
        print("❌ Could not load existing index images")
        sys.exit(1)
    
    # Step 2: Build current images lookup table (this replaces the slow scanning)
    current_lookup = build_current_images_lookup(pictures_directory)
    if not current_lookup:
        print("❌ Could not build current images lookup")
        sys.exit(1)
    
    # Step 3: Find delta using fast lookup
    delta_info = find_delta_images_fast(existing_images, current_lookup)
    
    # Step 4: Show samples
    show_sample_deltas(delta_info)
    
    # Step 5: Create file lists instantly using lookup
    delta_paths = create_delta_file_list_fast(delta_info, current_lookup)
    
    # Step 6: Save analysis
    save_delta_analysis(delta_info)
    
    # Summary
    print(f"\n🎯 FAST DELTA ANALYSIS COMPLETE")
    print("=" * 40)
    
    new_count = len(delta_info['new_images'])
    total_current = len(current_lookup)
    
    if new_count == 0:
        print("✅ All images are already indexed!")
        print("   No delta indexing needed.")
    else:
        percentage = (new_count / total_current) * 100
        print(f"📊 Summary:")
        print(f"   🆕 Images to index: {new_count}")
        print(f"   📁 Total current images: {total_current}")
        print(f"   📈 Percentage to process: {percentage:.1f}%")
        print(f"   ⏱️  Estimated time savings: {100-percentage:.1f}%")
        print()
        print(f"💡 Next steps:")
        print(f"   1. Review delta_analysis.json for details")
        print(f"   2. Run delta preparation: python prepare_delta_images.py")
        print(f"   3. Run delta indexing: python index_delta_only.py")
        print(f"   4. Merge indexes: python merge_indexes.py")
    

if __name__ == "__main__":
    main() 