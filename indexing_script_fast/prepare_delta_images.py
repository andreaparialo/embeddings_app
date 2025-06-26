#!/usr/bin/env python3
"""
Delta Image Preparation Script
Prepares only the new images that need to be indexed (delta set)
"""

import json
import sys
from pathlib import Path
from PIL import Image
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import os

def load_delta_analysis(analysis_file: str = "delta_analysis.json") -> Dict[str, Any]:
    """Load delta analysis results"""
    
    if not Path(analysis_file).exists():
        print(f"❌ Delta analysis file not found: {analysis_file}")
        print("   Run 'python analyze_delta.py' first")
        return {}
    
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    
    return analysis


def prepare_single_image(source_path: Path, target_path: Path, max_size: int = 512) -> Dict[str, Any]:
    """Prepare a single image (resize and optimize)"""
    
    result = {
        'source': str(source_path),
        'target': str(target_path),
        'success': False,
        'original_size': 0,
        'new_size': 0,
        'dimensions': None,
        'error': None
    }
    
    try:
        # Get original file size
        result['original_size'] = source_path.stat().st_size
        
        # Open and process image
        with Image.open(source_path) as img:
            # Convert to RGB if necessary
            if img.mode in ('RGBA', 'LA', 'P'):
                img = img.convert('RGB')
            
            # Calculate new dimensions (maintain aspect ratio)
            width, height = img.size
            if width > height:
                new_width = min(width, max_size)
                new_height = int((height * new_width) / width)
            else:
                new_height = min(height, max_size)
                new_width = int((width * new_height) / height)
            
            # Only resize if image is larger than max_size
            if width > max_size or height > max_size:
                img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            result['dimensions'] = f"{new_width}x{new_height}"
            
            # Save optimized image
            target_path.parent.mkdir(parents=True, exist_ok=True)
            img.save(target_path, 'JPEG', quality=95, optimize=True)
            
            # Get new file size
            result['new_size'] = target_path.stat().st_size
            result['success'] = True
            
    except Exception as e:
        result['error'] = str(e)
    
    return result


def prepare_delta_images(delta_file_list: str = "delta_images_list.txt",
                        target_dir: str = "pictures_delta_prepared",
                        max_workers: int = 8,
                        max_size: int = 512) -> Dict[str, Any]:
    """Prepare all delta images with multi-threading"""
    
    # Load delta image list
    if not Path(delta_file_list).exists():
        print(f"❌ Delta file list not found: {delta_file_list}")
        print("   Run 'python analyze_delta.py' first")
        return {}
    
    with open(delta_file_list, 'r') as f:
        image_paths = [Path(line.strip()) for line in f if line.strip()]
    
    if not image_paths:
        print("✅ No delta images to prepare")
        return {'total': 0, 'success': 0, 'failed': 0}
    
    print(f"🔄 Preparing {len(image_paths)} delta images...")
    print(f"   📁 Source: various paths")
    print(f"   📁 Target: {target_dir}/")
    print(f"   🔧 Max size: {max_size}px")
    print(f"   👥 Workers: {max_workers}")
    print()
    
    # Create target directory
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare tasks
    tasks = []
    for source_path in image_paths:
        if source_path.exists():
            # Create target path (preserve filename)
            target_file = target_path / source_path.name.lower()
            # Change extension to .jpg if not already
            if target_file.suffix.lower() not in ['.jpg', '.jpeg']:
                target_file = target_file.with_suffix('.jpg')
            
            tasks.append((source_path, target_file))
        else:
            print(f"⚠️  Warning: Source file not found: {source_path}")
    
    # Process images with threading
    results = []
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(prepare_single_image, source, target, max_size): (source, target)
            for source, target in tasks
        }
        
        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_task), 1):
            result = future.result()
            results.append(result)
            
            # Progress update
            if i % 50 == 0 or i == len(tasks):
                elapsed = time.time() - start_time
                rate = i / elapsed if elapsed > 0 else 0
                print(f"   📈 Progress: {i:,}/{len(tasks):,} ({i/len(tasks)*100:.1f}%) - {rate:.1f} imgs/sec")
    
    # Calculate statistics
    total_time = time.time() - start_time
    successful = sum(1 for r in results if r['success'])
    failed = len(results) - successful
    
    total_original_size = sum(r['original_size'] for r in results if r['success'])
    total_new_size = sum(r['new_size'] for r in results if r['success'])
    
    size_reduction = ((total_original_size - total_new_size) / total_original_size * 100) if total_original_size > 0 else 0
    
    # Print summary
    print(f"\n✅ DELTA PREPARATION COMPLETE")
    print("=" * 40)
    print(f"📊 Results:")
    print(f"   ✅ Successfully prepared: {successful:,}")
    print(f"   ❌ Failed: {failed:,}")
    print(f"   ⏱️  Total time: {total_time:.1f} seconds")
    print(f"   📈 Processing rate: {len(results)/total_time:.1f} images/second")
    print()
    print(f"💾 Storage optimization:")
    print(f"   📦 Original size: {total_original_size/1024/1024:.1f} MB")
    print(f"   📦 New size: {total_new_size/1024/1024:.1f} MB")
    print(f"   📉 Size reduction: {size_reduction:.1f}%")
    
    # Show failed images if any
    if failed > 0:
        print(f"\n❌ Failed images:")
        for result in results:
            if not result['success']:
                print(f"   {Path(result['source']).name}: {result['error']}")
    
    return {
        'total': len(results),
        'success': successful,
        'failed': failed,
        'processing_time': total_time,
        'processing_rate': len(results)/total_time if total_time > 0 else 0,
        'original_size_mb': total_original_size/1024/1024,
        'new_size_mb': total_new_size/1024/1024,
        'size_reduction_percent': size_reduction,
        'results': results
    }


def main():
    """Main delta preparation function"""
    
    print("🔄 DELTA IMAGE PREPARATION")
    print("=" * 50)
    print("Preparing only new images that need to be indexed")
    print()
    
    # Check if delta analysis exists
    if not Path("delta_analysis.json").exists():
        print("❌ Delta analysis not found. Running analysis first...")
        os.system("python analyze_delta.py")
        print()
    
    # Load delta analysis
    analysis = load_delta_analysis()
    if not analysis:
        print("❌ Could not load delta analysis")
        sys.exit(1)
    
    new_images_count = analysis['statistics']['new_images_count']
    
    if new_images_count == 0:
        print("✅ No new images to prepare - all images are already indexed!")
        return
    
    print(f"📊 Delta Analysis Summary:")
    print(f"   🆕 New images to prepare: {new_images_count:,}")
    print(f"   🔄 Already indexed: {analysis['statistics']['common_images_count']:,}")
    print()
    
    # Prepare delta images
    results = prepare_delta_images()
    
    if results and results['success'] > 0:
        print(f"\n💡 Next steps:")
        print(f"   1. Review prepared images in: pictures_delta_prepared/")
        print(f"   2. Run delta indexing: python index_delta_only.py")
        print(f"   3. Merge indexes: python merge_indexes.py")
    else:
        print("❌ No images were successfully prepared")


if __name__ == "__main__":
    main() 