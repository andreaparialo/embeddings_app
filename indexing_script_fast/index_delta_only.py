#!/usr/bin/env python3
"""
Delta Indexing Script - Index only new images
Uses the proven lora_optimized_indexing.py engine for consistency and performance
"""

import os
import sys
import json
import time
import numpy as np
import faiss
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import torch

# Import the proven LoRA indexing engine
from lora_optimized_indexing import WorkerBasedIndexingEngine

def setup_environment():
    """Setup environment for GPU processing"""
    
    # Suppress warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    
    # CUDA optimizations for A100
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    print(f"🔧 Environment configured for GPU processing")


def load_delta_images(delta_dir: str = "pictures_delta_prepared") -> List[Path]:
    """Load list of prepared delta images"""
    
    delta_path = Path(delta_dir)
    if not delta_path.exists():
        print(f"❌ Delta images directory not found: {delta_dir}")
        print("   Run 'python prepare_delta_images.py' first")
        return []
    
    # Find all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(delta_path.glob(f"*{ext}"))
        image_files.extend(delta_path.glob(f"*{ext.upper()}"))
    
    image_files.sort()
    print(f"📁 Found {len(image_files)} prepared delta images")
    
    return image_files


def save_delta_index_files(index_dir: str, index_name: str, output_prefix: str = "v11_delta") -> Dict[str, str]:
    """Copy and rename the index files with delta naming convention"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Source files (created by lora_optimized_indexing)
    source_files = {
        'faiss': f"{index_dir}/{index_name}.faiss",
        'embeddings': f"{index_dir}/{index_name}_embeddings.npy",
        'metadata': f"{index_dir}/{index_name}_metadata.json"
    }
    
    # Target files (delta naming)
    target_files = {
        'faiss': f"indexes/{output_prefix}_{timestamp}.faiss",
        'embeddings': f"indexes/{output_prefix}_{timestamp}_embeddings.npy",
        'metadata': f"indexes/{output_prefix}_{timestamp}_metadata.json"
    }
    
    # Ensure indexes directory exists
    Path("indexes").mkdir(exist_ok=True)
    
    # Copy files with delta naming
    import shutil
    for file_type in ['faiss', 'embeddings', 'metadata']:
        source = Path(source_files[file_type])
        target = Path(target_files[file_type])
        
        if source.exists():
            shutil.copy2(source, target)
            print(f"✅ Delta {file_type} saved: {target.name}")
        else:
            print(f"⚠️  Warning: Source file not found: {source}")
    
    # Update metadata to indicate this is a delta index
    if Path(target_files['metadata']).exists():
        with open(target_files['metadata'], 'r') as f:
            metadata = json.load(f)
        
        # Add delta-specific metadata
        metadata['index_type'] = 'delta'
        metadata['delta_timestamp'] = datetime.now().isoformat()
        metadata['base_index'] = 'v11_o00_index_1095'
        
        with open(target_files['metadata'], 'w') as f:
            json.dump(metadata, f, indent=2)
    
    return target_files


def main():
    """Main delta indexing function using proven LoRA engine"""
    
    print("🚀 DELTA INDEXING - Using Proven LoRA Engine")
    print("=" * 60)
    print("Indexing only new images with lora_optimized_indexing.py")
    print()
    
    # Setup environment
    setup_environment()
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("❌ CUDA not available - GPU acceleration required")
        sys.exit(1)
    
    print(f"✅ CUDA available - {torch.cuda.device_count()} GPUs detected")
    
    # Load delta images
    delta_images = load_delta_images()
    if not delta_images:
        print("❌ No delta images found to index")
        print("   Run 'python prepare_delta_images.py' first")
        sys.exit(1)
    
    # Check LoRA model exists
    lora_path = Path("loras/v11-20250620-105815/checkpoint-1095")
    if not lora_path.exists():
        print(f"❌ LoRA checkpoint not found: {lora_path}")
        sys.exit(1)
    
    print(f"✅ LoRA checkpoint found: {lora_path}")
    print(f"📊 Delta images to process: {len(delta_images)}")
    print()
    
    # Configuration for delta indexing
    delta_index_name = "delta_temp_index"
    
    print(f"🔧 Delta Indexing Configuration:")
    print(f"   📦 Base model: gme-Qwen2-VL-7B-Instruct")
    print(f"   🔧 LoRA checkpoint: {lora_path}")
    print(f"   📁 Delta images: pictures_delta_prepared/")
    print(f"   🎯 Target index: {delta_index_name}")
    print(f"   👥 Workers: 4 (optimized for delta)")
    print(f"   📦 Batch size: 16 (optimized for delta)")
    print()
    
    # Initialize the proven LoRA indexing engine
    try:
        print("🚀 Initializing proven LoRA indexing engine...")
        
        engine = WorkerBasedIndexingEngine(
            lora_checkpoint_path=str(lora_path),
            num_workers=4,  # Optimized for delta processing
            batch_size=16   # Good balance for delta images
        )
        
        print("✅ LoRA engine initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize LoRA engine: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Index delta images using the proven engine
    try:
        print(f"\n🔄 Starting delta indexing with proven engine...")
        start_time = time.time()
        
        # Use the proven indexing method
        engine.index_images_with_workers(
            image_directory="pictures_delta_prepared",
            index_name=delta_index_name
        )
        
        total_time = time.time() - start_time
        
        print(f"\n✅ DELTA INDEXING COMPLETE")
        print("=" * 40)
        print(f"📊 Results:")
        print(f"   ✅ Images processed: {len(delta_images)}")
        print(f"   ⏱️  Total time: {total_time:.1f} seconds")
        print(f"   📈 Processing rate: {len(delta_images)/total_time:.1f} images/second")
        
    except Exception as e:
        print(f"❌ Delta indexing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Rename index files to delta naming convention
    try:
        print(f"\n📝 Renaming index files to delta convention...")
        
        saved_files = save_delta_index_files(
            index_dir="indexes",
            index_name=delta_index_name,
            output_prefix="v11_delta"
        )
        
        # Clean up temporary files
        temp_files = [
            f"indexes/{delta_index_name}.faiss",
            f"indexes/{delta_index_name}_embeddings.npy", 
            f"indexes/{delta_index_name}_metadata.json"
        ]
        
        for temp_file in temp_files:
            temp_path = Path(temp_file)
            if temp_path.exists():
                temp_path.unlink()
        
        print(f"\n🎯 DELTA INDEX READY FOR MERGING")
        print("=" * 40)
        print(f"📁 Delta index files:")
        for file_type, file_path in saved_files.items():
            if Path(file_path).exists():
                size = Path(file_path).stat().st_size / 1024 / 1024
                print(f"   📄 {file_type.upper()}: {Path(file_path).name} ({size:.1f} MB)")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Review delta index files in indexes/")
        print(f"   2. Run index merging: python merge_indexes.py")
        print(f"   3. Test merged index with search")
        
    except Exception as e:
        print(f"❌ Failed to save delta index files: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 