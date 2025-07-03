#!/usr/bin/env python3
"""
Smart Delta Indexing for IAN/resized images
Only indexes truly new images (handling _P02 suffix differences)
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Set, List, Dict, Tuple
import os
import numpy as np
import faiss
import time
from datetime import datetime
from tqdm import tqdm
import logging
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_base_filename(filename):
    """Extract base filename without _P02 suffix and extension"""
    name_no_ext = Path(filename).stem
    if '_P' in name_no_ext:
        base = name_no_ext.split('_P')[0]
    else:
        base = name_no_ext
    return base

def load_existing_indexed_images(metadata_file: str = "indexes/v11_1095_db_pictures_512_metadata.json") -> Tuple[Set[str], Dict[str, str]]:
    """Load existing indexed images and return base names"""
    
    if not Path(metadata_file).exists():
        logger.error(f"❌ Existing index metadata not found: {metadata_file}")
        return set(), {}
    
    logger.info(f"📖 Loading existing index metadata: {metadata_file}")
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        existing_bases = set()
        existing_full = {}
        
        for path in metadata.get('image_paths', []):
            filename = Path(path).name
            base = get_base_filename(filename)
            existing_bases.add(base)
            existing_full[base] = filename
        
        logger.info(f"✅ Found {len(existing_bases)} unique base filenames in existing index")
        return existing_bases, existing_full
        
    except Exception as e:
        logger.error(f"❌ Error loading existing index: {e}")
        return set(), {}


def scan_ian_resized_folder(ian_folder: str = "IAN/resized") -> Dict[str, Path]:
    """Scan IAN/resized folder and return base->path mapping"""
    
    ian_path = Path(ian_folder)
    if not ian_path.exists():
        logger.error(f"❌ IAN folder not found: {ian_folder}")
        return {}
    
    logger.info(f"📁 Scanning: {ian_folder}")
    
    # Build lookup table with base names
    ian_bases = {}
    
    for img_file in ian_path.glob("*"):
        if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            base = get_base_filename(img_file.name)
            ian_bases[base] = img_file
    
    logger.info(f"✅ Found {len(ian_bases)} unique base filenames in IAN/resized")
    return ian_bases


def find_delta_images(existing_bases: Set[str], ian_bases: Dict[str, Path]) -> List[Path]:
    """Find truly new images not in the existing index"""
    
    logger.info("\n🔍 SMART DELTA ANALYSIS")
    logger.info("=" * 50)
    
    ian_base_set = set(ian_bases.keys())
    
    # Find new bases
    new_bases = ian_base_set - existing_bases
    already_indexed_bases = ian_base_set & existing_bases
    
    # Convert to paths
    delta_paths = [ian_bases[base] for base in new_bases]
    
    # Print analysis
    logger.info(f"📊 Analysis Results:")
    logger.info(f"   🆕 New images (need indexing): {len(delta_paths)}")
    logger.info(f"   ✅ Already indexed (different suffix): {len(already_indexed_bases)}")
    logger.info(f"   📈 Total in IAN/resized: {len(ian_bases)}")
    logger.info(f"   📊 Percentage truly new: {len(delta_paths)/len(ian_bases)*100:.1f}%")
    
    return delta_paths


def copy_delta_images_to_db_pictures(delta_paths: List[Path], target_dir: str = "db_pictures_512") -> List[Path]:
    """Copy delta images to db_pictures_512 folder with proper naming"""
    
    logger.info(f"\n📦 Copying {len(delta_paths)} new images to {target_dir}")
    
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    
    copied_paths = []
    skipped_count = 0
    
    for src_path in tqdm(delta_paths, desc="Copying images"):
        # Remove _P02 suffix for consistency with existing files
        src_name = src_path.name
        if '_P' in src_name:
            base_name = src_name.split('_P')[0]
            # Add back the extension (lowercase for consistency)
            dst_name = base_name + '.jpg'
        else:
            dst_name = src_name.lower()
        
        dst_path = target_path / dst_name
        
        # Skip if already exists
        if dst_path.exists():
            skipped_count += 1
            copied_paths.append(dst_path)  # Still add to list for indexing
            continue
        
        try:
            shutil.copy2(src_path, dst_path)
            copied_paths.append(dst_path)
        except Exception as e:
            logger.error(f"❌ Error copying {src_path.name}: {e}")
    
    logger.info(f"✅ Successfully copied {len(copied_paths) - skipped_count} images")
    if skipped_count > 0:
        logger.info(f"⏭️  Skipped {skipped_count} already existing images")
    
    return copied_paths


def index_delta_images(delta_paths: List[Path]) -> tuple:
    """Index delta images using LoRA model"""
    
    logger.info(f"\n🤖 Indexing {len(delta_paths)} delta images...")
    
    # Import indexing modules
    sys.path.append('indexing_script_fast')
    from gme_model import gme_model
    
    # Load LoRA model with checkpoint 1095
    logger.info("📦 Loading GME model with LoRA checkpoint 1095...")
    if not gme_model.load_model("gme-Qwen2-VL-7B-Instruct", checkpoint="1095"):
        logger.error("❌ Failed to load GME model")
        return None, None, None
    
    logger.info("✅ Model loaded successfully")
    
    # Process images
    embeddings = []
    processed_paths = []
    failed_count = 0
    
    # Adaptive batch size based on GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        batch_size = 16 if gpu_memory > 30 else 8
    else:
        batch_size = 4
    
    logger.info(f"🎯 Using batch size: {batch_size}")
    
    # Process in batches
    for i in tqdm(range(0, len(delta_paths), batch_size), desc="Indexing batches"):
        batch_paths = delta_paths[i:i+batch_size]
        
        for path in batch_paths:
            try:
                embedding = gme_model.encode_image(str(path))
                if embedding is not None:
                    embeddings.append(embedding)
                    processed_paths.append(str(path))
                else:
                    failed_count += 1
                    logger.warning(f"⚠️  Failed to encode: {path.name}")
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ Error encoding {path.name}: {e}")
        
        # Clear GPU cache periodically
        if torch.cuda.is_available() and i % 100 == 0:
            torch.cuda.empty_cache()
    
    if not embeddings:
        logger.error("❌ No embeddings generated!")
        return None, None, None
    
    logger.info(f"✅ Generated {len(embeddings)} embeddings ({failed_count} failures)")
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Create FAISS index
    dimension = embeddings_array.shape[1]
    delta_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_array)
    delta_index.add(embeddings_array)
    
    logger.info(f"✅ Created delta index with {delta_index.ntotal} vectors")
    
    return delta_index, embeddings_array, processed_paths


def merge_indexes(existing_index_path: str, delta_index, delta_embeddings, delta_paths) -> str:
    """Merge delta index with existing index"""
    
    logger.info("\n🔄 Merging indexes...")
    
    # Load existing index and data
    logger.info("📖 Loading existing index...")
    existing_index = faiss.read_index(existing_index_path)
    existing_embeddings = np.load(existing_index_path.replace('.faiss', '_embeddings.npy'))
    
    with open(existing_index_path.replace('.faiss', '_metadata.json'), 'r') as f:
        existing_metadata = json.load(f)
    
    logger.info(f"📊 Existing index: {existing_index.ntotal} vectors")
    logger.info(f"📊 Delta index: {delta_index.ntotal} vectors")
    
    # Create merged data
    merged_embeddings = np.concatenate([existing_embeddings, delta_embeddings])
    merged_paths = existing_metadata['image_paths'] + delta_paths
    
    # Create new merged index
    dimension = existing_index.d
    merged_index = faiss.IndexFlatIP(dimension)
    
    # Add all embeddings to the new index
    logger.info("🔄 Building merged index...")
    merged_index.add(merged_embeddings)
    
    # Save merged index
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_base = f"indexes/v11_1095_db_pictures_512_merged_{timestamp}"
    
    logger.info(f"💾 Saving merged index as: {merged_base}")
    faiss.write_index(merged_index, f"{merged_base}.faiss")
    np.save(f"{merged_base}_embeddings.npy", merged_embeddings)
    
    # Save metadata
    merged_metadata = {
        'image_paths': merged_paths,
        'total_embeddings': len(merged_paths),
        'embedding_dimension': dimension,
        'created_at': datetime.now().isoformat(),
        'existing_count': existing_index.ntotal,
        'delta_count': delta_index.ntotal,
        'merged_count': merged_index.ntotal,
        'model': 'gme-Qwen2-VL-7B-Instruct',
        'checkpoint': '1095'
    }
    
    with open(f"{merged_base}_metadata.json", 'w') as f:
        json.dump(merged_metadata, f, indent=2)
    
    logger.info(f"✅ Merged index created: {merged_base}")
    logger.info(f"   Total vectors: {merged_index.ntotal}")
    
    return merged_base


def main():
    """Main delta indexing workflow"""
    
    logger.info("🚀 SMART DELTA INDEXING FOR IAN/RESIZED")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Load existing indexed images
    existing_bases, existing_full = load_existing_indexed_images()
    if not existing_bases:
        logger.error("❌ Could not load existing index")
        return
    
    # Step 2: Scan IAN/resized folder
    ian_bases = scan_ian_resized_folder()
    if not ian_bases:
        logger.error("❌ No images found in IAN/resized")
        return
    
    # Step 3: Find truly new images
    delta_paths = find_delta_images(existing_bases, ian_bases)
    
    if not delta_paths:
        logger.info("✅ All images are already indexed! No delta indexing needed.")
        return
    
    # Ask for confirmation
    logger.info(f"\n⚠️  About to process {len(delta_paths)} truly new images")
    logger.info("   This will:")
    logger.info("   1. Copy images to db_pictures_512 (with proper naming)")
    logger.info("   2. Index them using GME + LoRA checkpoint 1095")
    logger.info("   3. Merge with existing index")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        logger.info("❌ Aborted by user")
        return
    
    # Step 4: Copy delta images to db_pictures_512
    copied_paths = copy_delta_images_to_db_pictures(delta_paths)
    
    if not copied_paths:
        logger.error("❌ No images were copied successfully")
        return
    
    # Step 5: Index delta images
    delta_index, delta_embeddings, indexed_paths = index_delta_images(copied_paths)
    
    if delta_index is None:
        logger.error("❌ Failed to index delta images")
        return
    
    # Step 6: Merge with existing index
    merged_path = merge_indexes(
        "indexes/v11_1095_db_pictures_512.faiss",
        delta_index,
        delta_embeddings,
        indexed_paths
    )
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info(f"\n🎉 SMART DELTA INDEXING COMPLETE!")
    logger.info("=" * 40)
    logger.info(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
    logger.info(f"🆕 New images indexed: {len(indexed_paths)}")
    logger.info(f"💾 Merged index saved: {merged_path}")
    logger.info(f"\n💡 Next steps:")
    logger.info(f"   1. Test the merged index with a few searches")
    logger.info(f"   2. Update app configuration to use: {merged_path}")
    logger.info(f"   3. Once verified, you can remove the old index")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 