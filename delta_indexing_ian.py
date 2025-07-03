#!/usr/bin/env python3
"""
Delta Indexing for IAN/resized images
Indexes only new images not already in v11_1095_db_pictures_512 index
"""

import json
import shutil
import sys
from pathlib import Path
from typing import Set, List, Dict
import os
import numpy as np
import faiss
import time
from datetime import datetime
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_existing_indexed_images(metadata_file: str = "indexes/v11_1095_db_pictures_512_metadata.json") -> Set[str]:
    """Load list of images already in the existing index"""
    
    if not Path(metadata_file).exists():
        logger.error(f"❌ Existing index metadata not found: {metadata_file}")
        return set()
    
    logger.info(f"📖 Loading existing index metadata: {metadata_file}")
    
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
        
        existing_images = set()
        image_paths = metadata.get('image_paths', [])
        
        for path in image_paths:
            # Extract filename from path
            filename = Path(path).name
            existing_images.add(filename)
        
        logger.info(f"✅ Found {len(existing_images)} images in existing index")
        return existing_images
        
    except Exception as e:
        logger.error(f"❌ Error loading existing index: {e}")
        return set()


def scan_ian_resized_folder(ian_folder: str = "IAN/resized") -> Dict[str, Path]:
    """Scan IAN/resized folder and return filename->path mapping"""
    
    ian_path = Path(ian_folder)
    if not ian_path.exists():
        logger.error(f"❌ IAN folder not found: {ian_folder}")
        return {}
    
    logger.info(f"📁 Scanning: {ian_folder}")
    
    # Build lookup table
    lookup_table = {}
    
    # Look for all image files
    for img_file in ian_path.glob("*"):
        if img_file.is_file() and img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            lookup_table[img_file.name] = img_file
    
    logger.info(f"✅ Found {len(lookup_table)} images in IAN/resized")
    return lookup_table


def find_delta_images(existing_images: Set[str], ian_lookup: Dict[str, Path]) -> List[Path]:
    """Find images in IAN/resized that are not in the existing index"""
    
    logger.info("\n🔍 DELTA ANALYSIS")
    logger.info("=" * 50)
    
    ian_images = set(ian_lookup.keys())
    
    # Find new images (in IAN but not in existing index)
    new_images = ian_images - existing_images
    
    # Find already indexed images
    already_indexed = ian_images & existing_images
    
    # Convert to paths
    delta_paths = [ian_lookup[filename] for filename in new_images]
    
    # Print analysis
    logger.info(f"📊 Analysis Results:")
    logger.info(f"   🆕 New images (need indexing): {len(new_images)}")
    logger.info(f"   ✅ Already indexed: {len(already_indexed)}")
    logger.info(f"   📈 Total in IAN/resized: {len(ian_images)}")
    logger.info(f"   📉 Total in existing index: {len(existing_images)}")
    
    if len(new_images) < 20:
        logger.info("\n🆕 New images to index:")
        for i, img in enumerate(sorted(new_images), 1):
            logger.info(f"   {i:2d}. {img}")
    
    return delta_paths


def move_delta_images_to_db_pictures(delta_paths: List[Path], target_dir: str = "db_pictures_512") -> List[Path]:
    """Move delta images to db_pictures_512 folder"""
    
    logger.info(f"\n📦 Moving {len(delta_paths)} new images to {target_dir}")
    
    target_path = Path(target_dir)
    target_path.mkdir(exist_ok=True)
    
    moved_paths = []
    
    for src_path in tqdm(delta_paths, desc="Moving images"):
        dst_path = target_path / src_path.name
        
        try:
            # Copy instead of move to preserve originals
            shutil.copy2(src_path, dst_path)
            moved_paths.append(dst_path)
        except Exception as e:
            logger.error(f"❌ Error moving {src_path.name}: {e}")
    
    logger.info(f"✅ Successfully moved {len(moved_paths)} images")
    return moved_paths


def index_delta_images(delta_paths: List[Path]) -> tuple:
    """Index delta images using LoRA model"""
    
    logger.info(f"\n🤖 Indexing {len(delta_paths)} delta images...")
    
    # Import indexing modules
    sys.path.append('indexing_script_fast')
    from gme_model import gme_model
    
    # Load LoRA model with checkpoint 1095
    if not gme_model.load_model("gme-Qwen2-VL-7B-Instruct", checkpoint="1095"):
        logger.error("❌ Failed to load GME model")
        return None, None, None
    
    # Process images
    embeddings = []
    processed_paths = []
    
    batch_size = 8
    for i in tqdm(range(0, len(delta_paths), batch_size), desc="Indexing"):
        batch_paths = delta_paths[i:i+batch_size]
        
        for path in batch_paths:
            try:
                embedding = gme_model.encode_image(str(path))
                if embedding is not None:
                    embeddings.append(embedding)
                    processed_paths.append(str(path))
            except Exception as e:
                logger.error(f"❌ Error encoding {path.name}: {e}")
    
    if not embeddings:
        logger.error("❌ No embeddings generated!")
        return None, None, None
    
    logger.info(f"✅ Generated {len(embeddings)} embeddings")
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Create FAISS index
    dimension = embeddings_array.shape[1]
    delta_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_array)
    delta_index.add(embeddings_array)
    
    return delta_index, embeddings_array, processed_paths


def merge_indexes(existing_index_path: str, delta_index, delta_embeddings, delta_paths) -> str:
    """Merge delta index with existing index"""
    
    logger.info("\n🔄 Merging indexes...")
    
    # Load existing index and data
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
    merged_index.add(merged_embeddings)
    
    # Save merged index
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_base = f"indexes/v11_1095_db_pictures_512_merged_{timestamp}"
    
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
        'merged_count': merged_index.ntotal
    }
    
    with open(f"{merged_base}_metadata.json", 'w') as f:
        json.dump(merged_metadata, f, indent=2)
    
    logger.info(f"✅ Merged index created: {merged_base}")
    logger.info(f"   Total vectors: {merged_index.ntotal}")
    
    return merged_base


def main():
    """Main delta indexing workflow"""
    
    logger.info("🚀 DELTA INDEXING FOR IAN/RESIZED")
    logger.info("=" * 60)
    
    start_time = time.time()
    
    # Step 1: Load existing indexed images
    existing_images = load_existing_indexed_images()
    if not existing_images:
        logger.error("❌ Could not load existing index")
        return
    
    # Step 2: Scan IAN/resized folder
    ian_lookup = scan_ian_resized_folder()
    if not ian_lookup:
        logger.error("❌ No images found in IAN/resized")
        return
    
    # Step 3: Find delta images
    delta_paths = find_delta_images(existing_images, ian_lookup)
    
    if not delta_paths:
        logger.info("✅ All images are already indexed! No delta indexing needed.")
        return
    
    # Ask for confirmation
    logger.info(f"\n⚠️  About to process {len(delta_paths)} new images")
    response = input("Continue? (y/n): ")
    if response.lower() != 'y':
        logger.info("❌ Aborted by user")
        return
    
    # Step 4: Move delta images to db_pictures_512
    moved_paths = move_delta_images_to_db_pictures(delta_paths)
    
    # Step 5: Index delta images
    delta_index, delta_embeddings, indexed_paths = index_delta_images(moved_paths)
    
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
    logger.info(f"\n🎉 DELTA INDEXING COMPLETE!")
    logger.info("=" * 40)
    logger.info(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
    logger.info(f"🆕 New images indexed: {len(delta_paths)}")
    logger.info(f"💾 Merged index saved: {merged_path}")
    logger.info(f"\n💡 Next steps:")
    logger.info(f"   1. Update app configuration to use the new merged index")
    logger.info(f"   2. Test the search functionality")
    logger.info(f"   3. Consider removing old index files if everything works")


if __name__ == "__main__":
    main() 