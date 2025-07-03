#!/usr/bin/env python3
"""
Fast Delta Indexing for IAN/resized using high-performance LoRA approach
Filtered version: Only indexes images starting with "1" or "3"
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import shutil
import sys
from pathlib import Path
from typing import Set, List, Dict, Tuple
import numpy as np
import faiss
import time
from datetime import datetime
from tqdm import tqdm
import logging
import torch
import gc

# Add the indexing_script_fast to path
sys.path.append('indexing_script_fast')
from lora_similarity_engine import LoRAImageSimilarityEngine

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class AdaptiveBatchProcessor:
    """Adaptive batch sizing for optimal GPU utilization"""
    
    def __init__(self, initial_batch_size: int = 16, memory_threshold: float = 0.90):
        self.current_batch_size = initial_batch_size
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = 128
        self.min_batch_size = 1
        self.memory_threshold = memory_threshold
        self.successful_batches = 0
        self.oom_count = 0
        
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on GPU memory"""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            memory_used = allocated_memory / total_memory if total_memory > 0 else 0
            
            logger.info(f"🔥 GPU Memory: {memory_used:.1%} ({allocated_memory/1e9:.1f}GB / {total_memory/1e9:.1f}GB)")
            
            # Reduce if close to limit
            if memory_used > self.memory_threshold:
                self.current_batch_size = max(int(self.current_batch_size * 0.8), self.min_batch_size)
                logger.info(f"🔽 Reducing batch size to {self.current_batch_size}")
                torch.cuda.empty_cache()
                
            # Scale up when memory is available
            elif memory_used < 0.7 and self.successful_batches >= 2:
                if self.current_batch_size < self.max_batch_size:
                    increment = min(16, self.current_batch_size // 2)
                    self.current_batch_size = min(self.current_batch_size + increment, self.max_batch_size)
                    logger.info(f"🚀 Scaling UP to {self.current_batch_size}")
                    self.successful_batches = 0
                
        return self.current_batch_size
    
    def report_success(self):
        self.successful_batches += 1
                
    def report_oom(self):
        self.oom_count += 1
        old_size = self.current_batch_size
        self.current_batch_size = max(int(self.current_batch_size * 0.7), self.min_batch_size)
        logger.info(f"💥 OOM! Reducing from {old_size} to {self.current_batch_size}")
        torch.cuda.empty_cache()
        gc.collect()


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
    """Find truly new images not in the existing index - FILTERED FOR 1 or 3"""
    
    logger.info("\n🔍 SMART DELTA ANALYSIS (FILTERED: 1 or 3)")
    logger.info("=" * 50)
    
    ian_base_set = set(ian_bases.keys())
    
    # Find new bases
    new_bases = ian_base_set - existing_bases
    already_indexed_bases = ian_base_set & existing_bases
    
    # Convert to paths and FILTER for images starting with 1 or 3
    all_delta_paths = [ian_bases[base] for base in new_bases]
    delta_paths = []
    
    for path in all_delta_paths:
        # Check if filename starts with 1 or 3
        if path.name[0] in ['1', '3']:
            delta_paths.append(path)
    
    # Print analysis
    logger.info(f"📊 Analysis Results:")
    logger.info(f"   🆕 Total new images: {len(all_delta_paths)}")
    logger.info(f"   🔢 Images starting with 1 or 3: {len(delta_paths)}")
    logger.info(f"   ✅ Already indexed (different suffix): {len(already_indexed_bases)}")
    logger.info(f"   📈 Total in IAN/resized: {len(ian_bases)}")
    logger.info(f"   📊 Percentage filtered (1 or 3): {len(delta_paths)/len(all_delta_paths)*100:.1f}%" if all_delta_paths else "N/A")
    
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


def fast_index_delta_images(delta_paths: List[Path], checkpoint: str = "1095") -> tuple:
    """Index delta images using high-performance LoRA approach"""
    
    logger.info(f"\n🔥 FAST INDEXING {len(delta_paths)} delta images...")
    logger.info("🔥" * 30)
    
    # Setup paths
    base_model_path = "gme-Qwen2-VL-7B-Instruct"
    lora_path = f"loras/v11-20250620-105815/checkpoint-{checkpoint}"
    
    # Create LoRA engine
    logger.info("📥 Loading LoRA model for fast indexing...")
    engine = LoRAImageSimilarityEngine(
        base_model_path=base_model_path,
        lora_path=lora_path
    )
    
    # Adaptive batch processor
    batch_processor = AdaptiveBatchProcessor(initial_batch_size=16)
    
    # Process images
    all_embeddings = []
    all_paths = []
    failed_count = 0
    
    start_time = time.time()
    
    with tqdm(total=len(delta_paths), desc="🚀 Fast LoRA Indexing") as pbar:
        i = 0
        while i < len(delta_paths):
            batch_start = time.time()
            
            # Get optimal batch size
            batch_size = batch_processor.get_optimal_batch_size()
            batch = delta_paths[i:i + batch_size]
            
            try:
                # Process batch - images are processed individually but efficiently
                batch_embeddings = []
                batch_paths = []
                
                for img_path in batch:
                    try:
                        embedding = engine.get_image_embedding(str(img_path))
                        if embedding is not None:
                            batch_embeddings.append(embedding)
                            batch_paths.append(str(img_path))
                    except Exception as e:
                        logger.error(f"Error processing {img_path.name}: {e}")
                        failed_count += 1
                        continue
                
                if batch_embeddings:
                    all_embeddings.extend(batch_embeddings)
                    all_paths.extend(batch_paths)
                    
                    # Calculate throughput
                    batch_time = time.time() - batch_start
                    batch_throughput = len(batch_embeddings) / batch_time
                    batch_processor.report_success()
                    
                    pbar.update(len(batch))
                    pbar.set_postfix({
                        'batch': batch_size,
                        'throughput': f"{batch_throughput:.1f}/s",
                        'failures': failed_count
                    })
                
                i += len(batch)
                
                # Clear cache periodically
                if i % 500 == 0:
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    batch_processor.report_oom()
                    # Don't increment i, retry with smaller batch
                    continue
                else:
                    raise e
    
    if not all_embeddings:
        logger.error("❌ No embeddings generated!")
        return None, None, None
    
    total_time = time.time() - start_time
    
    logger.info(f"\n✅ Fast indexing complete!")
    logger.info(f"   📊 Images processed: {len(all_embeddings)}")
    logger.info(f"   ❌ Failed: {failed_count}")
    logger.info(f"   ⏱️  Total time: {total_time:.1f}s")
    logger.info(f"   🚀 Throughput: {len(all_embeddings)/total_time:.1f} images/sec")
    
    # Convert to numpy array
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    
    # Create FAISS index
    dimension = embeddings_array.shape[1]
    delta_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_array)
    delta_index.add(embeddings_array)
    
    logger.info(f"✅ Created delta index with {delta_index.ntotal} vectors")
    
    return delta_index, embeddings_array, all_paths


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
    
    # Save merged index with suffix indicating filter
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_base = f"indexes/v11_1095_db_pictures_512_merged_13_{timestamp}"
    
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
        'checkpoint': '1095',
        'indexing_method': 'fast_lora',
        'filter': 'images_starting_with_1_or_3'
    }
    
    with open(f"{merged_base}_metadata.json", 'w') as f:
        json.dump(merged_metadata, f, indent=2)
    
    logger.info(f"✅ Merged index created: {merged_base}")
    logger.info(f"   Total vectors: {merged_index.ntotal}")
    
    return merged_base


def main():
    """Main delta indexing workflow with fast performance - FILTERED FOR 1 or 3"""
    
    logger.info("🚀 FAST DELTA INDEXING FOR IAN/RESIZED (FILTERED: 1 or 3)")
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
    
    # Step 3: Find truly new images (filtered for 1 or 3)
    delta_paths = find_delta_images(existing_bases, ian_bases)
    
    if not delta_paths:
        logger.info("✅ No images starting with 1 or 3 need indexing!")
        return
    
    # Ask for confirmation
    logger.info(f"\n⚠️  About to process {len(delta_paths)} images starting with 1 or 3")
    logger.info("   This will:")
    logger.info("   1. Copy images to db_pictures_512 (with proper naming)")
    logger.info("   2. Fast index using high-performance LoRA approach")
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
    
    # Step 5: Fast index delta images
    delta_index, delta_embeddings, indexed_paths = fast_index_delta_images(copied_paths)
    
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
    logger.info(f"\n🎉 FAST DELTA INDEXING COMPLETE (FILTERED: 1 or 3)!")
    logger.info("=" * 40)
    logger.info(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
    logger.info(f"🆕 New images indexed: {len(indexed_paths)}")
    logger.info(f"💾 Merged index saved: {merged_path}")
    logger.info(f"🚀 Average throughput: {len(indexed_paths)/(elapsed_time):.1f} images/second")
    logger.info(f"\n💡 Next steps:")
    logger.info(f"   1. Test the merged index with a few searches")
    logger.info(f"   2. Update app configuration to use: {merged_path}")
    logger.info(f"   3. Once verified, you can remove the old index")
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 