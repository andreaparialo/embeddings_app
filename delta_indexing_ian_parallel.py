#!/usr/bin/env python3
"""
Parallel Delta Indexing for IAN/resized using 8 GPUs
Splits images across GPUs and runs separate processes
"""

import os
# Disable nested parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
import multiprocessing as mp

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


def index_chunk_on_gpu(gpu_id: int, image_paths: List[Path], chunk_id: int, total_chunks: int, checkpoint: str = "1095") -> Tuple[str, int]:
    """Index a chunk of images on a specific GPU"""
    
    # Set GPU for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Import torch after setting CUDA_VISIBLE_DEVICES
    import torch
    
    # Disable any internal parallelism
    torch.set_num_threads(1)
    
    # Import here to ensure each process loads its own copy
    sys.path.append('indexing_script_fast')
    from lora_similarity_engine import LoRAImageSimilarityEngine
    
    print(f"\n🎮 GPU {gpu_id} - Starting chunk {chunk_id}/{total_chunks} with {len(image_paths)} images", flush=True)
    
    # Setup paths
    base_model_path = "gme-Qwen2-VL-7B-Instruct"
    lora_path = f"loras/v11-20250620-105815/checkpoint-{checkpoint}"
    
    # Create LoRA engine for this GPU
    engine = LoRAImageSimilarityEngine(
        base_model_path=base_model_path,
        lora_path=lora_path,
        device="cuda"  # Will use the GPU specified by CUDA_VISIBLE_DEVICES
    )
    
    print(f"GPU {gpu_id} - Model loaded, starting indexing...", flush=True)
    
    # Process images
    embeddings = []
    processed_paths = []
    failed_count = 0
    
    start_time = time.time()
    last_update_time = start_time
    
    # Process with decent batch size since we have dedicated GPU
    batch_size = 16
    
    for i in range(0, len(image_paths), batch_size):
        batch = image_paths[i:i+batch_size]
        
        for img_path in batch:
            try:
                embedding = engine.get_image_embedding(str(img_path))
                if embedding is not None:
                    embeddings.append(embedding)
                    processed_paths.append(str(img_path))
            except Exception as e:
                print(f"GPU {gpu_id} - Error processing {img_path.name}: {e}", flush=True)
                failed_count += 1
                continue
        
        # Progress update every 5 seconds or 100 images
        processed = min(i + batch_size, len(image_paths))
        current_time = time.time()
        
        if current_time - last_update_time >= 5.0 or processed % 100 == 0:
            elapsed = current_time - start_time
            throughput = processed / elapsed if elapsed > 0 else 0
            eta = (len(image_paths) - processed) / throughput if throughput > 0 else 0
            print(f"GPU {gpu_id} - Progress: {processed}/{len(image_paths)} ({processed/len(image_paths)*100:.1f}%) | "
                  f"Speed: {throughput:.1f} img/s | ETA: {eta/60:.1f} min", flush=True)
            last_update_time = current_time
        
        # Clear cache periodically
        if i % 100 == 0:
            torch.cuda.empty_cache()
    
    if not embeddings:
        print(f"GPU {gpu_id} - ❌ No embeddings generated!", flush=True)
        return None, gpu_id
    
    total_time = time.time() - start_time
    
    print(f"GPU {gpu_id} - ✅ Completed!", flush=True)
    print(f"GPU {gpu_id} -    Images: {len(embeddings)}", flush=True)
    print(f"GPU {gpu_id} -    Failed: {failed_count}", flush=True)
    print(f"GPU {gpu_id} -    Time: {total_time:.1f}s", flush=True)
    print(f"GPU {gpu_id} -    Throughput: {len(embeddings)/total_time:.1f} img/s", flush=True)
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    # Create FAISS index for this chunk
    dimension = embeddings_array.shape[1]
    chunk_index = faiss.IndexFlatIP(dimension)
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings_array)
    chunk_index.add(embeddings_array)
    
    # Save chunk index
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chunk_filename = f"indexes/chunk_gpu{gpu_id}_{timestamp}"
    
    faiss.write_index(chunk_index, f"{chunk_filename}.faiss")
    np.save(f"{chunk_filename}_embeddings.npy", embeddings_array)
    
    # Save metadata
    metadata = {
        'image_paths': processed_paths,
        'gpu_id': gpu_id,
        'chunk_id': chunk_id,
        'total_chunks': total_chunks,
        'embeddings_count': len(embeddings),
        'failed_count': failed_count,
        'processing_time': total_time,
        'throughput': len(embeddings)/total_time
    }
    
    with open(f"{chunk_filename}_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return chunk_filename, gpu_id


def merge_chunk_indexes(chunk_files: List[str], existing_index_path: str) -> str:
    """Merge all chunk indexes with the existing index"""
    
    logger.info("\n🔄 Merging all chunk indexes...")
    
    # Load existing index
    logger.info("📖 Loading existing index...")
    existing_index = faiss.read_index(existing_index_path)
    existing_embeddings = np.load(existing_index_path.replace('.faiss', '_embeddings.npy'))
    
    with open(existing_index_path.replace('.faiss', '_metadata.json'), 'r') as f:
        existing_metadata = json.load(f)
    
    # Collect all chunk data
    all_chunk_embeddings = [existing_embeddings]
    all_chunk_paths = existing_metadata['image_paths']
    
    for chunk_file in chunk_files:
        if chunk_file is None:
            continue
            
        # Load chunk embeddings
        chunk_embeddings = np.load(f"{chunk_file}_embeddings.npy")
        all_chunk_embeddings.append(chunk_embeddings)
        
        # Load chunk metadata
        with open(f"{chunk_file}_metadata.json", 'r') as f:
            chunk_metadata = json.load(f)
            all_chunk_paths.extend(chunk_metadata['image_paths'])
        
        # Clean up chunk files
        os.remove(f"{chunk_file}.faiss")
        os.remove(f"{chunk_file}_embeddings.npy")
        os.remove(f"{chunk_file}_metadata.json")
    
    # Merge all embeddings
    merged_embeddings = np.concatenate(all_chunk_embeddings)
    
    # Create merged index
    dimension = existing_index.d
    merged_index = faiss.IndexFlatIP(dimension)
    merged_index.add(merged_embeddings)
    
    # Save merged index
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_base = f"indexes/v11_1095_db_pictures_512_merged_{timestamp}"
    
    logger.info(f"💾 Saving merged index as: {merged_base}")
    faiss.write_index(merged_index, f"{merged_base}.faiss")
    np.save(f"{merged_base}_embeddings.npy", merged_embeddings)
    
    # Save metadata
    merged_metadata = {
        'image_paths': all_chunk_paths,
        'total_embeddings': len(all_chunk_paths),
        'embedding_dimension': dimension,
        'created_at': datetime.now().isoformat(),
        'existing_count': existing_index.ntotal,
        'delta_count': len(all_chunk_paths) - existing_index.ntotal,
        'merged_count': merged_index.ntotal,
        'model': 'gme-Qwen2-VL-7B-Instruct',
        'checkpoint': '1095',
        'indexing_method': 'parallel_gpu'
    }
    
    with open(f"{merged_base}_metadata.json", 'w') as f:
        json.dump(merged_metadata, f, indent=2)
    
    logger.info(f"✅ Merged index created: {merged_base}")
    logger.info(f"   Total vectors: {merged_index.ntotal}")
    
    return merged_base


def process_chunk_wrapper(gpu_id, chunk, chunk_id, total_chunks, results_list):
    """Wrapper function for multiprocessing that stores results in shared list"""
    try:
        result = index_chunk_on_gpu(gpu_id, chunk, chunk_id, total_chunks, "1095")
        results_list.append(result)
    except Exception as e:
        logger.error(f"❌ GPU {gpu_id} process failed: {e}")
        import traceback
        traceback.print_exc()
        results_list.append((None, gpu_id))


def main():
    """Main parallel delta indexing workflow"""
    
    logger.info("🚀 PARALLEL DELTA INDEXING FOR IAN/RESIZED (8 GPUs)")
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
    logger.info("   2. Split across 8 GPUs for parallel indexing")
    logger.info("   3. Merge all chunks with existing index")
    
    response = input("\nContinue? (y/n): ")
    if response.lower() != 'y':
        logger.info("❌ Aborted by user")
        return
    
    # Step 4: Copy delta images to db_pictures_512
    copied_paths = copy_delta_images_to_db_pictures(delta_paths)
    
    if not copied_paths:
        logger.error("❌ No images were copied successfully")
        return
    
    # Step 5: Split images across 8 GPUs
    num_gpus = 8
    chunk_size = len(copied_paths) // num_gpus
    remainder = len(copied_paths) % num_gpus
    
    chunks = []
    start_idx = 0
    
    for i in range(num_gpus):
        # Add one extra image to some chunks to handle remainder
        current_chunk_size = chunk_size + (1 if i < remainder else 0)
        end_idx = start_idx + current_chunk_size
        
        chunk = copied_paths[start_idx:end_idx]
        chunks.append(chunk)
        
        logger.info(f"🎮 GPU {i}: {len(chunk)} images")
        start_idx = end_idx
    
    # Step 6: Process chunks in parallel
    logger.info(f"\n🚀 Starting parallel processing on {num_gpus} GPUs...")
    
    # Use Process instead of Pool to avoid daemon processes
    processes = []
    chunk_results = mp.Manager().list()
    
    for gpu_id, chunk in enumerate(chunks):
        # Create non-daemon process
        p = mp.Process(
            target=process_chunk_wrapper,
            args=(gpu_id, chunk, gpu_id+1, num_gpus, chunk_results)
        )
        p.daemon = False  # Explicitly set as non-daemon
        p.start()
        processes.append(p)
    
    # Wait for all processes to complete
    for p in processes:
        p.join(timeout=3600)  # 1 hour timeout
        if p.is_alive():
            logger.error(f"❌ Process {p.name} timed out")
            p.terminate()
            p.join()
    
    # Extract results
    chunk_files = []
    for result in chunk_results:
        chunk_file, gpu_id = result
        chunk_files.append(chunk_file)
        if chunk_file:
            logger.info(f"✅ GPU {gpu_id} completed")
    
    # Step 7: Merge all chunks
    valid_chunks = [f for f in chunk_files if f is not None]
    if not valid_chunks:
        logger.error("❌ No chunks were successfully created")
        return
    
    merged_path = merge_chunk_indexes(valid_chunks, "indexes/v11_1095_db_pictures_512.faiss")
    
    # Summary
    elapsed_time = time.time() - start_time
    logger.info(f"\n🎉 PARALLEL DELTA INDEXING COMPLETE!")
    logger.info("=" * 40)
    logger.info(f"⏱️  Total time: {elapsed_time/60:.1f} minutes")
    logger.info(f"🆕 New images indexed: {len(copied_paths)}")
    logger.info(f"🎮 GPUs used: {num_gpus}")
    logger.info(f"💾 Merged index saved: {merged_path}")
    logger.info(f"🚀 Average throughput: {len(copied_paths)/(elapsed_time):.1f} images/second")
    logger.info(f"\n💡 Next steps:")
    logger.info(f"   1. Test the merged index with a few searches")
    logger.info(f"   2. Update app configuration to use: {merged_path}")
    logger.info(f"   3. Once verified, you can remove the old index")


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main() 