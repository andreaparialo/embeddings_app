#!/usr/bin/env python3
"""
Fast GPU-accelerated indexing for GME-Qwen model
Optimized for 4x A100 GPUs with large batch processing
"""

import os
import numpy as np
import pandas as pd
import faiss
import json
import time
import logging
import glob
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import torch
import gc
from PIL import Image

# Set environment variables to suppress warnings
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'

# Import our optimized modules
from gme_model import gme_model
from data_loader import data_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fast_gme_indexing(
    pictures_dir: str = "pictures",
    csv_path: str = "database_results/final_with_aws_shapes_20250625_155822.csv",
    output_dir: str = "indexes",
    checkpoint: str = "1095",
    batch_size: int = 16,  # Will be auto-scaled for multiple GPUs
    index_name: str = None
):
    """
    Fast GPU-accelerated indexing using GME-Qwen model
    """
    
    # Auto-generate index name if not provided
    if index_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        index_name = f"gme_gpu_index_{checkpoint}_{timestamp}"
    
    logger.info("=" * 80)
    logger.info("🚀 GME-QWEN FAST GPU INDEXING")
    logger.info("=" * 80)
    logger.info(f"📁 Pictures directory: {pictures_dir}")
    logger.info(f"📊 CSV path: {csv_path}")
    logger.info(f"💾 Output directory: {output_dir}")
    logger.info(f"🎯 Checkpoint: {checkpoint}")
    logger.info(f"🏷️  Index name: {index_name}")
    
    # Check GPU availability and set conservative batch size
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        logger.info(f"🚀 GPU acceleration enabled: {num_gpus}x {torch.cuda.get_device_name(0)}")
        
        # Conservative batch sizing to avoid OOM - A100s can handle ~8-16 images per batch safely
        # Don't scale by GPU count as the model uses device_map="auto" which handles distribution
        max_safe_batch = 16  # Conservative for vision models
        batch_size = min(batch_size, max_safe_batch)
        logger.info(f"📈 Using safe batch size: {batch_size} (conservative for vision model)")
    else:
        logger.warning("⚠️  No GPU available - using CPU (will be slower)")
    
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV data
    logger.info("\n📖 Loading CSV data...")
    data_loader.load_csv(csv_path)
    csv_filename_roots = set(data_loader.df['filename_root'].astype(str).unique())
    logger.info(f"✅ Loaded {len(data_loader.df)} rows, {len(csv_filename_roots)} unique filename_roots")
    
    # Find all image files
    logger.info("\n🔍 Scanning for images...")
    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    all_image_files = []
    
    for ext in image_extensions:
        all_image_files.extend(glob.glob(os.path.join(pictures_dir, ext)))
    
    logger.info(f"📁 Found {len(all_image_files)} total image files")
    
    # Filter to only images that have CSV entries
    valid_image_paths = []
    filename_root_to_path = {}
    
    for image_path in all_image_files:
        filename = os.path.basename(image_path)
        filename_root = filename.split('_')[0]
        
        if filename_root in csv_filename_roots:
            valid_image_paths.append(image_path)
            filename_root_to_path[filename_root] = image_path
    
    logger.info(f"✅ Found {len(valid_image_paths)} images with CSV entries")
    
    if len(valid_image_paths) == 0:
        logger.error("❌ No valid images found!")
        return False
    
    # Verify LoRA checkpoint exists
    lora_path = f"loras/v11-20250620-105815/checkpoint-{checkpoint}"
    if not os.path.exists(lora_path):
        logger.error(f"❌ LoRA checkpoint not found: {lora_path}")
        logger.info("📁 Available checkpoints:")
        base_lora_dir = "loras/v11-20250620-105815"
        if os.path.exists(base_lora_dir):
            for item in os.listdir(base_lora_dir):
                if item.startswith("checkpoint-"):
                    logger.info(f"   - {item}")
        return False
    
    # Load GME model (base model + LoRA checkpoint)
    logger.info(f"\n🤖 Loading GME base model with LoRA checkpoint {checkpoint}...")
    logger.info(f"📁 Base model: gme-Qwen2-VL-7B-Instruct")
    logger.info(f"🔧 LoRA checkpoint: {lora_path}")
    if not gme_model.load_model("gme-Qwen2-VL-7B-Instruct", checkpoint=checkpoint):
        logger.error("❌ Failed to load GME model with LoRA")
        return False
    
    # Show GPU memory usage after model loading
    if torch.cuda.is_available():
        memory_info = gme_model.get_memory_usage()
        logger.info(f"💾 GPU memory after model loading:")
        for gpu_info in memory_info['gpus']:
            logger.info(f"   GPU {gpu_info['gpu_id']}: {gpu_info['allocated_gb']:.1f}GB / {gpu_info['total_gb']:.1f}GB")
    
    # Process images in batches
    logger.info(f"\n⚡ Starting batch processing ({batch_size} images per batch)...")
    
    all_embeddings = []
    all_image_paths = []
    processed_count = 0
    
    # Estimate time
    if torch.cuda.is_available():
        estimated_time = len(valid_image_paths) / (batch_size * 10)  # ~10 batches per second estimate
        logger.info(f"⏱️  Estimated time: {estimated_time:.1f} minutes")
    
    # Process in batches with adaptive sizing to handle OOM
    current_batch_size = batch_size
    oom_count = 0
    
    with tqdm(total=len(valid_image_paths), desc="🚀 GPU Batch Processing", unit="imgs") as pbar:
        i = 0
        while i < len(valid_image_paths):
            batch_paths = valid_image_paths[i:i + current_batch_size]
            
            try:
                # Load batch of images
                batch_images = []
                batch_filenames = []
                
                for img_path in batch_paths:
                    try:
                        image = Image.open(img_path).convert('RGB')
                        batch_images.append(image)
                        batch_filenames.append(os.path.basename(img_path))
                    except Exception as e:
                        logger.warning(f"⚠️  Error loading {img_path}: {e}")
                        continue
                
                if batch_images:
                    # Clear GPU cache before processing
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Process batch with current batch size
                    batch_embeddings = gme_model.encode_image_batch(
                        images=batch_images,
                        batch_size=len(batch_images)
                    )
                    
                    if batch_embeddings is not None:
                        # Convert to numpy if needed
                        if torch.is_tensor(batch_embeddings):
                            batch_embeddings = batch_embeddings.cpu().numpy()
                        
                        # Add to results
                        for j, embedding in enumerate(batch_embeddings):
                            all_embeddings.append(embedding)
                            all_image_paths.append(batch_filenames[j])
                            processed_count += 1
                        
                        # Reset OOM counter on success
                        oom_count = 0
                
                # Update progress
                pbar.update(len(batch_paths))
                pbar.set_postfix({
                    'processed': processed_count,
                    'batch_size': current_batch_size,
                    'batch': len(batch_images),
                    'GPU_mem': f"{torch.cuda.memory_allocated(0)/1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
                })
                
                # Move to next batch
                i += current_batch_size
                    
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    # Handle OOM by reducing batch size
                    oom_count += 1
                    if current_batch_size > 1:
                        current_batch_size = max(1, current_batch_size // 2)
                        logger.warning(f"⚠️  OOM detected, reducing batch size to {current_batch_size}")
                        
                        # Clear GPU cache
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        
                        # Don't increment i, retry with smaller batch
                        continue
                    else:
                        logger.error(f"❌ OOM with batch size 1, cannot continue")
                        break
                else:
                    logger.error(f"❌ Runtime error: {e}")
                    i += current_batch_size
                    continue
                    
            except Exception as e:
                logger.error(f"❌ Batch processing error: {e}")
                import traceback
                logger.error(traceback.format_exc())
                i += current_batch_size
                continue
    
    if len(all_embeddings) == 0:
        logger.error("❌ No embeddings generated!")
        return False
    
    logger.info(f"✅ Generated {len(all_embeddings)} embeddings")
    
    # Convert to numpy array
    logger.info("\n📊 Converting embeddings to numpy array...")
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"📐 Embeddings shape: {embeddings_array.shape}")
    
    # Create FAISS index
    logger.info("\n⚡ Creating GPU-accelerated FAISS index...")
    dimension = embeddings_array.shape[1]
    
    # Create CPU index first
    cpu_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
    
    # Normalize embeddings for cosine similarity
    faiss.normalize_L2(embeddings_array)
    cpu_index.add(embeddings_array)
    
    logger.info(f"✅ Created CPU index with {cpu_index.ntotal} vectors")
    
    # Move to GPU if available
    if torch.cuda.is_available() and data_loader.use_gpu:
        logger.info("🚀 Moving index to GPU...")
        
        if data_loader.num_gpus > 1:
            # Multi-GPU setup
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = True
            co.usePrecomputed = False
            
            gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co, ngpu=data_loader.num_gpus)
            logger.info(f"✅ Index sharded across {data_loader.num_gpus} GPUs")
        else:
            # Single GPU
            res = faiss.StandardGpuResources()
            gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            logger.info("✅ Index moved to single GPU")
        
        final_index = gpu_index
    else:
        final_index = cpu_index
        logger.info("📍 Using CPU index")
    
    # Save everything
    logger.info(f"\n💾 Saving index and metadata...")
    
    # Save FAISS index (save CPU version for portability)
    faiss.write_index(cpu_index, os.path.join(output_dir, f"{index_name}.faiss"))
    
    # Save embeddings
    np.save(os.path.join(output_dir, f"{index_name}_embeddings.npy"), embeddings_array)
    
    # Create metadata
    metadata = {
        'image_paths': all_image_paths,
        'index_name': index_name,
        'checkpoint': checkpoint,
        'total_embeddings': len(all_embeddings),
        'embedding_dimension': dimension,
        'created_at': datetime.now().isoformat(),
        'processing_time_seconds': time.time() - start_time,
        'gpu_accelerated': torch.cuda.is_available(),
        'num_gpus_used': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    with open(os.path.join(output_dir, f"{index_name}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final summary
    total_time = time.time() - start_time
    
    logger.info("=" * 80)
    logger.info("🎉 INDEXING COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"📊 Total images processed: {len(all_embeddings)}")
    logger.info(f"📐 Embedding dimension: {dimension}")
    logger.info(f"⏱️  Total time: {total_time:.1f} seconds")
    logger.info(f"🚀 Throughput: {len(all_embeddings)/total_time:.1f} images/second")
    logger.info(f"💾 Index saved as: {index_name}")
    logger.info(f"📁 Files created:")
    logger.info(f"   - {index_name}.faiss")
    logger.info(f"   - {index_name}_embeddings.npy") 
    logger.info(f"   - {index_name}_metadata.json")
    
    if torch.cuda.is_available():
        final_memory = gme_model.get_memory_usage()
        logger.info(f"💾 Final GPU memory: {final_memory['total_allocated_gb']:.1f}GB allocated")
    
    logger.info("🎯 Index ready for use in search applications!")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Fast GPU-accelerated GME-Qwen indexing")
    parser.add_argument("--pictures-dir", default="pictures", help="Directory containing images")
    parser.add_argument("--csv-path", default="database_results/final_with_aws_shapes_20250625_155822.csv", help="CSV database path")
    parser.add_argument("--output-dir", default="indexes", help="Output directory")
    parser.add_argument("--checkpoint", default="1095", help="GME LoRA checkpoint to use (680, 1020, 1095)")
    parser.add_argument("--batch-size", type=int, default=16, help="Base batch size (auto-scaled for GPUs)")
    parser.add_argument("--index-name", help="Custom index name (auto-generated if not provided)")
    
    args = parser.parse_args()
    
    success = fast_gme_indexing(
        pictures_dir=args.pictures_dir,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        index_name=args.index_name
    )
    
    if success:
        print("\n🎉 SUCCESS: Index created successfully!")
        print("You can now use this index in your search applications.")
    else:
        print("\n❌ FAILED: Index creation failed!")
        exit(1) 