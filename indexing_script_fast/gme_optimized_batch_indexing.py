#!/usr/bin/env python3
"""
Optimized GME-Qwen batch indexing using native batch processing
This version properly uses the model's get_image_embeddings method for true batch processing
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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from typing import List, Tuple, Optional

# Set environment variables for optimal performance
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OptimizedBatchProcessor:
    """Optimized batch processor using native model batch processing"""
    
    def __init__(self, model_path: str = "gme-Qwen2-VL-7B-Instruct", checkpoint: str = "1095"):
        self.model_path = model_path
        self.checkpoint = checkpoint
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_gpus = torch.cuda.device_count() if self.device == "cuda" else 0
        
        # Optimize PyTorch settings
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            
    def load_model(self):
        """Load the model with optimizations"""
        from transformers import AutoModel
        from peft import PeftModel
        
        logger.info(f"Loading model on {self.num_gpus} GPUs...")
        
        # Load base model
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            use_fast=True,
            low_cpu_mem_usage=True
        )
        
        # Load LoRA
        lora_path = f"loras/v11-20250620-105815/checkpoint-{self.checkpoint}"
        if os.path.exists(lora_path):
            logger.info(f"Loading LoRA checkpoint: {self.checkpoint}")
            self.model = PeftModel.from_pretrained(self.model, lora_path)
            
        self.model.eval()
        logger.info("Model loaded successfully!")
        
    def process_batch(self, image_paths: List[str]) -> Tuple[List[np.ndarray], List[str]]:
        """Process a batch of images using native batch processing"""
        
        # Load images in parallel
        with ThreadPoolExecutor(max_workers=8) as executor:
            images_and_paths = list(executor.map(self._load_image_with_path, image_paths))
        
        # Filter out failed loads
        valid_data = [(img, path) for img, path in images_and_paths if img is not None]
        
        if not valid_data:
            return [], []
            
        images, paths = zip(*valid_data)
        
        # Use the model's native batch processing
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                # This is the key - using get_image_embeddings with a list of images
                embeddings = self.model.get_image_embeddings(images=list(images))
                
        # Convert to numpy
        if torch.is_tensor(embeddings):
            embeddings_np = embeddings.cpu().numpy()
        else:
            embeddings_np = embeddings
            
        return list(embeddings_np), list(paths)
    
    def _load_image_with_path(self, path: str) -> Tuple[Optional[Image.Image], str]:
        """Load image and return with its path"""
        try:
            img = Image.open(path).convert('RGB')
            return img, os.path.basename(path)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e}")
            return None, ""

def parallel_image_loading(image_paths: List[str], num_workers: int = 8) -> List[Tuple[str, bool]]:
    """Check which images can be loaded in parallel"""
    def check_image(path):
        try:
            Image.open(path).verify()
            return path, True
        except:
            return path, False
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(check_image, image_paths))
    
    return [path for path, valid in results if valid]

def create_optimized_index(
    pictures_dir: str = "pictures",
    csv_path: str = "database_results/final_with_aws_shapes_20250625_155822.csv",
    output_dir: str = "indexes",
    checkpoint: str = "1095",
    batch_size: int = 32,
    index_name: Optional[str] = None
):
    """Create index using optimized batch processing"""
    
    if index_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        index_name = f"gme_optimized_{checkpoint}_{timestamp}"
    
    logger.info("=" * 80)
    logger.info("🚀 OPTIMIZED GME-QWEN BATCH INDEXING")
    logger.info("=" * 80)
    logger.info(f"📁 Pictures: {pictures_dir}")
    logger.info(f"📊 CSV: {csv_path}")
    logger.info(f"💾 Output: {output_dir}")
    logger.info(f"🎯 Checkpoint: {checkpoint}")
    logger.info(f"📦 Batch size: {batch_size}")
    
    start_time = time.time()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load CSV data
    logger.info("\n📖 Loading CSV data...")
    df = pd.read_csv(csv_path, low_memory=False)
    csv_filename_roots = set(df['filename_root'].astype(str).unique())
    logger.info(f"✅ Loaded {len(df)} rows, {len(csv_filename_roots)} unique filename_roots")
    
    # Find all images
    logger.info("\n🔍 Scanning for images...")
    image_extensions = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    all_image_files = []
    
    for ext in image_extensions:
        all_image_files.extend(glob.glob(os.path.join(pictures_dir, ext)))
    
    logger.info(f"📁 Found {len(all_image_files)} total image files")
    
    # Filter valid images
    valid_image_paths = []
    for image_path in all_image_files:
        filename = os.path.basename(image_path)
        filename_root = filename.split('_')[0]
        if filename_root in csv_filename_roots:
            valid_image_paths.append(image_path)
    
    logger.info(f"✅ Found {len(valid_image_paths)} images with CSV entries")
    
    # Pre-validate images in parallel
    logger.info("\n🔍 Pre-validating images...")
    valid_image_paths = parallel_image_loading(valid_image_paths)
    logger.info(f"✅ {len(valid_image_paths)} images are valid and loadable")
    
    if not valid_image_paths:
        logger.error("❌ No valid images found!")
        return False
    
    # Initialize processor
    processor = OptimizedBatchProcessor(checkpoint=checkpoint)
    processor.load_model()
    
    # Show GPU status
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            mem_alloc = torch.cuda.memory_allocated(i) / 1024**3
            mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            logger.info(f"GPU {i}: {mem_alloc:.1f}/{mem_total:.1f} GB")
    
    # Process images in batches
    logger.info(f"\n⚡ Processing {len(valid_image_paths)} images in batches of {batch_size}...")
    
    all_embeddings = []
    all_filenames = []
    
    # Calculate optimal batch size based on available memory
    if torch.cuda.is_available():
        free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
        free_gb = free_memory / 1024**3
        
        # Adjust batch size based on free memory (conservative estimate)
        if free_gb > 30:
            batch_size = min(48, batch_size)  # More conservative for stability
        elif free_gb < 15:
            batch_size = max(8, batch_size // 2)
            
        logger.info(f"📈 Adjusted batch size: {batch_size} (based on {free_gb:.1f}GB free memory)")
    
    # Process with progress bar
    with tqdm(total=len(valid_image_paths), desc="🚀 Processing", unit="img") as pbar:
        for i in range(0, len(valid_image_paths), batch_size):
            batch_paths = valid_image_paths[i:i + batch_size]
            
            # Process batch
            batch_start = time.time()
            embeddings, filenames = processor.process_batch(batch_paths)
            batch_time = time.time() - batch_start
            
            if embeddings:
                all_embeddings.extend(embeddings)
                all_filenames.extend(filenames)
                
                # Update progress
                pbar.update(len(batch_paths))
                imgs_per_sec = len(embeddings) / batch_time
                pbar.set_postfix({
                    'imgs/s': f"{imgs_per_sec:.1f}",
                    'batch_time': f"{batch_time:.2f}s",
                    'GPU_mem': f"{torch.cuda.memory_allocated(0)/1024**3:.1f}GB"
                })
            
            # Clear cache periodically
            if (i // batch_size) % 10 == 0:
                torch.cuda.empty_cache()
    
    if not all_embeddings:
        logger.error("❌ No embeddings generated!")
        return False
    
    logger.info(f"\n✅ Generated {len(all_embeddings)} embeddings")
    
    # Convert to numpy array
    embeddings_array = np.array(all_embeddings, dtype=np.float32)
    logger.info(f"📐 Embeddings shape: {embeddings_array.shape}")
    
    # Create FAISS index
    logger.info("\n⚡ Creating FAISS index...")
    dimension = embeddings_array.shape[1]
    
    # Create index
    if torch.cuda.is_available():
        # Use GPU index
        import faiss.contrib.torch_utils
        
        # Normalize for cosine similarity
        faiss.normalize_L2(embeddings_array)
        
        # Create GPU index
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, dimension)
        index.add(embeddings_array)
        
        # Convert to CPU for saving
        cpu_index = faiss.index_gpu_to_cpu(index)
    else:
        # CPU index
        cpu_index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings_array)
        cpu_index.add(embeddings_array)
    
    logger.info(f"✅ Created index with {cpu_index.ntotal} vectors")
    
    # Save everything
    logger.info("\n💾 Saving index and metadata...")
    
    # Save index
    faiss.write_index(cpu_index, os.path.join(output_dir, f"{index_name}.faiss"))
    
    # Save embeddings
    np.save(os.path.join(output_dir, f"{index_name}_embeddings.npy"), embeddings_array)
    
    # Save metadata
    metadata = {
        'image_paths': all_filenames,
        'index_name': index_name,
        'checkpoint': checkpoint,
        'total_embeddings': len(all_embeddings),
        'embedding_dimension': dimension,
        'created_at': datetime.now().isoformat(),
        'processing_time_seconds': time.time() - start_time,
        'gpu_accelerated': torch.cuda.is_available(),
        'num_gpus_used': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'batch_size_used': batch_size
    }
    
    with open(os.path.join(output_dir, f"{index_name}_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Final summary
    total_time = time.time() - start_time
    
    logger.info("=" * 80)
    logger.info("🎉 INDEXING COMPLETED!")
    logger.info("=" * 80)
    logger.info(f"📊 Total images processed: {len(all_embeddings)}")
    logger.info(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"🚀 Throughput: {len(all_embeddings)/total_time:.1f} images/second")
    logger.info(f"💾 Index saved as: {index_name}")
    logger.info("=" * 80)
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized GME-Qwen batch indexing")
    parser.add_argument("--pictures-dir", default="pictures", help="Directory containing images")
    parser.add_argument("--csv-path", default="database_results/final_with_aws_shapes_20250625_155822.csv")
    parser.add_argument("--output-dir", default="indexes", help="Output directory")
    parser.add_argument("--checkpoint", default="1095", help="LoRA checkpoint (680, 1020, 1095)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--index-name", help="Custom index name")
    
    args = parser.parse_args()
    
    success = create_optimized_index(
        pictures_dir=args.pictures_dir,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        index_name=args.index_name
    )
    
    if success:
        print("\n🎉 SUCCESS: Optimized index created!")
    else:
        print("\n❌ FAILED: Index creation failed!")
        exit(1) 