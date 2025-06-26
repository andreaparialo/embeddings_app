#!/usr/bin/env python3
"""
Optimized LoRA indexing with worker patterns and multi-GPU support
Uses: base model gme-Qwen2-VL-7B-Instruct, LoRA loras/v11-20250620-105815/checkpoint-1095, pictures in pictures/
"""

import torch
import gc
import numpy as np
import faiss
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Union, Tuple
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
import threading

# Setup environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import required libraries
try:
    from transformers import AutoModel
    from transformers.utils.versions import require_version
    from peft import PeftModel
    DEPENDENCIES_OK = True
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.error("Please install: pip install transformers peft")
    DEPENDENCIES_OK = False
    sys.exit(1)


class MultiGPULoRAEngine:
    """Multi-GPU LoRA engine with worker pattern"""
    
    def __init__(self, base_model_path: str = "./gme-Qwen2-VL-7B-Instruct", 
                 lora_path: str = None, num_workers: int = 4):
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.num_workers = min(num_workers, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        self.models = {}
        self.device_queue = Queue()
        
        # Initialize models on different GPUs
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize models on available GPUs"""
        if torch.cuda.is_available():
            available_gpus = list(range(torch.cuda.device_count()))
            print(f"🔥 Initializing models on {len(available_gpus)} GPUs")
            
            for i, gpu_id in enumerate(available_gpus):
                if i >= self.num_workers:
                    break
                    
                device = f"cuda:{gpu_id}"
                print(f"📥 Loading model on GPU {gpu_id}...")
                
                try:
                    # Load base model
                    base_model = AutoModel.from_pretrained(
                        self.base_model_path,
                        torch_dtype=torch.float16,
                        device_map=device,
                        trust_remote_code=True
                    )
                    
                    # Load LoRA if provided
                    if self.lora_path:
                        model = PeftModel.from_pretrained(
                            base_model,
                            self.lora_path,
                            torch_dtype=torch.float16
                        )
                    else:
                        model = base_model
                    
                    self.models[gpu_id] = model
                    self.device_queue.put(gpu_id)
                    print(f"✅ Model loaded on GPU {gpu_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to load model on GPU {gpu_id}: {e}")
                    
        else:
            # CPU fallback
            print("⚠️  No CUDA available, using CPU")
            base_model = AutoModel.from_pretrained(
                self.base_model_path,
                torch_dtype=torch.float32,
                trust_remote_code=True
            )
            
            if self.lora_path:
                model = PeftModel.from_pretrained(base_model, self.lora_path)
            else:
                model = base_model
                
            self.models[0] = model
            self.device_queue.put(0)
    
    def get_image_embedding(self, image_path: Union[str, Path]) -> np.ndarray:
        """Get embedding for a single image using available GPU"""
        gpu_id = self.device_queue.get()
        
        try:
            model = self.models[gpu_id]
            image = Image.open(image_path).convert('RGB')
            
            with torch.no_grad():
                embeddings = model.get_image_embeddings(images=[image])
                result = embeddings.cpu().numpy()[0]
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path} on GPU {gpu_id}: {e}")
            return None
        finally:
            self.device_queue.put(gpu_id)  # Return GPU to queue
    
    def get_stats(self):
        """Get model statistics"""
        stats = {
            "num_gpus": len(self.models),
            "base_model": self.base_model_path,
            "lora_path": self.lora_path,
            "workers": self.num_workers
        }
        
        if torch.cuda.is_available():
            gpu_stats = {}
            for gpu_id in self.models.keys():
                if gpu_id != 0 or torch.cuda.is_available():  # Skip CPU stats
                    torch.cuda.set_device(gpu_id)
                    gpu_stats[f"gpu_{gpu_id}"] = {
                        "memory_allocated": f"{torch.cuda.memory_allocated()/1e9:.2f}GB",
                        "memory_reserved": f"{torch.cuda.memory_reserved()/1e9:.2f}GB"
                    }
            stats["gpu_memory"] = gpu_stats
            
        return stats


class WorkerBasedIndexingEngine:
    """Worker-based indexing engine with thread pool"""
    
    def __init__(self, lora_checkpoint_path: str, num_workers: int = 8, batch_size: int = 32):
        self.lora_checkpoint_path = lora_checkpoint_path
        self.num_workers = num_workers
        self.batch_size = batch_size
        
        # Load multi-GPU LoRA engine
        print("📥 Initializing multi-GPU LoRA engine...")
        self.engine = MultiGPULoRAEngine(
            base_model_path="./gme-Qwen2-VL-7B-Instruct",
            lora_path=lora_checkpoint_path,
            num_workers=min(num_workers, torch.cuda.device_count()) if torch.cuda.is_available() else 1
        )
        
        # Storage
        self.image_paths = []
        self.image_embeddings = None
        self.faiss_index = None
        
        # Performance tracking
        self.processed_count = 0
        self.error_count = 0
        self.start_time = None
        
    def process_image_batch(self, image_paths: List[Path]) -> Tuple[List[np.ndarray], List[str]]:
        """Process a batch of images using thread workers"""
        embeddings = []
        valid_paths = []
        
        # Use ThreadPoolExecutor for I/O bound operations (image loading)
        # while GPU operations are handled by the multi-GPU engine
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all images in the batch
            future_to_path = {
                executor.submit(self.engine.get_image_embedding, img_path): img_path 
                for img_path in image_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                img_path = future_to_path[future]
                try:
                    embedding = future.result()
                    if embedding is not None:
                        embeddings.append(embedding)
                        valid_paths.append(str(img_path))
                        self.processed_count += 1
                    else:
                        self.error_count += 1
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
                    self.error_count += 1
        
        return embeddings, valid_paths
    
    def index_images_with_workers(self, image_directory: str, index_name: str = "lora_worker"):
        """Index images using worker pattern"""
        
        print("🔥" * 60)
        print("🚀 LoRA WORKER-BASED INDEXING")
        print("🔥" * 60)
        
        image_directory = Path(image_directory)
        if not image_directory.exists():
            raise ValueError(f"Directory {image_directory} does not exist")
        
        # Find all image files
        supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp', '.JPG', '.JPEG', '.PNG']
        image_files = []
        for ext in supported_formats:
            image_files.extend(image_directory.rglob(f"*{ext}"))
        
        if not image_files:
            raise ValueError(f"No image files found in {image_directory}")
        
        print(f"📁 Found {len(image_files)} images to index")
        print(f"🎯 LoRA checkpoint: {self.lora_checkpoint_path}")
        print(f"👥 Workers: {self.num_workers}")
        print(f"📦 Batch size: {self.batch_size}")
        print(f"🔧 Engine stats: {self.engine.get_stats()}")
        print()
        
        # Process in batches with workers
        all_embeddings = []
        all_paths = []
        
        self.start_time = time.time()
        
        # Process images in batches
        total_batches = (len(image_files) + self.batch_size - 1) // self.batch_size
        
        with tqdm(total=len(image_files), desc="🚀 Worker Indexing") as pbar:
            for i in range(0, len(image_files), self.batch_size):
                batch = image_files[i:i + self.batch_size]
                batch_num = i // self.batch_size + 1
                
                try:
                    # Process batch with workers
                    batch_embeddings, batch_paths = self.process_image_batch(batch)
                    
                    if batch_embeddings:
                        all_embeddings.extend(batch_embeddings)
                        all_paths.extend(batch_paths)
                    
                    # Update progress
                    pbar.update(len(batch))
                    
                    # Calculate and display stats
                    elapsed = time.time() - self.start_time
                    rate = self.processed_count / elapsed if elapsed > 0 else 0
                    
                    pbar.set_postfix({
                        'batch': f"{batch_num}/{total_batches}",
                        'processed': self.processed_count,
                        'errors': self.error_count,
                        'rate': f"{rate:.1f}/s"
                    })
                    
                    # Memory cleanup every 10 batches
                    if batch_num % 10 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                    
                except Exception as e:
                    logger.error(f"Error processing batch {batch_num}: {e}")
                    pbar.update(len(batch))
                    continue
        
        total_time = time.time() - self.start_time
        
        if not all_embeddings:
            raise ValueError("No valid embeddings generated")
        
        # Store results
        self.image_embeddings = np.array(all_embeddings)
        self.image_paths = all_paths
        
        # Create FAISS index
        dimension = self.image_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.image_embeddings / np.linalg.norm(self.image_embeddings, axis=1, keepdims=True)
        self.faiss_index.add(normalized_embeddings.astype('float32'))
        
        # Save index
        self.save_index(index_name)
        
        # Performance summary
        print(f"\n🎯 WORKER-BASED PERFORMANCE RESULTS:")
        print(f"   📊 Images processed: {len(self.image_paths)}")
        print(f"   ❌ Errors: {self.error_count}")
        print(f"   👥 Workers used: {self.num_workers}")
        print(f"   📦 Batch size: {self.batch_size}")
        print(f"   ⏱️  Total time: {total_time:.1f}s")
        print(f"   🚀 Images/second: {len(self.image_paths)/total_time:.1f}")
        print(f"   💡 Success rate: {len(self.image_paths)/(len(self.image_paths) + self.error_count)*100:.1f}%")
        
        # GPU stats
        if torch.cuda.is_available():
            print(f"   🔧 Final GPU stats: {self.engine.get_stats()}")
        
        return self
    
    def save_index(self, index_name: str) -> None:
        """Save the LoRA index to disk"""
        index_dir = Path("indexes")
        index_dir.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, str(index_dir / f"{index_name}.faiss"))
        
        # Save metadata
        metadata = {
            'image_paths': self.image_paths,
            'embeddings_shape': self.image_embeddings.shape,
            'lora_checkpoint_path': self.lora_checkpoint_path,
            'model_type': 'lora_gme_worker',
            'num_workers': self.num_workers,
            'batch_size': self.batch_size,
            'processing_time': time.time() - self.start_time if self.start_time else 0,
            'engine_stats': self.engine.get_stats()
        }
        
        with open(index_dir / f"{index_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save embeddings
        np.save(index_dir / f"{index_name}_embeddings.npy", self.image_embeddings)
        
        logger.info(f"Worker-based LoRA index saved as {index_name}")


def main():
    """Main function for worker-based LoRA indexing"""
    
    # Configuration
    lora_checkpoint_path = "loras/v11-20250620-105815/checkpoint-1095"
    image_directory = "pictures_prepared"  # Use prepared images
    index_name = "lora_v11_prepared"
    num_workers = 1  # Thread workers for I/O (increased since images are smaller)
    batch_size = 16  # Images per batch (increased since images are smaller)
    
    # Check paths exist
    if not Path(lora_checkpoint_path).exists():
        print(f"❌ LoRA checkpoint not found at: {lora_checkpoint_path}")
        sys.exit(1)
        
    if not Path(image_directory).exists():
        print(f"❌ Image directory not found at: {image_directory}")
        sys.exit(1)
    
    print(f"🎯 Worker-Based Configuration:")
    print(f"   Base model: gme-Qwen2-VL-7B-Instruct")
    print(f"   LoRA checkpoint: {lora_checkpoint_path}")
    print(f"   Image directory: {image_directory}")
    print(f"   Index name: {index_name}")
    print(f"   Thread workers: {num_workers}")
    print(f"   Batch size: {batch_size}")
    if torch.cuda.is_available():
        print(f"   Available GPUs: {torch.cuda.device_count()}")
    print()
    
    # Run indexing
    try:
        engine = WorkerBasedIndexingEngine(
            lora_checkpoint_path=lora_checkpoint_path,
            num_workers=num_workers,
            batch_size=batch_size
        )
        
        engine.index_images_with_workers(image_directory, index_name)
        
        print(f"\n✅ Worker-based LoRA indexing completed!")
        print(f"   Index name: {index_name}")
        print(f"   Use this index for high-performance LoRA-enhanced search!")
        
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 