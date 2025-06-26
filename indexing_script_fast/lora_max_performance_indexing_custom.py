#!/usr/bin/env python3
"""
LoRA EXTREME PERFORMANCE MODE - Maximum GPU utilization with fine-tuned models
CUSTOMIZED VERSION - Batch size 128, custom save location
"""

import torch
import gc
import numpy as np
import faiss
import json
import logging
import time
from pathlib import Path
from typing import List, Union
import torchvision.io as tvio
import torchvision.transforms.functional as F
from tqdm import tqdm
import sys
import os

from lora_similarity_engine import LoRAImageSimilarityEngine, create_lora_engine
from lora_model_utils import get_available_lora_models, get_latest_lora_model, find_lora_model_by_version, list_lora_models_summary

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProfilingTimer:
    """Utility class for measuring performance bottlenecks"""
    
    def __init__(self):
        self.timings = {}
        self.start_times = {}
    
    def start(self, name: str):
        self.start_times[name] = time.time()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    def end(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            if name not in self.timings:
                self.timings[name] = []
            self.timings[name].append(elapsed)
    
    def get_stats(self):
        stats = {}
        for name, times in self.timings.items():
            stats[name] = {
                'avg': np.mean(times),
                'total': np.sum(times),
                'count': len(times),
                'min': np.min(times),
                'max': np.max(times)
            }
        return stats
    
    def print_stats(self):
        print("\n🔍 PERFORMANCE BOTTLENECK ANALYSIS:")
        print("=" * 60)
        
        stats = self.get_stats()
        
        # Sort by total time (biggest bottlenecks first)
        sorted_stats = sorted(stats.items(), key=lambda x: x[1]['total'], reverse=True)
        
        for name, data in sorted_stats:
            avg_ms = data['avg'] * 1000
            total_s = data['total']
            count = data['count']
            throughput = count / total_s if total_s > 0 else 0
            
            print(f"📊 {name}:")
            print(f"   Total: {total_s:.2f}s | Avg: {avg_ms:.1f}ms | Count: {count}")
            print(f"   Throughput: {throughput:.1f} ops/sec | Min: {data['min']*1000:.1f}ms | Max: {data['max']*1000:.1f}ms")
            print()


class OptimizedLoRAEngine(LoRAImageSimilarityEngine):
    """Optimized LoRA engine with torchvision.io for GPU-accelerated image loading"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.timer = ProfilingTimer()
    
    def load_image_fast(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Load image using torchvision.io - GPU accelerated"""
        try:
            # Read image directly as tensor (RGB format)
            img_tensor = tvio.read_image(str(image_path), mode=tvio.ImageReadMode.RGB)
            
            # Convert to float and normalize to [0, 1]
            img_tensor = img_tensor.float() / 255.0
            
            # Move to GPU if available
            if torch.cuda.is_available():
                img_tensor = img_tensor.cuda()
            
            return img_tensor
        except Exception as e:
            raise ValueError(f"Failed to load image: {image_path}, error: {e}")
    
    @torch.no_grad()
    def get_image_embedding_fast(self, image_path: Union[str, Path]) -> np.ndarray:
        """Get embedding for a single image using fast GPU loading"""
        try:
            # Time image loading
            self.timer.start('image_loading')
            img_tensor = self.load_image_fast(image_path)
            self.timer.end('image_loading')
            
            # Time PIL conversion (if this is needed)
            self.timer.start('tensor_to_pil')  
            from PIL import Image
            # Convert back to CPU, scale to 0-255, and convert to numpy
            img_numpy = (img_tensor.cpu() * 255).byte().permute(1, 2, 0).numpy()
            image = Image.fromarray(img_numpy)
            self.timer.end('tensor_to_pil')
            
            # Time model inference (the actual embedding generation)
            self.timer.start('model_inference')
            embeddings = self.model.get_image_embeddings(images=[image])
            self.timer.end('model_inference')
            
            # Time tensor conversion
            self.timer.start('tensor_conversion')
            if isinstance(embeddings, torch.Tensor):
                result = embeddings.cpu().numpy()[0]
            else:
                result = embeddings[0]
            self.timer.end('tensor_conversion')
            
            return result
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None


class LoRAExtremeAdaptiveBatchProcessor:
    """Aggressive batch processor for LoRA models - CUSTOMIZED FOR BATCH 16-128"""
    
    def __init__(self, initial_batch_size: int = 16, memory_threshold: float = 0.90):
        # Start with batch size 16 as requested
        self.current_batch_size = initial_batch_size
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = 128  # Max batch size as requested
        self.min_batch_size = 1  # Keep minimum at 1 for reliability
        self.memory_threshold = memory_threshold
        self.successful_batches = 0
        self.oom_count = 0
        self.performance_history = []
        self.aggressive_scaling = True
        
    def get_optimal_batch_size(self) -> int:
        """AGGRESSIVE batch sizing optimized for LoRA models - STARTING AT 128"""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            memory_used = allocated_memory / total_memory if total_memory > 0 else 0
            
            print(f"🔥 LoRA GPU Memory: {memory_used:.1%} ({allocated_memory/1e9:.1f}GB / {total_memory/1e9:.1f}GB)")
            
            # AGGRESSIVE: Only reduce if we're very close to limit
            if memory_used > self.memory_threshold:
                old_size = self.current_batch_size
                self.current_batch_size = max(int(self.current_batch_size * 0.8), self.min_batch_size)
                print(f"🔽 LoRA AGGRESSIVE: Reducing batch size to {self.current_batch_size}")
                torch.cuda.empty_cache()
                
            # ULTRA AGGRESSIVE: Rapid scaling when memory is available
            elif memory_used < 0.7 and self.successful_batches >= 2:
                if self.aggressive_scaling and self.current_batch_size < self.max_batch_size:
                    old_size = self.current_batch_size
                    # More conservative increments for stability
                    increment = min(16, self.current_batch_size // 2)  
                    self.current_batch_size = min(self.current_batch_size + increment, self.max_batch_size)
                    
                    if old_size != self.current_batch_size:
                        print(f"🚀 LoRA AGGRESSIVE: Scaling UP to {self.current_batch_size}")
                        self.successful_batches = 0
                
        return self.current_batch_size
    
    def report_success(self, throughput: float = None):
        self.successful_batches += 1
        if throughput:
            self.performance_history.append(throughput)
            if len(self.performance_history) > 10:
                self.performance_history = self.performance_history[-10:]
                
    def report_oom(self):
        self.oom_count += 1
        old_size = self.current_batch_size
        self.current_batch_size = max(int(self.current_batch_size * 0.7), self.min_batch_size)
        print(f"💥 LoRA OOM! Reducing from {old_size} to {self.current_batch_size}")
        torch.cuda.empty_cache()
        gc.collect()


class LoRAFastIndexingEngine:
    """Fast indexing engine optimized for LoRA models - CUSTOM SAVE LOCATION"""
    
    def __init__(self, lora_checkpoint_path: str, max_image_size: int = 512, save_directory: str = "indexes"):
        self.lora_checkpoint_path = lora_checkpoint_path
        self.max_image_size = max_image_size
        self.save_directory = save_directory  # Custom save directory
        self.timer = ProfilingTimer()
        
        # Load LoRA model with optimized engine
        print("📥 Loading LoRA model for indexing...")
        base_model_path = "gme-Qwen2-VL-7B-Instruct"  # Local path in this directory
        self.engine = OptimizedLoRAEngine(
            base_model_path=base_model_path,
            lora_path=lora_checkpoint_path
        )
        
        # Batch processor with batch size 16 as requested
        self.adaptive_batch = LoRAExtremeAdaptiveBatchProcessor(initial_batch_size=16)
        
        # Storage
        self.image_paths = []
        self.image_embeddings = None
        self.faiss_index = None
        
    def process_batch(self, image_batch: List[Path]) -> List[np.ndarray]:
        """Process a batch of images individually (reliable method)"""
        self.timer.start('batch_processing')
        
        embeddings = []
        valid_paths = []
        
        # Process images individually (reliable approach)
        for img_path in image_batch:
            try:
                # Use the standard reliable path  
                embedding = self.engine.get_image_embedding(str(img_path))
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_paths.append(img_path)
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")
                continue
        
        self.timer.end('batch_processing')
        return embeddings, valid_paths
    
    def index_images_extreme(self, image_directory: str, index_name: str = "lora_extreme"):
        """Index images with extreme performance optimization"""
        
        print("🔥" * 60)
        print("🚀 LoRA EXTREME PERFORMANCE INDEXING WITH PROFILING - BATCH 16-128")
        print("🔥" * 60)
        
        image_directory = Path(image_directory)
        if not image_directory.exists():
            raise ValueError(f"Directory {image_directory} does not exist")
        
        # Time directory scanning
        self.timer.start('directory_scan')
        supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        image_files = []
        for ext in supported_formats:
            image_files.extend(image_directory.rglob(f"*{ext}"))
            image_files.extend(image_directory.rglob(f"*{ext.upper()}"))
        self.timer.end('directory_scan')
        
        if not image_files:
            raise ValueError(f"No image files found in {image_directory}")
        
        print(f"📁 Found {len(image_files)} images to index with LoRA model")
        print(f"🎯 LoRA checkpoint: {self.lora_checkpoint_path}")
        print(f"💾 Save directory: {self.save_directory}")
        
        # Process in adaptive batches
        all_embeddings = []
        all_paths = []
        
        total_processed = 0
        start_time = time.time()
        
        with tqdm(total=len(image_files), desc="🚀 LoRA Indexing") as pbar:
            i = 0
            while i < len(image_files):
                batch_start = time.time()
                
                # Get optimal batch size
                batch_size = self.adaptive_batch.get_optimal_batch_size()
                batch = image_files[i:i + batch_size]
                
                try:
                    # Process batch
                    batch_embeddings, batch_paths = self.process_batch(batch)
                    
                    if batch_embeddings:
                        all_embeddings.extend(batch_embeddings)
                        all_paths.extend([str(p) for p in batch_paths])
                        
                        # Calculate throughput
                        batch_time = time.time() - batch_start
                        batch_throughput = len(batch_embeddings) / batch_time
                        self.adaptive_batch.report_success(batch_throughput)
                        
                        total_processed += len(batch_embeddings)
                        pbar.update(len(batch))
                        pbar.set_postfix({
                            'batch': batch_size,
                            'processed': total_processed,
                            'throughput': f"{batch_throughput:.1f}/s"
                        })
                    
                    i += len(batch)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.adaptive_batch.report_oom()
                        continue
                    else:
                        raise e
        
        # Time FAISS index creation
        self.timer.start('faiss_index_creation')
        
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
        
        self.timer.end('faiss_index_creation')
        
        # Time index saving
        self.timer.start('index_saving')
        self.save_index(index_name)
        self.timer.end('index_saving')
        
        total_time = time.time() - start_time
        
        # Performance summary
        print(f"\n🎯 LoRA EXTREME PERFORMANCE RESULTS:")
        print(f"   📊 Images processed: {len(self.image_paths)}")
        print(f"   📈 Final batch size: {self.adaptive_batch.current_batch_size}")
        print(f"   💥 OOM events: {self.adaptive_batch.oom_count}")
        print(f"   ⏱️  Total time: {total_time:.1f}s")
        print(f"   🚀 Overall throughput: {len(self.image_paths)/total_time:.1f} images/sec")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   💾 Final GPU memory: {allocated:.1f}GB / {total_mem:.1f}GB")
        
        # Print detailed profiling results
        self.engine.timer.print_stats()
        self.timer.print_stats()
        
        return self
    
    def save_index(self, index_name: str) -> None:
        """Save the LoRA index to disk - CUSTOM SAVE LOCATION"""
        index_dir = Path(self.save_directory)
        index_dir.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, str(index_dir / f"{index_name}.faiss"))
        
        # Save metadata
        metadata = {
            'image_paths': self.image_paths,
            'embeddings_shape': self.image_embeddings.shape,
            'lora_checkpoint_path': self.lora_checkpoint_path,
            'model_type': 'lora_gme',
            'max_image_size': self.max_image_size
        }
        
        with open(index_dir / f"{index_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save embeddings
        np.save(index_dir / f"{index_name}_embeddings.npy", self.image_embeddings)
        
        logger.info(f"LoRA index saved as {index_name} in {index_dir}")


def lora_extreme_performance_indexing(
    lora_checkpoint_path: str,
    image_directory: str,
    max_image_size: int = 512,
    index_name: str = "lora_extreme",
    save_directory: str = "indexes"
):
    """Main function for LoRA extreme performance indexing - CUSTOMIZED"""
    
    # Create LoRA indexing engine
    engine = LoRAFastIndexingEngine(
        lora_checkpoint_path=lora_checkpoint_path,
        max_image_size=max_image_size,
        save_directory=save_directory
    )
    
    # Index images
    engine.index_images_extreme(image_directory, index_name)
    
    return engine


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
🔥 LoRA Extreme Performance Indexing WITH PROFILING - CUSTOMIZED VERSION

Usage:
    python lora_max_performance_indexing_custom.py [checkpoint_path] [image_directory] [index_name] [save_directory]

Arguments:
    checkpoint_path   Full path to LoRA checkpoint directory
    image_directory   Directory containing images
    index_name        Name for the index
    save_directory    Directory to save the index files

Examples:
    python lora_max_performance_indexing_custom.py loras/v8-20250617-143533/checkpoint-680 pictures v8_all_index_680 indexes

""")
        sys.exit(0)
    
    # Get parameters with defaults for the specific use case
    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else "loras/v8-20250617-143533/checkpoint-680"
    image_directory = sys.argv[2] if len(sys.argv) > 2 else "pictures"
    index_name = sys.argv[3] if len(sys.argv) > 3 else "v8_all_index_680"
    save_directory = sys.argv[4] if len(sys.argv) > 4 else "indexes"
    
    # Validate paths
    if not Path(checkpoint_path).exists():
        print(f"❌ Checkpoint path '{checkpoint_path}' not found!")
        sys.exit(1)
        
    if not Path(image_directory).exists():
        print(f"❌ Image directory '{image_directory}' not found!")
        sys.exit(1)
    
    print(f"🎯 Using LoRA checkpoint: {checkpoint_path}")
    print(f"📁 Processing images from: {image_directory}")
    print(f"🏷️  Index name: {index_name}")
    print(f"💾 Save directory: {save_directory}")
    print(f"🔥 Starting batch size: 16")
    print()
    
    # Run indexing
    try:
        engine = lora_extreme_performance_indexing(
            lora_checkpoint_path=checkpoint_path,
            image_directory=image_directory,
            max_image_size=512,
            index_name=index_name,
            save_directory=save_directory
        )
        
        print(f"\n✅ LoRA indexing completed!")
        print(f"   Checkpoint: {checkpoint_path}")
        print(f"   Index name: {index_name}")
        print(f"   Saved to: {save_directory}")
        print(f"   Use this index in the web UI for fine-tuned search!")
        
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        sys.exit(1) 