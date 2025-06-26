#!/usr/bin/env python3
"""
LoRA EXTREME PERFORMANCE MODE - Maximum GPU utilization with fine-tuned models
"""

import torch
import gc
import numpy as np
import faiss
import json
import logging
from pathlib import Path
from typing import List, Union
from PIL import Image
from tqdm import tqdm

from lora_similarity_engine import LoRAImageSimilarityEngine, create_lora_engine
from lora_model_utils import get_available_lora_models, get_latest_lora_model, find_lora_model_by_version, list_lora_models_summary

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LoRAExtremeAdaptiveBatchProcessor:
    """Aggressive batch processor for LoRA models"""
    
    def __init__(self, initial_batch_size: int = 256, memory_threshold: float = 0.95):
        # Start smaller for LoRA models (they use more memory)
        self.current_batch_size = initial_batch_size
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = 1024  # Lower max for LoRA models
        self.min_batch_size = 8
        self.memory_threshold = memory_threshold
        self.successful_batches = 0
        self.oom_count = 0
        self.performance_history = []
        self.aggressive_scaling = True
        
    def get_optimal_batch_size(self) -> int:
        """AGGRESSIVE batch sizing optimized for LoRA models"""
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
                
            # AGGRESSIVE: Rapid scaling when memory is available
            elif memory_used < 0.8 and self.successful_batches >= 3:  # Slightly more conservative for LoRA
                if self.aggressive_scaling and self.current_batch_size < self.max_batch_size:
                    old_size = self.current_batch_size
                    # Moderate jumps for LoRA models
                    increment = max(16, self.current_batch_size // 6)  
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
    """Fast indexing engine optimized for LoRA models"""
    
    def __init__(self, lora_checkpoint_path: str, max_image_size: int = 512):
        self.lora_checkpoint_path = lora_checkpoint_path
        self.max_image_size = max_image_size
        
        # Load LoRA model
        print("📥 Loading LoRA model for indexing...")
        self.engine = create_lora_engine(lora_checkpoint_path)
        
        # Batch processor
        self.adaptive_batch = LoRAExtremeAdaptiveBatchProcessor()
        
        # Storage
        self.image_paths = []
        self.image_embeddings = None
        self.faiss_index = None
        

    
    def process_batch(self, image_batch: List[Path]) -> List[np.ndarray]:
        """Process a batch of images with LoRA model"""
        embeddings = []
        valid_paths = []
        
        # Process images individually (more reliable for LoRA)
        for img_path in image_batch:
            try:
                # Use the path directly - let the engine handle loading
                embedding = self.engine.get_image_embedding(str(img_path))
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_paths.append(img_path)
            except Exception as e:
                logger.error(f"Error processing image {img_path}: {e}")
                continue
        
        return embeddings, valid_paths
    
    def index_images_extreme(self, image_directory: str, index_name: str = "lora_extreme"):
        """Index images with extreme performance optimization"""
        
        print("🔥" * 60)
        print("🚀 LoRA EXTREME PERFORMANCE INDEXING")
        print("🔥" * 60)
        
        image_directory = Path(image_directory)
        if not image_directory.exists():
            raise ValueError(f"Directory {image_directory} does not exist")
        
        # Find all image files
        supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        image_files = []
        for ext in supported_formats:
            image_files.extend(image_directory.rglob(f"*{ext}"))
            image_files.extend(image_directory.rglob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No image files found in {image_directory}")
        
        print(f"📁 Found {len(image_files)} images to index with LoRA model")
        print(f"🎯 LoRA checkpoint: {self.lora_checkpoint_path}")
        
        # Process in adaptive batches
        all_embeddings = []
        all_paths = []
        
        total_processed = 0
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if start_time:
            start_time.record()
        
        with tqdm(total=len(image_files), desc="🚀 LoRA Indexing") as pbar:
            i = 0
            while i < len(image_files):
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
                        throughput = len(batch_embeddings) / max(1, len(batch))
                        self.adaptive_batch.report_success(throughput)
                        
                        total_processed += len(batch_embeddings)
                        pbar.update(len(batch))
                        pbar.set_postfix({
                            'batch': batch_size,
                            'processed': total_processed,
                            'throughput': f"{throughput:.2f}"
                        })
                    
                    i += len(batch)
                    
                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        self.adaptive_batch.report_oom()
                        continue
                    else:
                        raise e
        
        if end_time and start_time:
            end_time.record()
            torch.cuda.synchronize()
            total_time = start_time.elapsed_time(end_time) / 1000.0  # Convert to seconds
        else:
            total_time = 0
        
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
        print(f"\n🎯 LoRA EXTREME PERFORMANCE RESULTS:")
        print(f"   📊 Images processed: {len(self.image_paths)}")
        print(f"   📈 Final batch size: {self.adaptive_batch.current_batch_size}")
        print(f"   💥 OOM events: {self.adaptive_batch.oom_count}")
        print(f"   ⏱️  Total time: {total_time:.1f}s")
        print(f"   🚀 Images/second: {len(self.image_paths)/max(1, total_time):.1f}")
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"   💾 Final GPU memory: {allocated:.1f}GB / {total_mem:.1f}GB")
        
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
            'model_type': 'lora_gme',
            'max_image_size': self.max_image_size
        }
        
        with open(index_dir / f"{index_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save embeddings
        np.save(index_dir / f"{index_name}_embeddings.npy", self.image_embeddings)
        
        logger.info(f"LoRA index saved as {index_name}")


def lora_extreme_performance_indexing(
    lora_checkpoint_path: str,
    image_directory: str,
    max_image_size: int = 512,
    index_name: str = "lora_extreme"
):
    """Main function for LoRA extreme performance indexing"""
    
    # Create LoRA indexing engine
    engine = LoRAFastIndexingEngine(
        lora_checkpoint_path=lora_checkpoint_path,
        max_image_size=max_image_size
    )
    
    # Index images
    engine.index_images_extreme(image_directory, index_name)
    
    return engine


if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help']:
        print("""
🔥 LoRA Extreme Performance Indexing

Usage:
    python lora_max_performance_indexing.py [model_version] [max_image_size] [index_name] [image_directory]

Arguments:
    model_version     LoRA model version to use (e.g., 'v9', 'v8', 'latest') [default: latest]
    max_image_size    Maximum image size for processing [default: 512]
    index_name        Name for the index [default: lora_extreme]
    image_directory   Directory containing images [default: test_subset_1000]

Examples:
    python lora_max_performance_indexing.py                    # Use latest model
    python lora_max_performance_indexing.py v8                 # Use specific version v8
    python lora_max_performance_indexing.py v9 1024 my_index   # v9 model with custom settings
    python lora_max_performance_indexing.py --list             # List available models

""")
        sys.exit(0)
    
    if len(sys.argv) > 1 and sys.argv[1] == '--list':
        print("🔍 Available LoRA Models:")
        list_lora_models_summary()
        sys.exit(0)
    
    # Get parameters
    model_version = sys.argv[1] if len(sys.argv) > 1 else "latest"
    max_image_size = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    index_name = sys.argv[3] if len(sys.argv) > 3 else "lora_extreme"
    image_directory = sys.argv[4] if len(sys.argv) > 4 else "test_subset_1000"
    
    # Show available models first
    available_models = get_available_lora_models()
    if not available_models:
        print("❌ No LoRA models found")
        print("   Please run fine-tuning first!")
        sys.exit(1)
    
    # Select model based on version
    if model_version == "latest":
        checkpoint_path = get_latest_lora_model()
        selected_model = available_models[0]  # First one is latest
        print(f"🏆 Auto-selected latest model: {selected_model['version']}")
    else:
        checkpoint_path = find_lora_model_by_version(model_version)
        selected_model = next((m for m in available_models if m['version'] == model_version), None)
        
        if not checkpoint_path or not selected_model:
            print(f"❌ Model version '{model_version}' not found!")
            print("\n📋 Available models:")
            list_lora_models_summary()
            sys.exit(1)
        print(f"🎯 Selected model: {selected_model['version']}")
    
    # Show model details
    print(f"📊 Model Details:")
    print(f"   Version: {selected_model['version']}")
    print(f"   Trained: {selected_model['timestamp']}")
    print(f"   Learning Rate: {selected_model['learning_rate']}")
    print(f"   LoRA Rank: {selected_model['lora_rank']}")
    print(f"   LoRA Alpha: {selected_model['lora_alpha']}")
    print(f"   Path: {checkpoint_path}")
    print()
    
    # Run indexing
    try:
        engine = lora_extreme_performance_indexing(
            lora_checkpoint_path=checkpoint_path,
            image_directory=image_directory,
            max_image_size=max_image_size,
            index_name=index_name
        )
        
        print(f"\n✅ LoRA indexing completed!")
        print(f"   Model: {selected_model['version']}")
        print(f"   Index name: {index_name}")
        print(f"   Use this index in the web UI for fine-tuned search!")
        
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        sys.exit(1) 