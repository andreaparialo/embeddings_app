#!/usr/bin/env python3
"""
Adapted LoRA indexing for the current setup
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
from pathlib import Path
from typing import List, Union
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel
from transformers.utils.versions import require_version

# Setup logging with environment variable suppression
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check transformers version
try:
    require_version(
        "transformers<4.52.0",
        "The remote code has some issues with transformers>=4.52.0, please downgrade: pip install transformers==4.51.3"
    )
except ImportError:
    logger.warning("Could not check transformers version")

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    logger.error("PEFT not available. Please install: pip install peft")
    PEFT_AVAILABLE = False
    sys.exit(1)


class LoRAImageSimilarityEngine:
    """
    A LoRA-compatible similarity engine for fine-tuned GME models.
    Adapted for the current directory structure.
    """
    
    def __init__(self, base_model_path: str = "./gme-Qwen2-VL-7B-Instruct", 
                 lora_path: str = None, device: str = "auto"):
        """
        Initialize the LoRA similarity engine.
        
        Args:
            base_model_path: Path to the base GME model
            lora_path: Path to the LoRA adapters (if None, uses base model only)
            device: Device to run the model on ('auto', 'cuda', 'cpu')
        """
        self.base_model_path = base_model_path
        self.lora_path = lora_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        if lora_path:
            logger.info(f"Loading base model from {base_model_path}")
            logger.info(f"Loading LoRA adapters from {lora_path}")
            
            # Load base model
            self.base_model = AutoModel.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            
            # Load LoRA adapters
            self.model = PeftModel.from_pretrained(
                self.base_model,
                lora_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            logger.info("LoRA model loaded successfully!")
        else:
            logger.info(f"Loading base model only from {base_model_path}")
            self.model = AutoModel.from_pretrained(
                base_model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map=self.device,
                trust_remote_code=True
            )
            logger.info("Base model loaded successfully!")
        
        # Index storage
        self.image_paths = []
        self.image_embeddings = None
        self.faiss_index = None
        self.metadata = {}
        
        logger.info("LoRA similarity engine initialized successfully!")
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load an image from file path."""
        return Image.open(image_path).convert('RGB')
    
    def get_image_embedding(self, image_path: Union[str, Path]) -> np.ndarray:
        """Get embedding for a single image."""
        try:
            image = self.load_image(image_path)
            embeddings = self.model.get_image_embeddings(images=[image])
            return embeddings.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return None
    
    def get_text_embedding(self, text: str, instruction: str = None) -> np.ndarray:
        """Get embedding for text query."""
        try:
            if instruction:
                embeddings = self.model.get_text_embeddings(texts=[text], instruction=instruction)
            else:
                embeddings = self.model.get_text_embeddings(texts=[text])
            return embeddings.cpu().numpy()[0]
        except Exception as e:
            logger.error(f"Error processing text '{text}': {e}")
            return None


class LoRAAdaptiveBatchProcessor:
    """Adaptive batch processor for LoRA models"""
    
    def __init__(self, initial_batch_size: int = 64, memory_threshold: float = 0.9):
        # Start smaller for LoRA models (they use more memory)
        self.current_batch_size = initial_batch_size
        self.initial_batch_size = initial_batch_size
        self.max_batch_size = 256  # Conservative max for LoRA models
        self.min_batch_size = 4
        self.memory_threshold = memory_threshold
        self.successful_batches = 0
        self.oom_count = 0
        self.performance_history = []
        
    def get_optimal_batch_size(self) -> int:
        """Get optimal batch size based on GPU memory"""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            memory_used = allocated_memory / total_memory if total_memory > 0 else 0
            
            print(f"🔥 LoRA GPU Memory: {memory_used:.1%} ({allocated_memory/1e9:.1f}GB / {total_memory/1e9:.1f}GB)")
            
            # Reduce if memory usage is high
            if memory_used > self.memory_threshold:
                old_size = self.current_batch_size
                self.current_batch_size = max(int(self.current_batch_size * 0.8), self.min_batch_size)
                if old_size != self.current_batch_size:
                    print(f"🔽 Reducing batch size to {self.current_batch_size}")
                torch.cuda.empty_cache()
                
            # Increase if memory is available and we've had successful batches
            elif memory_used < 0.7 and self.successful_batches >= 5:
                if self.current_batch_size < self.max_batch_size:
                    old_size = self.current_batch_size
                    increment = max(8, self.current_batch_size // 8)  
                    self.current_batch_size = min(self.current_batch_size + increment, self.max_batch_size)
                    
                    if old_size != self.current_batch_size:
                        print(f"🚀 Scaling UP to {self.current_batch_size}")
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
        self.current_batch_size = max(int(self.current_batch_size * 0.6), self.min_batch_size)
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
        self.engine = LoRAImageSimilarityEngine(
            base_model_path="./gme-Qwen2-VL-7B-Instruct",
            lora_path=lora_checkpoint_path
        )
        
        # Batch processor
        self.adaptive_batch = LoRAAdaptiveBatchProcessor()
        
        # Storage
        self.image_paths = []
        self.image_embeddings = None
        self.faiss_index = None
    
    def process_batch(self, image_batch: List[Path]) -> tuple:
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
    
    def index_images_optimized(self, image_directory: str, index_name: str = "lora_optimized"):
        """Index images with optimized performance"""
        
        print("🔥" * 60)
        print("🚀 LoRA OPTIMIZED PERFORMANCE INDEXING")
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
        print(f"\n🎯 LoRA PERFORMANCE RESULTS:")
        print(f"   📊 Images processed: {len(self.image_paths)}")
        print(f"   📈 Final batch size: {self.adaptive_batch.current_batch_size}")
        print(f"   💥 OOM events: {self.adaptive_batch.oom_count}")
        print(f"   ⏱️  Total time: {total_time:.1f}s")
        if total_time > 0:
            print(f"   🚀 Images/second: {len(self.image_paths)/total_time:.1f}")
        
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


def main():
    """Main function for LoRA indexing"""
    
    # Configuration
    lora_checkpoint_path = "loras/v11-20250620-105815/checkpoint-1095"
    image_directory = "pictures"
    index_name = "lora_v11_pictures"
    max_image_size = 512
    
    # Check paths exist
    if not Path(lora_checkpoint_path).exists():
        print(f"❌ LoRA checkpoint not found at: {lora_checkpoint_path}")
        sys.exit(1)
        
    if not Path(image_directory).exists():
        print(f"❌ Image directory not found at: {image_directory}")
        sys.exit(1)
    
    print(f"🎯 Configuration:")
    print(f"   Base model: gme-Qwen2-VL-7B-Instruct")
    print(f"   LoRA checkpoint: {lora_checkpoint_path}")
    print(f"   Image directory: {image_directory}")
    print(f"   Index name: {index_name}")
    print(f"   Max image size: {max_image_size}")
    print()
    
    # Run indexing
    try:
        engine = LoRAFastIndexingEngine(
            lora_checkpoint_path=lora_checkpoint_path,
            max_image_size=max_image_size
        )
        
        engine.index_images_optimized(image_directory, index_name)
        
        print(f"\n✅ LoRA indexing completed!")
        print(f"   Index name: {index_name}")
        print(f"   Use this index in the web UI for fine-tuned search!")
        
    except Exception as e:
        print(f"❌ Indexing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 