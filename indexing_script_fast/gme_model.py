import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel
from transformers.utils.versions import require_version
from PIL import Image
import os
import logging
from typing import List, Union, Optional
import gc

# Set environment variables to suppress common warnings and disable multiprocessing
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
os.environ['OMP_NUM_THREADS'] = '1'  # Disable OpenMP threading
os.environ['MKL_NUM_THREADS'] = '1'  # Disable MKL threading
os.environ['NUMEXPR_NUM_THREADS'] = '1'  # Disable NumExpr threading
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Reduce CUDA memory fragmentation

# Ensure compatible transformers version
try:
    require_version(
        "transformers<4.52.0",
        "The remote code has some issues with transformers>=4.52.0"
    )
except ImportError:
    pass

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GMEModel:
    def __init__(self):
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.num_gpus = torch.cuda.device_count() if self.device == "cuda" else 0
        self.current_checkpoint = None
        self.use_data_parallel = self.num_gpus > 1
        
        logger.info(f"GMEModel initialized with device: {self.device}")
        logger.info(f"Number of GPUs available: {self.num_gpus}")
        logger.info(f"Using DataParallel: {self.use_data_parallel}")
        
    def load_model(self, model_path: str = "gme-Qwen2-VL-7B-Instruct", checkpoint: str = "1095"):
        """Load GME model with LoRA and multi-GPU support"""
        try:
            logger.info(f"Loading GME model on {self.device}")
            
            # Clear GPU memory if switching checkpoints
            if self.model is not None and self.current_checkpoint != checkpoint:
                logger.info(f"Clearing previous model (checkpoint {self.current_checkpoint})")
                del self.model
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Load base model if not already loaded or if checkpoint changed
            if self.model is None or self.current_checkpoint != checkpoint:
                logger.info(f"Loading base model from {model_path}")
                
                # Use threading with timeout for model loading
                import threading
                import time
                
                model_loaded = False
                model_error = None
                
                def load_model_worker():
                    nonlocal model_loaded, model_error
                    try:
                        # Load model with appropriate device map for multi-GPU
                        if self.num_gpus > 1:
                            # Auto device map for model parallelism
                            self.model = AutoModel.from_pretrained(
                                model_path,
                                torch_dtype=torch.float16,
                                device_map="auto",  # Automatically distribute across GPUs
                                trust_remote_code=True,
                                use_fast=True,  # Use fast processor to avoid warnings
                                max_memory={i: "35GB" for i in range(self.num_gpus)}  # Leave some memory for activations
                            )
                            logger.info(f"Model loaded with automatic device mapping across {self.num_gpus} GPUs")
                        else:
                            # Single GPU or CPU
                            self.model = AutoModel.from_pretrained(
                                model_path,
                                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                                device_map=self.device,
                                trust_remote_code=True,
                                use_fast=True  # Use fast processor to avoid warnings
                            )
                        
                        logger.info("Base model loaded successfully")
                        model_loaded = True
                        
                    except Exception as e:
                        model_error = e
                        logger.error(f"Error loading base model: {e}")
                
                # Start model loading in separate thread
                model_thread = threading.Thread(target=load_model_worker)
                model_thread.daemon = True
                model_thread.start()
                
                # Wait for model loading with timeout
                model_thread.join(timeout=180)  # 3 minutes timeout
                
                if model_thread.is_alive():
                    logger.warning("⏰ Model loading timed out after 3 minutes")
                    return False
                
                if not model_loaded:
                    logger.error(f"❌ Model loading failed: {model_error}")
                    return False
                
                # Load LoRA if checkpoint specified
                if checkpoint and checkpoint != "base":
                    lora_path = f"loras/v11-20250620-105815/checkpoint-{checkpoint}"
                    if os.path.exists(lora_path):
                        logger.info(f"Loading LoRA checkpoint: {checkpoint}")
                        try:
                            from peft import PeftModel
                            self.model = PeftModel.from_pretrained(self.model, lora_path)
                            logger.info(f"LoRA checkpoint {checkpoint} loaded successfully")
                        except Exception as e:
                            logger.warning(f"Could not load LoRA: {e}")
                    else:
                        logger.warning(f"LoRA checkpoint not found: {lora_path}")
                
                self.current_checkpoint = checkpoint
                logger.info(f"GME model loaded successfully with checkpoint {checkpoint}")
                
                # Log GPU memory usage
                if self.device == "cuda":
                    memory_info = self.get_memory_usage()
                    logger.info(f"GPU memory usage: {memory_info}")
            else:
                logger.info(f"Model already loaded with checkpoint {checkpoint}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading GME model: {e}")
            self.model = None
            return False
    
    def encode_image(self, image_path: str) -> Optional[np.ndarray]:
        """Encode single image to embedding"""
        try:
            if self.model is None:
                logger.error("Model not loaded - cannot encode image")
                return None
            
            if not hasattr(self.model, 'get_image_embeddings'):
                logger.error("Model does not have get_image_embeddings method")
                return None
            
            # Load and preprocess image
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return None
            
            image = Image.open(image_path).convert('RGB')
            
            # Get image embedding
            with torch.no_grad():
                embedding = self.model.get_image_embeddings(images=[image])
                result = embedding.cpu().numpy()[0]
                return result
                
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def encode_images(self, image_paths: List[str], batch_size: int = 16) -> Optional[np.ndarray]:
        """Encode multiple images to embeddings with optimized batching for A100s"""
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return None
            
            # Load images
            images = []
            valid_paths = []
            for path in image_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    images.append(image)
                    valid_paths.append(path)
                except Exception as e:
                    logger.warning(f"Could not load image {path}: {e}")
                    continue
            
            if not images:
                return None
            
            logger.info(f"Loaded {len(images)} images for encoding")
            
            # Use the model's batch processing method
            return self.encode_image_batch(images, batch_size=batch_size)
                
        except Exception as e:
            logger.error(f"Error encoding images: {e}")
            return None
    
    def encode_image_batch(self, images: List[Image.Image], batch_size: int = 16) -> Optional[np.ndarray]:
        """Encode a batch of PIL images to embeddings using GPU acceleration"""
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return None
            
            if not hasattr(self.model, 'get_image_embeddings'):
                logger.error("Model does not have get_image_embeddings method")
                return None
            
            logger.info(f"Processing {len(images)} images in batches of {batch_size}")
            
            # Use the model's native batch processing
            with torch.no_grad():
                embeddings = self.model.get_image_embeddings(
                    images=images,
                    batch_size=batch_size,
                    show_progress_bar=False
                )
            
            # Convert to numpy
            if torch.is_tensor(embeddings):
                result = embeddings.cpu().numpy()
            else:
                result = embeddings
            
            logger.info(f"Batch processed {len(images)} images, output shape: {result.shape}")
            return result
                
        except Exception as e:
            logger.error(f"Error in batch encoding: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def encode_text(self, text: str, instruction: Optional[str] = None) -> Optional[np.ndarray]:
        """Encode text to embedding"""
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return None
            
            with torch.no_grad():
                if instruction:
                    embedding = self.model.get_text_embeddings(texts=[text], instruction=instruction)
                else:
                    embedding = self.model.get_text_embeddings(texts=[text])
                return embedding.cpu().numpy()[0]
                
        except Exception as e:
            logger.error(f"Error encoding text: {e}")
            return None
    
    def get_memory_usage(self) -> dict:
        """Get current GPU memory usage for all GPUs"""
        if torch.cuda.is_available():
            memory_info = {
                "total_gpus": self.num_gpus,
                "gpus": []
            }
            
            for i in range(self.num_gpus):
                gpu_info = {
                    "gpu_id": i,
                    "allocated_gb": torch.cuda.memory_allocated(i) / 1024**3,
                    "cached_gb": torch.cuda.memory_reserved(i) / 1024**3,
                    "total_gb": torch.cuda.get_device_properties(i).total_memory / 1024**3,
                    "free_gb": (torch.cuda.get_device_properties(i).total_memory - torch.cuda.memory_allocated(i)) / 1024**3
                }
                memory_info["gpus"].append(gpu_info)
            
            # Add summary
            memory_info["total_allocated_gb"] = sum(gpu["allocated_gb"] for gpu in memory_info["gpus"])
            memory_info["total_free_gb"] = sum(gpu["free_gb"] for gpu in memory_info["gpus"])
            
            return memory_info
        return {"message": "CUDA not available"}
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None
        
        if torch.cuda.is_available():
            for i in range(self.num_gpus):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
        
        gc.collect()

# Global instance
gme_model = GMEModel() 