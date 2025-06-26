import os
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
        self.current_checkpoint = None
        logger.info(f"GMEModel initialized with device: {self.device}")
        logger.info(f"Model attribute initialized: {hasattr(self, 'model')}")
        logger.info(f"Model value: {self.model}")
        
    def load_model(self, model_path: str = "gme-Qwen2-VL-7B-Instruct", checkpoint: str = "1095"):
        """Load GME model with LoRA"""
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
                # Check if model exists in parent directory
                if not os.path.exists(model_path) and os.path.exists(f"../{model_path}"):
                    model_path = f"../{model_path}"
                    logger.info(f"Using model from parent directory: {model_path}")
                
                logger.info(f"Loading base model from {model_path}")
                self.model = AutoModel.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map=self.device,
                    trust_remote_code=True
                )
                logger.info("Base model loaded successfully")
                
                # Load LoRA if checkpoint specified
                if checkpoint and checkpoint != "base":
                    lora_path = f"loras/v11-20250620-105815/checkpoint-{checkpoint}"
                    # Check in parent directory if not found
                    if not os.path.exists(lora_path) and os.path.exists(f"../{lora_path}"):
                        lora_path = f"../{lora_path}"
                        logger.info(f"Using LoRA from parent directory: {lora_path}")
                    
                    if os.path.exists(lora_path):
                        logger.info(f"Loading LoRA checkpoint: {checkpoint}")
                        # Note: The actual LoRA loading depends on how the model was saved
                        # This might need adjustment based on the actual LoRA implementation
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
            else:
                logger.info(f"Model already loaded with checkpoint {checkpoint}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading GME model: {e}")
            # Make sure model is set to None if loading failed
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
            logger.info(f"Loaded image: {image.size}")
            
            # Get image embedding
            with torch.no_grad():
                embedding = self.model.get_image_embeddings(images=[image])
                result = embedding.cpu().numpy()[0]
                logger.info(f"Generated embedding shape: {result.shape}")
                return result
                
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def encode_images(self, image_paths: List[str]) -> Optional[np.ndarray]:
        """Encode multiple images to embeddings"""
        try:
            if self.model is None:
                logger.error("Model not loaded")
                return None
            
            # Load images
            images = []
            for path in image_paths:
                try:
                    image = Image.open(path).convert('RGB')
                    images.append(image)
                except Exception as e:
                    logger.warning(f"Could not load image {path}: {e}")
                    continue
            
            if not images:
                return None
            
            # Get embeddings in batches to manage memory
            batch_size = 8  # Adjust based on GPU memory
            all_embeddings = []
            
            for i in range(0, len(images), batch_size):
                batch = images[i:i+batch_size]
                with torch.no_grad():
                    embeddings = self.model.get_image_embeddings(images=batch)
                    all_embeddings.append(embeddings.cpu().numpy())
                
                # Clear GPU cache
                torch.cuda.empty_cache()
            
            return np.vstack(all_embeddings)
                
        except Exception as e:
            logger.error(f"Error encoding images: {e}")
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
        """Get current GPU memory usage"""
        if torch.cuda.is_available():
            return {
                "allocated": torch.cuda.memory_allocated() / 1024**3,  # GB
                "cached": torch.cuda.memory_reserved() / 1024**3,      # GB
                "max_memory": torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            }
        return {"message": "CUDA not available"}
    
    def cleanup(self):
        """Clean up GPU memory"""
        if self.model is not None:
            del self.model
            self.model = None
        torch.cuda.empty_cache()
        gc.collect()

# Global instance
gme_model = GMEModel() 