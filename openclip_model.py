import torch
import numpy as np
import open_clip
from PIL import Image
import os
import logging
from typing import List, Union, Optional
import gc
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OpenCLIPModel:
    def __init__(self):
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.current_checkpoint = None
        logger.info(f"OpenCLIPModel initialized with device: {self.device}")
        
    def load_model(self, checkpoint_path: str = "epoch_008_model.pth"):
        """Load OpenCLIP model from checkpoint"""
        try:
            logger.info(f"Loading OpenCLIP model on {self.device}")
            
            # Clear GPU memory if switching checkpoints
            if self.model is not None and self.current_checkpoint != checkpoint_path:
                logger.info(f"Clearing previous model (checkpoint {self.current_checkpoint})")
                del self.model
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            
            # Load base model if not already loaded or if checkpoint changed
            if self.model is None or self.current_checkpoint != checkpoint_path:
                logger.info(f"Loading OpenCLIP ViT-L-14 model")
                
                # Create model with custom settings as specified
                # First create on CPU, then move to device after loading checkpoint
                self.model, _, _ = open_clip.create_model_and_transforms(
                    'ViT-L-14',
                    pretrained=None,  # Using custom checkpoint
                    precision='fp16' if self.device == 'cuda' else 'fp32',  # Use fp16 on GPU for speed
                    device='cpu',  # Create on CPU first
                    force_quick_gelu=True
                )
                
                # Define custom preprocessing as specified
                self.preprocess = transforms.Compose([
                    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=(0.48145466, 0.4578275, 0.40821073),
                        std=(0.26862954, 0.26130258, 0.27577711)
                    )
                ])
                
                # Load checkpoint
                # Try multiple possible paths
                possible_paths = [
                    os.path.join("/home/ubuntu/SPEEDINGTHEPROCESS/openclip_embeddings_v1/checkpoints_clean/checkpoints", checkpoint_path),
                    os.path.join("openclip_embeddings_v1/checkpoints_clean/checkpoints", checkpoint_path),
                    checkpoint_path  # If full path is provided
                ]
                
                full_checkpoint_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        full_checkpoint_path = path
                        break
                
                if full_checkpoint_path:
                    logger.info(f"Loading checkpoint from {full_checkpoint_path}")
                    checkpoint = torch.load(full_checkpoint_path, map_location=self.device)
                    
                    # Handle different checkpoint formats
                    if isinstance(checkpoint, dict):
                        if 'state_dict' in checkpoint:
                            state_dict = checkpoint['state_dict']
                        elif 'model' in checkpoint:
                            state_dict = checkpoint['model']
                        else:
                            state_dict = checkpoint
                    else:
                        state_dict = checkpoint
                    
                    # Load state dict
                    self.model.load_state_dict(state_dict, strict=False)
                    logger.info(f"Checkpoint {checkpoint_path} loaded successfully")
                else:
                    logger.error(f"Checkpoint not found in any of the expected paths: {possible_paths}")
                    return False
                
                # Move model to device (GPU if available)
                self.model = self.model.to(self.device)
                logger.info(f"Model moved to {self.device}")
                
                # Set model to eval mode
                self.model.eval()
                
                # Don't use torch.compile for now - it can slow down initial runs
                # if self.device == 'cuda' and hasattr(torch, 'compile'):
                #     try:
                #         self.model = torch.compile(self.model, mode='reduce-overhead')
                #         logger.info("Model compiled with torch.compile for faster inference")
                #     except Exception as e:
                #         logger.warning(f"Could not compile model: {e}")
                
                # Get tokenizer for text encoding
                self.tokenizer = open_clip.get_tokenizer('ViT-L-14')
                
                self.current_checkpoint = checkpoint_path
                logger.info(f"OpenCLIP model loaded successfully with checkpoint {checkpoint_path}")
            else:
                logger.info(f"Model already loaded with checkpoint {checkpoint_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading OpenCLIP model: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.model = None
            return False
    
    def encode_image(self, image_path: str) -> Optional[np.ndarray]:
        """Encode single image to embedding"""
        try:
            if self.model is None or self.preprocess is None:
                logger.error("Model not loaded - cannot encode image")
                return None
            
            # Load and preprocess image
            if not os.path.exists(image_path):
                logger.error(f"Image file does not exist: {image_path}")
                return None
            
            image = Image.open(image_path).convert('RGB')
            logger.debug(f"Loaded image: {image.size}")
            
            # Preprocess image
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            # Convert to same dtype as model
            if self.device == 'cuda':
                image_tensor = image_tensor.half()  # Convert to fp16 for GPU
            
            # Get image embedding
            with torch.no_grad():
                image_features = self.model.encode_image(image_tensor)
                # Normalize the features
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                result = image_features.cpu().numpy()[0]
                logger.debug(f"Generated embedding shape: {result.shape}")
                return result
                
        except Exception as e:
            logger.error(f"Error encoding image {image_path}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def encode_images(self, image_paths: List[str], batch_size: int = 256) -> Optional[np.ndarray]:
        """Encode multiple images to embeddings"""
        try:
            if self.model is None or self.preprocess is None:
                logger.error("Model not loaded")
                return None
            
            all_embeddings = []
            valid_paths = []
            
            # Filter valid image paths
            for path in image_paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    logger.warning(f"Image not found: {path}")
            
            if not valid_paths:
                logger.error("No valid image paths provided")
                return None
            
            logger.info(f"Encoding {len(valid_paths)} images in batches of {batch_size}")
            
            # Process in batches
            for i in range(0, len(valid_paths), batch_size):
                batch_paths = valid_paths[i:i+batch_size]
                batch_images = []
                
                # Load and preprocess batch
                for path in batch_paths:
                    try:
                        image = Image.open(path).convert('RGB')
                        image_tensor = self.preprocess(image)
                        batch_images.append(image_tensor)
                    except Exception as e:
                        logger.warning(f"Could not load image {path}: {e}")
                        continue
                
                if not batch_images:
                    continue
                
                # Stack into batch tensor
                batch_tensor = torch.stack(batch_images).to(self.device)
                # Convert to same dtype as model
                if self.device == 'cuda':
                    batch_tensor = batch_tensor.half()  # Convert to fp16 for GPU
                
                # Encode batch
                with torch.no_grad():
                    batch_features = self.model.encode_image(batch_tensor)
                    # Normalize the features
                    batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                    all_embeddings.append(batch_features.cpu().numpy())
                
                # Clear GPU cache periodically
                if i % (batch_size * 10) == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Log progress more frequently
                if (i + batch_size) % 100 == 0 or i == 0:
                    processed = min(i + batch_size, len(valid_paths))
                    percentage = (processed / len(valid_paths)) * 100
                    logger.info(f"Encoded {processed}/{len(valid_paths)} images ({percentage:.1f}%)")
            
            if all_embeddings:
                return np.vstack(all_embeddings)
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error encoding images: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
    
    def encode_text(self, text: str) -> Optional[np.ndarray]:
        """Encode text to embedding"""
        try:
            if self.model is None or self.tokenizer is None:
                logger.error("Model not loaded")
                return None
            
            with torch.no_grad():
                # Tokenize text
                text_tokens = self.tokenizer([text]).to(self.device)
                
                # Encode text
                text_features = self.model.encode_text(text_tokens)
                # Normalize the features
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                return text_features.cpu().numpy()[0]
                
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# Global instance
openclip_model = OpenCLIPModel() 