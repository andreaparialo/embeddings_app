#!/usr/bin/env python3
"""
LoRA-compatible similarity engine for fine-tuned GME models
"""

import torch
import numpy as np
import faiss
import json
import logging
from pathlib import Path
from typing import List, Tuple, Union, Dict
from PIL import Image
import requests
from io import BytesIO
from tqdm import tqdm

from transformers import AutoModel
from transformers.utils.versions import require_version
from peft import PeftModel

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check transformers version
require_version(
    "transformers<4.52.0",
    "The remote code has some issues with transformers>=4.52.0, please downgrade: pip install transformers==4.51.3"
)


class LoRAImageSimilarityEngine:
    """
    A LoRA-compatible similarity engine for fine-tuned GME models.
    Properly loads and uses LoRA adapters for improved performance.
    """
    
    def __init__(self, base_model_path: str = "../gme-Qwen2-VL-7B-Instruct", 
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
            base_model = AutoModel.from_pretrained(
                base_model_path,
                torch_dtype="float16" if self.device == "cuda" else "float32",
                device_map=self.device,
                trust_remote_code=True
            )
            
            # Load LoRA adapters
            self.model = PeftModel.from_pretrained(
                base_model,
                lora_path,
                torch_dtype="float16" if self.device == "cuda" else "float32"
            )
            
            # Don't merge - keep as PeftModel to preserve methods
            # self.model = self.model.merge_and_unload()
            
            logger.info("LoRA model loaded successfully!")
        else:
            logger.info(f"Loading base model only from {base_model_path}")
            self.model = AutoModel.from_pretrained(
                base_model_path,
                torch_dtype="float16" if self.device == "cuda" else "float32",
                device_map=self.device,
                trust_remote_code=True
            )
            logger.info("Base model loaded successfully!")
        
        # Set model to eval mode
        self.model.eval()
        
        # Index storage
        self.image_paths = []
        self.image_embeddings = None
        self.faiss_index = None
        self.metadata = {}
        
        logger.info("LoRA similarity engine initialized successfully!")
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """Load an image from file path or URL."""
        if isinstance(image_path, str) and (image_path.startswith('http://') or image_path.startswith('https://')):
            response = requests.get(image_path)
            return Image.open(BytesIO(response.content)).convert('RGB')
        else:
            return Image.open(image_path).convert('RGB')
    
    @torch.no_grad()
    def get_image_embedding(self, image_input: Union[str, Path, Image.Image]) -> np.ndarray:
        """Get embedding for a single image."""
        try:
            # Handle both file paths and PIL Image objects
            if isinstance(image_input, Image.Image):
                image = image_input
            else:
                image = self.load_image(image_input)
            
            # Use the model's get_image_embeddings method
            embeddings = self.model.get_image_embeddings(images=[image])
            
            # Convert to numpy
            if isinstance(embeddings, torch.Tensor):
                return embeddings.cpu().numpy()[0]
            else:
                return embeddings[0]
        except Exception as e:
            logger.error(f"Error processing image {image_input}: {e}")
            return None
    
    @torch.no_grad()
    def get_text_embedding(self, text: str, instruction: str = None) -> np.ndarray:
        """Get embedding for text query."""
        try:
            if instruction:
                embeddings = self.model.get_text_embeddings(texts=[text], instruction=instruction)
            else:
                embeddings = self.model.get_text_embeddings(texts=[text])
                
            # Convert to numpy
            if isinstance(embeddings, torch.Tensor):
                return embeddings.cpu().numpy()[0]
            else:
                return embeddings[0]
        except Exception as e:
            logger.error(f"Error processing text '{text}': {e}")
            return None
    
    def process_images_batch(self, image_paths: List[Union[str, Path]], batch_size: int = 8) -> List[np.ndarray]:
        """Process multiple images in batches for efficiency."""
        embeddings = []
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            valid_indices = []
            
            # Load images
            for idx, img_path in enumerate(batch_paths):
                try:
                    image = self.load_image(img_path)
                    batch_images.append(image)
                    valid_indices.append(i + idx)
                except Exception as e:
                    logger.error(f"Error loading image {img_path}: {e}")
                    continue
            
            if batch_images:
                try:
                    # Process batch
                    with torch.no_grad():
                        batch_embeddings = self.model.get_image_embeddings(images=batch_images)
                    
                    # Convert to numpy
                    if isinstance(batch_embeddings, torch.Tensor):
                        batch_embeddings = batch_embeddings.cpu().numpy()
                    
                    embeddings.extend(batch_embeddings)
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    # Process individually as fallback
                    for img_path in batch_paths:
                        emb = self.get_image_embedding(img_path)
                        if emb is not None:
                            embeddings.append(emb)
        
        return embeddings
    
    def search_by_text(self, query: str, k: int = 5, 
                      instruction: str = "Find images that match the given text.") -> List[Tuple[str, float]]:
        """
        Search for images using text query.
        
        Args:
            query: Text query
            k: Number of results to return
            instruction: Instruction for the embedding model
            
        Returns:
            List of tuples (image_path, similarity_score)
        """
        if self.faiss_index is None:
            raise ValueError("No index loaded. Please index images first.")
        
        # Get query embedding
        query_embedding = self.get_text_embedding(query, instruction)
        if query_embedding is None:
            raise ValueError("Failed to generate embedding for query")
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        similarities, indices = self.faiss_index.search(query_embedding, k)
        
        # Return results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.image_paths):
                results.append((self.image_paths[idx], float(similarity)))
        
        return results
    
    def search_by_image(self, query_image: Union[str, Path, Image.Image], k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar images using an image query.
        
        Args:
            query_image: Path to the query image or PIL Image object
            k: Number of results to return
            
        Returns:
            List of tuples (image_path, similarity_score)
        """
        if self.faiss_index is None:
            raise ValueError("No index loaded. Please index images first.")
        
        # Get query embedding
        query_embedding = self.get_image_embedding(query_image)
        if query_embedding is None:
            raise ValueError("Failed to generate embedding for query image")
        
        # Normalize query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # Search
        similarities, indices = self.faiss_index.search(query_embedding, k)
        
        # Return results
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.image_paths):
                results.append((self.image_paths[idx], float(similarity)))
        
        return results
    
    def index_images(self, image_directory: Union[str, Path], 
                    supported_formats: List[str] = None,
                    save_index: bool = True,
                    index_name: str = "lora_image_index",
                    batch_size: int = 8) -> None:
        """
        Index all images in a directory.
        
        Args:
            image_directory: Directory containing images to index
            supported_formats: List of supported image formats
            save_index: Whether to save the index to disk
            index_name: Name for the saved index
            batch_size: Batch size for processing images
        """
        if supported_formats is None:
            supported_formats = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp']
        
        image_directory = Path(image_directory)
        if not image_directory.exists():
            raise ValueError(f"Directory {image_directory} does not exist")
        
        # Find all image files
        image_files = []
        for ext in supported_formats:
            image_files.extend(image_directory.rglob(f"*{ext}"))
            image_files.extend(image_directory.rglob(f"*{ext.upper()}"))
        
        if not image_files:
            raise ValueError(f"No image files found in {image_directory}")
        
        logger.info(f"Found {len(image_files)} images to index")
        
        # Generate embeddings using batch processing
        embeddings = self.process_images_batch(image_files, batch_size=batch_size)
        
        if not embeddings:
            raise ValueError("No valid embeddings generated")
        
        # Store embeddings and paths
        self.image_embeddings = np.array(embeddings)
        self.image_paths = [str(img_path) for img_path in image_files[:len(embeddings)]]
        
        # Create FAISS index
        dimension = self.image_embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        normalized_embeddings = self.image_embeddings / np.linalg.norm(self.image_embeddings, axis=1, keepdims=True)
        self.faiss_index.add(normalized_embeddings.astype('float32'))
        
        logger.info(f"Successfully indexed {len(self.image_paths)} images")
        
        # Save index if requested
        if save_index:
            self.save_index(index_name)
    
    def save_index(self, index_name: str) -> None:
        """Save the current index to disk."""
        # Try both index directory locations
        index_dir = Path("indexes") if Path("indexes").exists() else Path("../indexes")
        index_dir.mkdir(exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.faiss_index, str(index_dir / f"{index_name}.faiss"))
        
        # Save metadata
        metadata = {
            'image_paths': self.image_paths,
            'embeddings_shape': self.image_embeddings.shape,
            'base_model_path': self.base_model_path,
            'lora_path': self.lora_path
        }
        
        with open(index_dir / f"{index_name}_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save embeddings
        np.save(index_dir / f"{index_name}_embeddings.npy", self.image_embeddings)
        
        logger.info(f"LoRA index saved as {index_name}")
    
    def load_index(self, index_name: str) -> None:
        """Load a previously saved index."""
        # Try both index directory locations
        index_dir = Path("indexes") if Path("indexes").exists() else Path("../indexes")
        
        # Load FAISS index
        self.faiss_index = faiss.read_index(str(index_dir / f"{index_name}.faiss"))
        
        # Load metadata
        with open(index_dir / f"{index_name}_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.image_paths = metadata['image_paths']
        
        # Load embeddings
        self.image_embeddings = np.load(index_dir / f"{index_name}_embeddings.npy")
        
        logger.info(f"LoRA index {index_name} loaded successfully")

    def get_stats(self) -> Dict:
        """Get statistics about the current index."""
        if self.faiss_index is None:
            return {"status": "No index loaded"}
        
        stats = {
            "total_images": len(self.image_paths),
            "embedding_dimension": self.image_embeddings.shape[1] if self.image_embeddings is not None else 0,
            "index_type": type(self.faiss_index).__name__,
            "base_model_path": self.base_model_path,
            "lora_path": self.lora_path,
            "device": self.device
        }
        
        if self.device == "cuda":
            stats["gpu_memory"] = {
                "allocated": f"{torch.cuda.memory_allocated() / 1e9:.2f} GB",
                "reserved": f"{torch.cuda.memory_reserved() / 1e9:.2f} GB",
                "max_allocated": f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB"
            }
        
        return stats


def create_lora_engine(checkpoint_path: str) -> LoRAImageSimilarityEngine:
    """
    Convenience function to create a LoRA engine from a checkpoint path.
    
    Args:
        checkpoint_path: Path to the LoRA checkpoint directory
        
    Returns:
        LoRAImageSimilarityEngine instance
    """
    base_model_path = "../gme-Qwen2-VL-7B-Instruct"
    return LoRAImageSimilarityEngine(
        base_model_path=base_model_path,
        lora_path=checkpoint_path
    ) 