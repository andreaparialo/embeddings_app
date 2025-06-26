import os
import numpy as np
import pandas as pd
import logging
import glob
import faiss
import json
from datetime import datetime
from openclip_model import openclip_model
from openclip_data_loader import openclip_data_loader
import time
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_openclip_embeddings(pictures_dir: str = "pictures", 
                             csv_path: str = "database_results/final_with_aws_shapes_20250625_155822.csv",
                             output_dir: str = "indexes/openclip",
                             batch_size: int = 256,  # Increased for GPU
                             checkpoint: str = "epoch_008_model.pth"):
    """
    Create OpenCLIP embeddings for all images in the pictures directory with GPU optimization
    """
    try:
        logger.info("=" * 80)
        logger.info("Starting OpenCLIP embedding creation with GPU optimization")
        logger.info(f"Pictures directory: {pictures_dir}")
        logger.info(f"CSV path: {csv_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Checkpoint: {checkpoint}")
        
        # Check GPU availability
        if torch.cuda.is_available():
            logger.info(f"GPU available: {torch.cuda.get_device_name()}")
            logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        else:
            logger.warning("No GPU available - using CPU (will be slower)")
        
        logger.info("=" * 80)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Load CSV to get filename_root information
        logger.info("Loading CSV data...")
        df = openclip_data_loader.load_csv(csv_path)
        logger.info(f"Loaded {len(df)} rows from CSV")
        
        # Get unique filename_roots from CSV
        csv_filename_roots = set(df['filename_root'].astype(str).unique())
        logger.info(f"Found {len(csv_filename_roots)} unique filename_roots in CSV")
        
        # Get all image files
        logger.info("Scanning for image files...")
        jpg_files = glob.glob(os.path.join(pictures_dir, "*.jpg"))
        JPG_files = glob.glob(os.path.join(pictures_dir, "*.JPG"))
        all_image_files = jpg_files + JPG_files
        logger.info(f"Found {len(all_image_files)} total image files")
        
        # Filter to only process images that have corresponding CSV entries
        valid_image_paths = []
        filename_root_to_path = {}
        
        for image_path in all_image_files:
            filename = os.path.basename(image_path)
            # Extract filename_root (everything before first underscore)
            filename_root = filename.split('_')[0]
            
            if filename_root in csv_filename_roots:
                valid_image_paths.append(image_path)
                filename_root_to_path[filename_root] = image_path
        
        logger.info(f"Found {len(valid_image_paths)} images with corresponding CSV entries")
        
        if len(valid_image_paths) == 0:
            logger.error("No valid images found to process!")
            return False
        
        # Load OpenCLIP model
        logger.info("Loading OpenCLIP model...")
        checkpoint_path = f"openclip_embeddings_v1/checkpoints_clean/checkpoints/{checkpoint}" if checkpoint else None
        if not openclip_model.load_model(checkpoint_path):
            logger.error("Failed to load OpenCLIP model")
            return False
        
        # Adjust batch size based on GPU availability
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            # Scale batch size with number of GPUs
            batch_size = batch_size * torch.cuda.device_count()
            logger.info(f"Using {torch.cuda.device_count()} GPUs with batch size {batch_size}")
        
        # Process images in batches
        logger.info(f"Starting embedding generation for {len(valid_image_paths)} images...")
        if openclip_model.device == 'cuda':
            logger.info(f"Using GPU - estimated time: {len(valid_image_paths) / 1000:.1f} - {len(valid_image_paths) / 500:.1f} minutes")
        else:
            logger.info(f"Using CPU - estimated time: {len(valid_image_paths) / 100:.1f} - {len(valid_image_paths) / 50:.1f} minutes")
        
        start_time = datetime.now()
        
        # Generate embeddings
        embeddings = openclip_model.encode_images(valid_image_paths, batch_size=batch_size)
        
        if embeddings is None:
            logger.error("Failed to generate embeddings")
            return False
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Generated embeddings in {duration:.2f} seconds ({len(valid_image_paths)/duration:.2f} images/sec)")
        
        # Log GPU memory usage
        if torch.cuda.is_available():
            memory_info = openclip_model.get_memory_usage()
            logger.info(f"GPU memory usage: {memory_info}")
        
        # Prepare metadata
        logger.info("Preparing metadata...")
        image_paths = [os.path.basename(path) for path in valid_image_paths]
        
        # Save embeddings and metadata
        logger.info("Saving embeddings and metadata...")
        
        # Save in NPZ format
        npz_path = os.path.join(output_dir, "openclip_embeddings.npz")
        np.savez(npz_path,
                 embeddings=embeddings,
                 image_paths=np.array(image_paths))
        logger.info(f"Saved NPZ format to {npz_path}")
        
        # Save embeddings in numpy format
        np.save(os.path.join(output_dir, "openclip_embeddings.npy"), embeddings)
        
        # Create and save FAISS index
        logger.info("Creating FAISS index...")
        dimension = embeddings.shape[1]  # Should be 768 for OpenCLIP ViT-B-32 or 512/1024 for others
        logger.info(f"Embedding dimension: {dimension}")
        
        # Create CPU index first
        cpu_index = faiss.IndexFlatIP(dimension)  # Inner Product for cosine similarity
        cpu_index.add(embeddings.astype(np.float32))
        logger.info(f"Created FAISS CPU index with {cpu_index.ntotal} vectors")
        
        # Save FAISS index
        faiss.write_index(cpu_index, os.path.join(output_dir, "openclip_index.faiss"))
        
        # Save metadata as JSON
        metadata = {
            'image_paths': image_paths,
            'embedding_dimension': dimension,
            'total_embeddings': len(embeddings),
            'checkpoint': checkpoint,
            'created_at': datetime.now().isoformat()
        }
        
        with open(os.path.join(output_dir, "openclip_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("=" * 80)
        logger.info("✅ OpenCLIP embedding creation completed successfully!")
        logger.info(f"Total images processed: {len(valid_image_paths)}")
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("Files created:")
        logger.info(f"  - {npz_path}")
        logger.info(f"  - {os.path.join(output_dir, 'openclip_embeddings.npy')}")
        logger.info(f"  - {os.path.join(output_dir, 'openclip_index.faiss')}")
        logger.info(f"  - {os.path.join(output_dir, 'openclip_metadata.json')}")
        logger.info("=" * 80)
        
        # Clean up GPU memory
        if torch.cuda.is_available():
            openclip_model.cleanup()
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create OpenCLIP embeddings for product images with GPU optimization")
    parser.add_argument("--pictures-dir", default="pictures", help="Directory containing images")
    parser.add_argument("--csv-path", default="database_results/final_with_aws_shapes_20250625_155822.csv", 
                        help="Path to CSV database")
    parser.add_argument("--output-dir", default="indexes/openclip", help="Output directory for embeddings")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for processing (will be scaled by number of GPUs)")
    parser.add_argument("--checkpoint", default="epoch_008_model.pth", 
                        help="OpenCLIP checkpoint to use (epoch_008_model.pth or epoch_011_model.pth)")
    
    args = parser.parse_args()
    
    success = create_openclip_embeddings(
        pictures_dir=args.pictures_dir,
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        checkpoint=args.checkpoint
    )
    
    if success:
        logger.info("Embedding creation completed successfully!")
    else:
        logger.error("Embedding creation failed!")
        exit(1) 