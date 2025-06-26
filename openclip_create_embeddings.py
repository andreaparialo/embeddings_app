import os
import numpy as np
import pandas as pd
import logging
import glob
from datetime import datetime
from openclip_model import openclip_model
from openclip_data_loader import openclip_data_loader
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_openclip_embeddings(pictures_dir: str = "pictures", 
                             csv_path: str = "database_results/final_with_aws_shapes_20250625_155822.csv",
                             output_dir: str = "indexes/openclip",
                             batch_size: int = 256,  # Increased for GPU
                             checkpoint: str = "epoch_008_model.pth"):
    """
    Create OpenCLIP embeddings for all images in the pictures directory
    """
    try:
        logger.info("=" * 80)
        logger.info("Starting OpenCLIP embedding creation")
        logger.info(f"Pictures directory: {pictures_dir}")
        logger.info(f"CSV path: {csv_path}")
        logger.info(f"Output directory: {output_dir}")
        logger.info(f"Checkpoint: {checkpoint}")
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
        if not openclip_model.load_model(checkpoint):
            logger.error("Failed to load OpenCLIP model")
            return False
        
        # Process images in batches
        logger.info(f"Starting embedding generation for {len(valid_image_paths)} images...")
        if openclip_model.device == 'cuda':
            logger.info(f"Using GPU - this should take approximately {len(valid_image_paths) / 1000:.1f} - {len(valid_image_paths) / 500:.1f} minutes")
        else:
            logger.info(f"Using CPU - this will take approximately {len(valid_image_paths) / 100:.1f} - {len(valid_image_paths) / 50:.1f} minutes")
        start_time = datetime.now()
        
        # Show initial progress
        logger.info("Processing batches... (progress will be shown every 100 images)")
        
        embeddings = openclip_model.encode_images(valid_image_paths, batch_size=batch_size)
        
        if embeddings is None:
            logger.error("Failed to generate embeddings")
            return False
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        logger.info(f"Generated embeddings in {duration:.2f} seconds ({len(valid_image_paths)/duration:.2f} images/sec)")
        
        # Prepare metadata
        logger.info("Preparing metadata...")
        image_paths = [os.path.basename(path) for path in valid_image_paths]
        
        # Save embeddings and metadata
        logger.info("Saving embeddings and metadata...")
        
        # Save in NPZ format as specified by user
        npz_path = os.path.join(output_dir, "openclip_embeddings.npz")
        np.savez(npz_path,
                 embeddings=embeddings,
                 image_paths=np.array(image_paths))
        logger.info(f"Saved NPZ format to {npz_path}")
        
        # Also save in the format expected by data loader
        openclip_data_loader.embeddings = embeddings
        openclip_data_loader.metadata = {'image_paths': image_paths}
        
        # Create FAISS index
        logger.info("Creating FAISS index...")
        import faiss
        dimension = embeddings.shape[1]  # Should be 768 for OpenCLIP ViT-L-14
        logger.info(f"Embedding dimension: {dimension}")
        
        # Use Inner Product (IP) for cosine similarity since embeddings are normalized
        openclip_data_loader.index = faiss.IndexFlatIP(dimension)
        openclip_data_loader.index.add(embeddings.astype(np.float32))
        logger.info(f"Created FAISS index with {openclip_data_loader.index.ntotal} vectors")
        
        # Save everything
        openclip_data_loader.save_index(output_dir)
        
        logger.info("=" * 80)
        logger.info("✅ OpenCLIP embedding creation completed successfully!")
        logger.info(f"Total images processed: {len(valid_image_paths)}")
        logger.info(f"Embeddings shape: {embeddings.shape}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create OpenCLIP embeddings for product images")
    parser.add_argument("--pictures-dir", default="pictures", help="Directory containing images")
    parser.add_argument("--csv-path", default="database_results/final_with_aws_shapes_20250625_155822.csv", 
                        help="Path to CSV database")
    parser.add_argument("--output-dir", default="indexes/openclip", help="Output directory for embeddings")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for processing")
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