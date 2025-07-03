#!/usr/bin/env python3
"""
Generate embeddings for missing products using GME model with LoRA
Optimized for batch processing on multiple GPUs
"""

import pandas as pd
import numpy as np
import torch
import json
import os
import faiss
from datetime import datetime
import time
from tqdm import tqdm
from PIL import Image
import gc

# Import from existing indexing scripts
import sys
sys.path.append('indexing_script_fast')
from lora_similarity_engine import LoraSimilarityEngine

def prepare_images_for_batch(image_paths, target_size=512):
    """Prepare and resize images for batch processing"""
    prepared_paths = []
    
    print(f"\n📸 Preparing {len(image_paths)} images...")
    for path in tqdm(image_paths, desc="Resizing images"):
        try:
            img = Image.open(path)
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize while maintaining aspect ratio
            img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            
            # Save to temp location
            temp_path = path.replace('/pictures/', '/tmp/resized_')
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            img.save(temp_path, quality=95)
            
            prepared_paths.append(temp_path)
            img.close()
        except Exception as e:
            print(f"⚠️  Error preparing {path}: {e}")
            prepared_paths.append(path)  # Use original if resize fails
    
    return prepared_paths

def generate_delta_embeddings():
    print("🚀 Generating Delta Embeddings with GME Model + LoRA")
    print("=" * 80)
    
    # Load the list of products with pictures
    missing_with_pics = pd.read_csv('missing_indexes_have_pictures.csv')
    print(f"\n📊 Products to index: {len(missing_with_pics):,}")
    
    # Prepare image paths
    pictures_dir = "pictures"
    image_data = []  # List of (filename_root, image_path) tuples
    
    print("\n🔍 Checking image files...")
    for idx, row in missing_with_pics.iterrows():
        filename_root = row['filename_root']
        
        # Try different image path variations
        image_found = False
        for ext in ['.jpg', '.JPG']:
            for suffix in ['_O00', '']:
                image_path = os.path.join(pictures_dir, f"{filename_root}{suffix}{ext}")
                if os.path.exists(image_path):
                    image_data.append((filename_root, image_path))
                    image_found = True
                    break
            if image_found:
                break
        
        if not image_found:
            print(f"⚠️  No image found for {filename_root}")
    
    print(f"✅ Found images for {len(image_data):,} products")
    
    # Check available GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\n🖥️  Available GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        # Show memory info
        mem_allocated = torch.cuda.memory_allocated(i) / 1024**3
        mem_total = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"      Memory: {mem_allocated:.1f}GB / {mem_total:.1f}GB")
    
    # Initialize the LoRA similarity engine
    print("\n🔧 Initializing LoRA Similarity Engine...")
    lora_path = "loras/v11-20250620-105815/checkpoint-1095"
    model_path = "gme-Qwen2-VL-7B-Instruct"
    
    engine = LoraSimilarityEngine(
        base_model_path=model_path,
        lora_adapter_path=lora_path,
        batch_size=8,  # Adjust based on GPU memory
        max_workers=num_gpus  # Use all available GPUs
    )
    
    # Extract just the image paths for processing
    image_paths = [path for _, path in image_data]
    filename_roots = [root for root, _ in image_data]
    
    # Optional: Prepare images (resize for faster processing)
    print("\n🎨 Preparing images for faster processing...")
    # prepared_paths = prepare_images_for_batch(image_paths, target_size=512)
    prepared_paths = image_paths  # Skip resizing for now
    
    # Generate embeddings in batches
    print("\n🚀 Generating embeddings...")
    start_time = time.time()
    
    embeddings = engine.generate_embeddings_batch(
        image_paths=prepared_paths,
        show_progress=True,
        checkpoint_interval=100  # Save progress every 100 images
    )
    
    # Convert to numpy array
    embeddings_array = np.array(embeddings, dtype=np.float32)
    print(f"\n✅ Generated embeddings shape: {embeddings_array.shape}")
    
    # Create metadata
    metadata = {
        'image_paths': [f"pictures/{os.path.basename(path)}" for path in image_paths],
        'filename_roots': filename_roots,
        'creation_date': datetime.now().isoformat(),
        'model': 'gme-Qwen2-VL-7B-Instruct',
        'lora_checkpoint': 'checkpoint-1095',
        'total_embeddings': len(embeddings),
        'embedding_dimension': embeddings_array.shape[1]
    }
    
    # Save delta index
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "indexes"
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, f"delta_gme_{timestamp}_embeddings.npy")
    np.save(embeddings_path, embeddings_array)
    print(f"\n💾 Saved embeddings: {embeddings_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, f"delta_gme_{timestamp}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"💾 Saved metadata: {metadata_path}")
    
    # Create FAISS index
    dimension = embeddings_array.shape[1]
    delta_index = faiss.IndexFlatL2(dimension)
    delta_index.add(embeddings_array)
    
    index_path = os.path.join(output_dir, f"delta_gme_{timestamp}.faiss")
    faiss.write_index(delta_index, index_path)
    print(f"💾 Saved FAISS index: {index_path}")
    
    # Performance stats
    elapsed_time = time.time() - start_time
    images_per_second = len(embeddings) / elapsed_time
    
    print(f"\n📊 Performance Summary:")
    print(f"  Total images processed: {len(embeddings):,}")
    print(f"  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
    print(f"  Speed: {images_per_second:.2f} images/second")
    print(f"  Time per image: {1000 * elapsed_time / len(embeddings):.2f} ms")
    
    # Save paths for merging
    delta_info = {
        'index_path': index_path,
        'embeddings_path': embeddings_path,
        'metadata_path': metadata_path,
        'num_embeddings': len(embeddings),
        'timestamp': timestamp,
        'embedding_dimension': dimension
    }
    
    with open('delta_index_info.json', 'w') as f:
        json.dump(delta_info, f, indent=2)
    
    print(f"\n✅ Delta embedding generation complete!")
    print(f"📁 Delta index info saved to: delta_index_info.json")
    
    # Clean up
    gc.collect()
    torch.cuda.empty_cache()
    
    return delta_info

if __name__ == "__main__":
    generate_delta_embeddings() 