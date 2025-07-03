#!/usr/bin/env python3
"""
Create a reduced FAISS index matching the new database
"""

import pandas as pd
import numpy as np
import faiss
import json
import os
import time
from datetime import datetime

def create_reduced_index():
    print("🚀 Creating Reduced FAISS Index")
    print("=" * 80)
    
    # Load the reduction plan
    with open('faiss_reduction_plan.json', 'r') as f:
        reduction_plan = json.load(f)
    
    print(f"\n📊 Reduction Plan:")
    print(f"  Original size: {reduction_plan['original_size']:,} embeddings")
    print(f"  Target size: {reduction_plan['new_size']:,} embeddings")
    print(f"  Coverage: {reduction_plan['coverage']:.1f}%")
    print(f"  Missing products: {len(reduction_plan['missing_roots']):,}")
    
    # Ask user to confirm
    if len(reduction_plan['missing_roots']) > 0:
        print(f"\n⚠️  WARNING: {len(reduction_plan['missing_roots'])} products won't have embeddings!")
        print("Options:")
        print("  1. Create partial index (82.9% coverage) - faster")
        print("  2. Generate missing embeddings first - complete but slower")
        
        # For now, we'll proceed with option 1 (can be changed)
        print("\n➡️  Proceeding with Option 1: Create partial index")
    
    # Load existing FAISS index
    index_dir = "indexes"
    index_path = os.path.join(index_dir, "v11_complete_merged_20250625_115302.faiss")
    embeddings_path = os.path.join(index_dir, "v11_complete_merged_20250625_115302_embeddings.npy")
    metadata_path = os.path.join(index_dir, "v11_complete_merged_20250625_115302_metadata_fixed.json")
    
    if not os.path.exists(metadata_path):
        metadata_path = os.path.join(index_dir, "v11_complete_merged_20250625_115302_metadata.json")
    
    print(f"\n📁 Loading existing index...")
    start_time = time.time()
    
    # Load FAISS index
    cpu_index = faiss.read_index(index_path)
    print(f"  ✅ Loaded FAISS index: {cpu_index.ntotal} vectors")
    
    # Load embeddings
    embeddings = np.load(embeddings_path)
    print(f"  ✅ Loaded embeddings: {embeddings.shape}")
    
    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"  ✅ Loaded metadata: {len(metadata['image_paths'])} paths")
    
    load_time = time.time() - start_time
    print(f"  ⏱️  Load time: {load_time:.2f} seconds")
    
    # Create reduced embeddings and metadata
    print(f"\n🔧 Creating reduced data...")
    indices_to_keep = reduction_plan['indices_to_keep']
    
    # Extract reduced embeddings
    reduced_embeddings = embeddings[indices_to_keep]
    print(f"  ✅ Reduced embeddings: {reduced_embeddings.shape}")
    
    # Create reduced metadata
    reduced_metadata = {
        'image_paths': [metadata['image_paths'][i] for i in indices_to_keep],
        'original_indices': indices_to_keep,
        'creation_date': datetime.now().isoformat(),
        'source_index': 'v11_complete_merged_20250625_115302',
        'coverage': reduction_plan['coverage'],
        'missing_roots': reduction_plan['missing_roots'][:100]  # Save first 100 for reference
    }
    print(f"  ✅ Reduced metadata: {len(reduced_metadata['image_paths'])} paths")
    
    # Create new FAISS index
    print(f"\n🏗️  Building new FAISS index...")
    start_time = time.time()
    
    # Create a new flat index (same type as original)
    dimension = reduced_embeddings.shape[1]
    if cpu_index.metric_type == faiss.METRIC_INNER_PRODUCT:
        new_index = faiss.IndexFlatIP(dimension)
    else:
        new_index = faiss.IndexFlatL2(dimension)
    
    # Add embeddings to new index
    new_index.add(reduced_embeddings.astype(np.float32))
    print(f"  ✅ Added {new_index.ntotal} vectors to new index")
    
    build_time = time.time() - start_time
    print(f"  ⏱️  Build time: {build_time:.2f} seconds")
    
    # Save everything
    print(f"\n💾 Saving reduced index...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output paths
    reduced_index_path = os.path.join(index_dir, f"v11_reduced_{timestamp}.faiss")
    reduced_embeddings_path = os.path.join(index_dir, f"v11_reduced_{timestamp}_embeddings.npy")
    reduced_metadata_path = os.path.join(index_dir, f"v11_reduced_{timestamp}_metadata.json")
    
    # Save files
    faiss.write_index(new_index, reduced_index_path)
    print(f"  ✅ Saved index: {reduced_index_path}")
    
    np.save(reduced_embeddings_path, reduced_embeddings)
    print(f"  ✅ Saved embeddings: {reduced_embeddings_path}")
    
    with open(reduced_metadata_path, 'w') as f:
        json.dump(reduced_metadata, f, indent=2)
    print(f"  ✅ Saved metadata: {reduced_metadata_path}")
    
    # Create a summary report
    print(f"\n📊 Summary Report:")
    print(f"  Original embeddings: {reduction_plan['original_size']:,}")
    print(f"  Reduced embeddings: {len(indices_to_keep):,}")
    print(f"  Reduction: {(1 - len(indices_to_keep)/reduction_plan['original_size'])*100:.1f}%")
    print(f"  Coverage: {reduction_plan['coverage']:.1f}%")
    print(f"  Missing products: {len(reduction_plan['missing_roots']):,}")
    
    # Save missing products list for potential future indexing
    missing_products_path = 'missing_products_for_indexing.json'
    with open(missing_products_path, 'w') as f:
        # Load new database to get full info about missing products
        new_df = pd.read_csv('database_results/DB_FINAL_SIMILARIT_270615.csv')
        missing_df = new_df[new_df['filename_root'].isin(reduction_plan['missing_roots'])]
        
        missing_info = {
            'total_missing': len(reduction_plan['missing_roots']),
            'filename_roots': reduction_plan['missing_roots'],
            'sample_skus': missing_df.head(20)[['SKU_COD', 'filename_root', 'BRAND_DES', 'MODEL_COD']].to_dict('records')
        }
        json.dump(missing_info, f, indent=2)
    
    print(f"\n💾 Saved missing products list to: {missing_products_path}")
    
    print(f"\n✅ Reduced index creation complete!")
    print(f"\n🔧 Next steps:")
    print("  1. Update data_loader.py to use the new reduced index")
    print("  2. Or generate embeddings for missing products first")
    
    return {
        'index_path': reduced_index_path,
        'embeddings_path': reduced_embeddings_path,
        'metadata_path': reduced_metadata_path,
        'coverage': reduction_plan['coverage'],
        'missing_count': len(reduction_plan['missing_roots'])
    }

if __name__ == "__main__":
    create_reduced_index() 