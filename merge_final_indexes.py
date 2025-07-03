#!/usr/bin/env python3
"""
Merge the reduced FAISS index with the delta index to create the final index
"""

import numpy as np
import faiss
import json
import os
from datetime import datetime
import pandas as pd

def merge_indexes():
    print("🔀 Merging Reduced and Delta Indexes")
    print("=" * 80)
    
    # Load the reduced index info
    print("\n📊 Loading reduced index...")
    reduced_index_path = "indexes/v11_reduced_20250627_191631.faiss"
    reduced_embeddings_path = "indexes/v11_reduced_20250627_191631_embeddings.npy"
    reduced_metadata_path = "indexes/v11_reduced_20250627_191631_metadata.json"
    
    # Load reduced index
    reduced_index = faiss.read_index(reduced_index_path)
    reduced_embeddings = np.load(reduced_embeddings_path)
    with open(reduced_metadata_path, 'r') as f:
        reduced_metadata = json.load(f)
    
    print(f"  ✅ Reduced index: {reduced_embeddings.shape[0]} embeddings")
    print(f"  Dimension: {reduced_embeddings.shape[1]}")
    
    # Wait for delta index to be ready
    delta_index_path = "indexes/delta_gme_v11.faiss"
    delta_embeddings_path = "indexes/delta_gme_v11_embeddings.npy"
    delta_metadata_path = "indexes/delta_gme_v11_metadata.json"
    
    print("\n📊 Looking for delta index...")
    import time
    max_wait = 7200  # Wait up to 2 hours
    waited = 0
    
    while not os.path.exists(delta_index_path) or not os.path.exists(delta_embeddings_path):
        if waited > max_wait:
            print("❌ Delta index not found after 2 hours. Please check indexing process.")
            return
        
        print(f"  ⏳ Waiting for delta index... ({waited}s elapsed)")
        time.sleep(30)
        waited += 30
    
    # Load delta index
    print("\n📊 Loading delta index...")
    delta_index = faiss.read_index(delta_index_path)
    delta_embeddings = np.load(delta_embeddings_path)
    with open(delta_metadata_path, 'r') as f:
        delta_metadata = json.load(f)
    
    print(f"  ✅ Delta index: {delta_embeddings.shape[0]} embeddings")
    
    # Extract filename roots from paths
    def extract_filename_root(path):
        basename = os.path.basename(path)
        # Remove _O00 suffix and extension
        root = basename.replace('_O00', '').replace('.jpg', '').replace('.JPG', '')
        return root
    
    # Get filename roots
    reduced_roots = [extract_filename_root(p) for p in reduced_metadata['image_paths']]
    delta_roots = delta_metadata.get('filename_roots', [extract_filename_root(p) for p in delta_metadata['image_paths']])
    
    # Merge embeddings and metadata
    print("\n🔧 Merging indexes...")
    
    # Combine all embeddings
    all_embeddings = np.vstack([reduced_embeddings, delta_embeddings])
    all_paths = reduced_metadata['image_paths'] + delta_metadata['image_paths']
    all_roots = reduced_roots + delta_roots
    
    print(f"  Total embeddings: {all_embeddings.shape[0]}")
    
    # Create new combined index
    dimension = all_embeddings.shape[1]
    merged_index = faiss.IndexFlatL2(dimension)
    merged_index.add(all_embeddings.astype(np.float32))
    
    # Save merged index
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_name = f"v11_merged_{timestamp}"
    
    print(f"\n💾 Saving merged index as '{merged_name}'...")
    
    # Save FAISS index
    faiss.write_index(merged_index, f"indexes/{merged_name}.faiss")
    
    # Save embeddings
    np.save(f"indexes/{merged_name}_embeddings.npy", all_embeddings)
    
    # Save metadata
    merged_metadata = {
        'image_paths': all_paths,
        'filename_roots': all_roots,
        'creation_date': datetime.now().isoformat(),
        'source_indexes': {
            'reduced': reduced_metadata_path,
            'delta': delta_metadata_path
        },
        'total_embeddings': len(all_paths),
        'embedding_dimension': dimension,
        'reduced_count': len(reduced_metadata['image_paths']),
        'delta_count': len(delta_metadata['image_paths'])
    }
    
    with open(f"indexes/{merged_name}_metadata.json", 'w') as f:
        json.dump(merged_metadata, f, indent=2)
    
    # Create a symlink to the latest merged index for easy access
    latest_link = "indexes/v11_merged_latest.faiss"
    if os.path.exists(latest_link):
        os.remove(latest_link)
    os.symlink(f"{merged_name}.faiss", latest_link)
    
    # Also create symlinks for embeddings and metadata
    for ext in ['_embeddings.npy', '_metadata.json']:
        latest_link = f"indexes/v11_merged_latest{ext}"
        if os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(f"{merged_name}{ext}", latest_link)
    
    # Load new database to verify coverage
    print("\n📊 Verifying coverage with new database...")
    new_db = pd.read_csv('database_results/DB_FINAL_SIMILARIT_270615.csv')
    db_roots = set(new_db['filename_root'].unique())
    indexed_roots = set(all_roots)
    
    covered = db_roots.intersection(indexed_roots)
    missing = db_roots - indexed_roots
    
    print(f"\n✅ Coverage Report:")
    print(f"  Database products: {len(db_roots):,}")
    print(f"  Indexed products: {len(indexed_roots):,}")
    print(f"  Covered products: {len(covered):,} ({100*len(covered)/len(db_roots):.1f}%)")
    print(f"  Missing products: {len(missing):,} ({100*len(missing)/len(db_roots):.1f}%)")
    
    # Save coverage report
    coverage_report = {
        'database_products': len(db_roots),
        'indexed_products': len(indexed_roots),
        'covered_products': len(covered),
        'coverage_percentage': round(100*len(covered)/len(db_roots), 2),
        'missing_products': len(missing),
        'missing_roots': list(missing)
    }
    
    with open(f"indexes/{merged_name}_coverage_report.json", 'w') as f:
        json.dump(coverage_report, f, indent=2)
    
    print(f"\n✅ Merge complete!")
    print(f"📁 Final index: indexes/{merged_name}.faiss")
    print(f"🔗 Latest symlink: indexes/v11_merged_latest.faiss")
    
    return merged_name

if __name__ == "__main__":
    merge_indexes() 