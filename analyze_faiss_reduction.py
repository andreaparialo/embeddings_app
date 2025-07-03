#!/usr/bin/env python3
"""
Analyze FAISS index reduction strategy
"""

import pandas as pd
import numpy as np
import faiss
import json
import os

def analyze_faiss_reduction():
    print("🔍 FAISS Index Reduction Analysis")
    print("=" * 80)
    
    # Load new database
    new_df = pd.read_csv('database_results/DB_FINAL_SIMILARIT_270615.csv')
    print(f"\n📊 New Database:")
    print(f"  Total products: {len(new_df):,}")
    print(f"  Unique filename_roots: {new_df['filename_root'].nunique():,}")
    
    # Get unique filename_roots from new database
    new_filename_roots = set(new_df['filename_root'].unique())
    
    # Check existing FAISS indexes
    index_dir = "indexes"
    print(f"\n📁 Checking indexes in {index_dir}:")
    
    # Try to load metadata to understand current index
    metadata_files = [
        "v11_complete_merged_20250625_115302_metadata_fixed.json",
        "v11_complete_merged_20250625_115302_metadata.json"
    ]
    
    metadata = None
    for metadata_file in metadata_files:
        metadata_path = os.path.join(index_dir, metadata_file)
        if os.path.exists(metadata_path):
            print(f"  Found metadata: {metadata_file}")
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            break
    
    if metadata:
        # Analyze metadata
        existing_paths = metadata.get('image_paths', [])
        print(f"  Total embeddings in index: {len(existing_paths):,}")
        
        # Extract filename_roots from paths
        existing_roots = set()
        for path in existing_paths:
            # Extract filename from path
            filename = os.path.basename(path)
            # Get filename_root (everything before first underscore)
            filename_root = filename.split('_')[0]
            existing_roots.add(filename_root)
        
        print(f"  Unique filename_roots in index: {len(existing_roots):,}")
        
        # Find overlap
        overlap = new_filename_roots & existing_roots
        missing_in_index = new_filename_roots - existing_roots
        extra_in_index = existing_roots - new_filename_roots
        
        print(f"\n📊 Overlap Analysis:")
        print(f"  Filename_roots in both: {len(overlap):,} ({len(overlap)/len(new_filename_roots)*100:.1f}%)")
        print(f"  In new DB but not in index: {len(missing_in_index):,}")
        print(f"  In index but not in new DB: {len(extra_in_index):,}")
        
        if missing_in_index:
            print(f"\n⚠️  Missing filename_roots (first 10):")
            for root in list(missing_in_index)[:10]:
                print(f"    - {root}")
        
        # Create mapping of which indices to keep
        indices_to_keep = []
        for idx, path in enumerate(existing_paths):
            filename = os.path.basename(path)
            filename_root = filename.split('_')[0]
            if filename_root in new_filename_roots:
                indices_to_keep.append(idx)
        
        print(f"\n✅ Reduction Strategy:")
        print(f"  Keep {len(indices_to_keep):,} out of {len(existing_paths):,} embeddings")
        print(f"  Reduction: {(1 - len(indices_to_keep)/len(existing_paths))*100:.1f}%")
        
        # Save the indices mapping
        reduction_plan = {
            'original_size': len(existing_paths),
            'new_size': len(indices_to_keep),
            'indices_to_keep': indices_to_keep,
            'missing_roots': list(missing_in_index),
            'coverage': len(overlap) / len(new_filename_roots) * 100
        }
        
        with open('faiss_reduction_plan.json', 'w') as f:
            json.dump(reduction_plan, f, indent=2)
        
        print(f"\n💾 Saved reduction plan to faiss_reduction_plan.json")
        
        # Check if we need to reindex missing products
        if missing_in_index:
            print(f"\n⚠️  WARNING: {len(missing_in_index)} products in new DB don't have embeddings!")
            print("  Options:")
            print("  1. Create reduced index without these products (faster)")
            print("  2. Generate embeddings for missing products first (complete)")
    else:
        print("❌ No metadata found - cannot analyze existing index")
    
    # Load old database to compare
    old_csv_path = 'database_results/final_with_aws_shapes_enriched.csv'
    if os.path.exists(old_csv_path):
        old_df = pd.read_csv(old_csv_path)
        print(f"\n📊 Old vs New Database:")
        print(f"  Old: {len(old_df):,} products")
        print(f"  New: {len(new_df):,} products")
        
        # Check which products were removed
        old_skus = set(old_df['SKU_COD'])
        new_skus = set(new_df['SKU_COD'])
        removed_skus = old_skus - new_skus
        
        if removed_skus:
            # Check status of removed products
            removed_df = old_df[old_df['SKU_COD'].isin(removed_skus)]
            status_counts = removed_df['MD_SKU_STATUS_COD'].value_counts()
            print(f"\n📊 Removed products by status:")
            for status, count in status_counts.items():
                print(f"    {status}: {count:,}")

if __name__ == "__main__":
    analyze_faiss_reduction() 