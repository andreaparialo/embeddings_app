#!/usr/bin/env python3
"""
Verify that measurement embeddings were extracted correctly
"""

import numpy as np
import json
import faiss

# Load measurement embeddings
embeddings = np.load('indexes/index_measurements/embeddings.npy')
print(f"Embeddings shape: {embeddings.shape}")

# Load metadata
with open('indexes/index_measurements/corrected_metadata.json', 'r') as f:
    metadata = json.load(f)

# Load index
index = faiss.read_index('indexes/index_measurements/index.faiss')
print(f"Index size: {index.ntotal}")

# Test a few embeddings by searching with them
print("\nTesting extracted embeddings...")
for test_idx in [0, 100, 1000, 10000]:
    # Use the extracted embedding as query
    query = embeddings[test_idx:test_idx+1]
    
    # Search in the index
    distances, indices = index.search(query, 5)
    
    print(f"\nTest {test_idx}:")
    print(f"  Top 5 results: {indices[0]}")
    print(f"  Distances: {distances[0]}")
    print(f"  Self found at position: {np.where(indices[0] == test_idx)[0]}")
    
    # The first result should be itself with distance ~0
    if indices[0][0] == test_idx and distances[0][0] < 0.01:
        print(f"  ✅ Embedding {test_idx} verified - found itself with distance {distances[0][0]:.6f}")
    else:
        print(f"  ❌ Embedding {test_idx} FAILED - expected self at position 0, got {indices[0][0]} with distance {distances[0][0]}")

# Check overlap with main index
print("\n\nChecking overlap with main index...")
from dual_index_data_loader import dual_index_loader

# Initialize to load metadata
csv_path = "database_results/DB_ACTIVE.csv"
dual_index_loader.initialize(csv_path)

print(f"\nOverlap statistics:")
print(f"  Main index products: {len(dual_index_loader.main_metadata['image_paths'])}")
print(f"  Measurement index products: {len(metadata['product_mapping'])}")
print(f"  Overlapping products: {len(dual_index_loader.measurement_to_main_mapping)}")

# Check a few specific products
print("\nChecking specific products:")
test_products = ['10003502M200', '1000350FG400', '1000350KY200']
for filename in test_products:
    normalized_path = f"db_pictures_512/{filename}.jpg"
    
    # Check in main index
    in_main = normalized_path in dual_index_loader.main_metadata['image_paths']
    
    # Check in measurement index
    in_measurement = False
    meas_idx = None
    for idx_str, path in metadata['product_mapping'].items():
        if path == normalized_path:
            in_measurement = True
            meas_idx = int(idx_str)
            break
    
    print(f"\n{filename}:")
    print(f"  In main index: {in_main}")
    print(f"  In measurement index: {in_measurement}")
    if in_measurement:
        print(f"  Measurement index position: {meas_idx}")
        # Check if embedding exists
        if meas_idx < len(embeddings):
            print(f"  Has embedding: ✅ (norm={np.linalg.norm(embeddings[meas_idx]):.4f})")
        else:
            print(f"  Has embedding: ❌ (index out of bounds)")