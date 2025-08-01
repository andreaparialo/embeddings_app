#!/usr/bin/env python3
"""
Dual index search that retrieves ALL products matching filters, not just top K
"""

import numpy as np
import logging
from typing import Dict, List, Tuple
from dual_index_data_loader import dual_index_loader

logger = logging.getLogger(__name__)

def search_all_filtered_products(query_embedding_main: np.ndarray, 
                               query_embedding_meas: np.ndarray,
                               filters: Dict,
                               search_mode: str = "filtered") -> List[Dict]:
    """
    Search for ALL products that match the filters and combine scores
    
    Args:
        query_embedding_main: The main (GME) embedding for the query
        query_embedding_meas: The measurement embedding for the query (can be None)
        filters: Filter criteria to apply
        search_mode: "global" or "filtered"
    
    Returns:
        List of all products with combined scores
    """
    
    # In filtered mode, we need to search with a very large top_k to get all filtered products
    # In global mode, we search all and filter later
    if search_mode == "filtered":
        # Search with filters, using a large top_k to get all matching products
        max_k = 10000  # Large enough to get all filtered products
        main_distances, main_indices = dual_index_loader.search_main_index(
            query_embedding_main, max_k, filters
        )
        
        if query_embedding_meas is not None:
            meas_distances, meas_indices = dual_index_loader.search_measurement_index_with_filters(
                query_embedding_meas, max_k, filters
            )
        else:
            meas_distances = np.array([])
            meas_indices = np.array([])
    else:
        # Global mode: search without filters, get more results (but respect GPU limit)
        max_k = min(dual_index_loader.main_index.ntotal, 2048)  # GPU limit
        main_distances, main_indices = dual_index_loader.search_main_index(
            query_embedding_main, max_k, filters=None
        )
        
        if query_embedding_meas is not None:
            max_k_meas = min(dual_index_loader.measurement_index.ntotal, 2048)  # GPU limit
            meas_distances, meas_indices = dual_index_loader.search_measurement_index(
                query_embedding_meas, max_k_meas
            )
        else:
            meas_distances = np.array([])
            meas_indices = np.array([])
    
    # Build a comprehensive result set
    all_results = {}
    
    # Process main index results
    for i, (idx, dist) in enumerate(zip(main_indices, main_distances)):
        if idx < 0:  # Invalid index
            continue
            
        # Get filename root
        if idx < len(dual_index_loader.main_metadata['image_paths']):
            image_path = dual_index_loader.main_metadata['image_paths'][idx]
            filename_root = image_path.split('/')[-1].replace('.jpg', '')
            
            all_results[filename_root] = {
                'filename_root': filename_root,
                'image_path': image_path,
                'main_distance': float(dist),
                'main_rank': i + 1,
                'measurement_distance': None,
                'measurement_rank': None
            }
    
    # Process measurement index results
    for i, (idx, dist) in enumerate(zip(meas_indices, meas_distances)):
        if idx < 0:  # Invalid index
            continue
            
        # Get filename root
        if idx in dual_index_loader.measurement_path_mapping:
            path_info = dual_index_loader.measurement_path_mapping[idx]
            image_path = path_info['normalized']
            filename_root = image_path.split('/')[-1].replace('.jpg', '')
            
            if filename_root in all_results:
                # Update existing entry
                all_results[filename_root]['measurement_distance'] = float(dist)
                all_results[filename_root]['measurement_rank'] = i + 1
            else:
                # New entry from measurement only
                all_results[filename_root] = {
                    'filename_root': filename_root,
                    'image_path': image_path,
                    'main_distance': None,
                    'main_rank': None,
                    'measurement_distance': float(dist),
                    'measurement_rank': i + 1
                }
    
    # Convert distances to similarities and calculate combined scores
    for result in all_results.values():
        # Convert distances to similarities (lower distance = higher similarity)
        if result['main_distance'] is not None:
            result['main_similarity'] = 1.0 / (1.0 + result['main_distance'])
        else:
            result['main_similarity'] = 0.0
            
        if result['measurement_distance'] is not None:
            result['measurement_similarity'] = 1.0 / (1.0 + result['measurement_distance'])
        else:
            result['measurement_similarity'] = 0.0
        
        # Calculate combined score
        if result['main_distance'] is not None and result['measurement_distance'] is not None:
            # Both indexes have results
            result['combined_score'] = (
                result['main_similarity'] * dual_index_loader.main_weight +
                result['measurement_similarity'] * dual_index_loader.measurement_weight
            )
            result['score_source'] = 'both_searches'
        elif result['main_distance'] is not None:
            # Only main index has result
            result['combined_score'] = result['main_similarity'] * dual_index_loader.main_weight
            result['score_source'] = 'main_search'
        else:
            # Only measurement index has result
            result['combined_score'] = result['measurement_similarity'] * dual_index_loader.measurement_weight
            result['score_source'] = 'measurement_search'
        
        # Add index membership info
        in_main = result['filename_root'] in dual_index_loader.main_filename_roots
        in_meas = result['filename_root'] in dual_index_loader.measurement_filename_roots
        
        if in_main and in_meas:
            result['index_membership'] = 'both_indexes'
        elif in_main:
            result['index_membership'] = 'main_only'
        else:
            result['index_membership'] = 'measurement_only'
    
    # Convert to list and sort by combined score
    results_list = list(all_results.values())
    results_list.sort(key=lambda x: x['combined_score'], reverse=True)
    
    return results_list