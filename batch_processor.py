import os
# Fix MKL threading conflicts in multiprocessing
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import logging
import asyncio
import concurrent.futures
from threading import Lock
import torch
from collections import defaultdict
import time
import config_filtering

logger = logging.getLogger(__name__)

class BatchImageProcessor:
    """Optimized batch processing for multiple images"""
    
    def __init__(self, search_engine, data_loader, gme_model):
        self.search_engine = search_engine
        self.data_loader = data_loader
        self.gme_model = gme_model
        self.gpu_lock = Lock()  # Ensure GPU operations are thread-safe
        
    def process_image_groups_parallel(self, image_groups: Dict[str, Dict], 
                                    matching_cols: List[str],
                                    max_results_per_sku: int = 10,
                                    exclude_same_model: bool = False,
                                    allowed_statuses: List[str] = None,
                                    group_unisex: bool = False,
                                    dual_engine_enabled: bool = False,
                                    batch_size: int = 8,
                                    search_pool_size: int = 500) -> List[Dict]:
        """
        Process multiple image groups in parallel
        
        Args:
            image_groups: Dict mapping filename_root to group data
            batch_size: Number of images to process concurrently
        """
        start_time = time.time()
        logger.info(f"🚀 Starting parallel batch processing for {len(image_groups)} unique images")
        logger.info(f"🔧 Batch size: {batch_size}")
        
        # Split into batches
        all_results = []
        image_items = list(image_groups.items())
        
        # Process in batches
        for batch_start in range(0, len(image_items), batch_size):
            batch_end = min(batch_start + batch_size, len(image_items))
            batch_items = image_items[batch_start:batch_end]
            
            logger.info(f"📦 Processing batch {batch_start//batch_size + 1}/{(len(image_items) + batch_size - 1)//batch_size}")
            
            # Process batch concurrently
            batch_results = self._process_batch_concurrent(
                batch_items, matching_cols, max_results_per_sku,
                exclude_same_model, allowed_statuses, group_unisex,
                dual_engine_enabled, search_pool_size
            )
            
            all_results.extend(batch_results)
            
            # Log progress
            processed = min(batch_end, len(image_items))
            elapsed = time.time() - start_time
            rate = processed / elapsed
            eta = (len(image_items) - processed) / rate if rate > 0 else 0
            
            logger.info(f"⏱️ Progress: {processed}/{len(image_items)} images processed")
            logger.info(f"⚡ Rate: {rate:.1f} images/sec, ETA: {eta:.1f} seconds")
        
        # Sort results by Input_SKU and then by Similarity_Score (descending)
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df = results_df.sort_values(
                by=['Input_SKU', 'Similarity_Score'], 
                ascending=[True, False]
            )
            all_results = results_df.to_dict('records')
        
        total_time = time.time() - start_time
        logger.info(f"✅ Batch processing complete in {total_time:.1f} seconds")
        logger.info(f"📊 Average: {len(image_groups)/total_time:.1f} images/sec")
        
        return all_results
    
    def _process_batch_concurrent(self, batch_items: List[Tuple], 
                                matching_cols: List[str],
                                max_results_per_sku: int,
                                exclude_same_model: bool,
                                allowed_statuses: List[str],
                                group_unisex: bool,
                                dual_engine_enabled: bool,
                                search_pool_size: int = 500) -> List[Dict]:
        """Process a batch of images concurrently"""
        
        # Step 1: Encode all images in batch on GPU
        embeddings = self._batch_encode_images(batch_items)
        
        # Step 2: Batch FAISS search
        all_indices, all_distances = self._batch_faiss_search(embeddings, k=search_pool_size)
        
        # Step 3: Batch database lookup
        all_results_data = self._batch_database_lookup(all_indices, all_distances)
        
        # Step 4: Apply filters and format results for each image
        batch_results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            future_to_item = {}
            
            for i, (filename_root, group_data) in enumerate(batch_items):
                if i < len(all_results_data):
                    future = executor.submit(
                        self._process_single_image_results,
                        filename_root, group_data, all_results_data[i],
                        matching_cols, max_results_per_sku,
                        exclude_same_model, allowed_statuses,
                        group_unisex, dual_engine_enabled
                    )
                    future_to_item[future] = (filename_root, group_data)
            
            for future in concurrent.futures.as_completed(future_to_item):
                try:
                    results = future.result()
                    batch_results.extend(results)
                except Exception as e:
                    filename_root, _ = future_to_item[future]
                    logger.error(f"❌ Error processing {filename_root}: {e}")
        
        return batch_results
    
    def _batch_encode_images(self, batch_items: List[Tuple]) -> np.ndarray:
        """Get pre-computed embeddings for the batch - NO re-encoding needed!"""
        embeddings = []
        
        for filename_root, group_data in batch_items:
            embedding_found = False
            
            # Try variations in order of likelihood
            variations = [
                filename_root,  # Exact match
                filename_root.lower(),  # Lowercase version
                filename_root.upper(),  # Uppercase version
            ]
            
            # If it ends with a letter, also try without it
            if filename_root and filename_root[-1].isalpha():
                variations.append(filename_root[:-1])
                variations.append(filename_root[:-1].lower())
            
            # Try each variation
            for variant in variations:
                if variant in self.data_loader.filename_to_idx:
                    idx = self.data_loader.filename_to_idx[variant]
                    embedding = self.data_loader.embeddings[idx]
                    embeddings.append(embedding)
                    embedding_found = True
                    if variant != filename_root:
                        logger.debug(f"✅ Found embedding for {filename_root} using variant: {variant}")
                    else:
                        logger.debug(f"✅ Using pre-computed embedding for {filename_root}")
                    break
            
            if not embedding_found:
                # Still not found - this might be a new image not in the index
                logger.warning(f"⚠️ No pre-computed embedding found for {filename_root} or its variants {variations[:3]}")
                embeddings.append(np.zeros(3584))
        
        return np.array(embeddings).astype(np.float32)
    
    def _batch_faiss_search(self, embeddings: np.ndarray, k: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Perform batch FAISS search for multiple embeddings"""
        if len(embeddings) == 0:
            return np.array([]), np.array([])
        
        # Increase k to ensure we have enough results after filtering
        # With strict filters, many results get eliminated, so we need a larger initial pool
        logger.info(f"🔍 Searching for top {k} similar items per image (to ensure sufficient results after filtering)")
        
        # Search all embeddings at once
        distances, indices = self.data_loader.index.search(embeddings, k)
        return indices, distances
    
    def _batch_database_lookup(self, all_indices: np.ndarray, all_distances: np.ndarray) -> List[List[Dict]]:
        """Perform efficient batch database lookup using vectorized operations"""
        all_results = []
        
        # Flatten all indices for one big query
        flat_indices = all_indices.flatten()
        unique_indices = np.unique(flat_indices[flat_indices >= 0])
        
        # Get all filename_roots at once
        idx_to_filename = {idx: self.data_loader.idx_to_filename_root.get(idx) 
                          for idx in unique_indices}
        
        # Get all unique filename_roots
        unique_filenames = list(set(fn for fn in idx_to_filename.values() if fn))
        
        # Bulk query database
        if unique_filenames:
            bulk_df = self.data_loader.df[
                self.data_loader.df['filename_root'].isin(unique_filenames)
            ].set_index('filename_root')
            
            # Process each query's results
            for query_idx in range(len(all_indices)):
                query_results = []
                
                for idx, distance in zip(all_indices[query_idx], all_distances[query_idx]):
                    if idx >= 0 and idx in idx_to_filename:
                        filename_root = idx_to_filename[idx]
                        if filename_root and filename_root in bulk_df.index:
                            result = bulk_df.loc[filename_root].to_dict()
                            result['similarity_score'] = float(distance)
                            result['image_path'] = self._get_image_path(filename_root)
                            query_results.append(result)
                
                all_results.append(query_results)
        else:
            all_results = [[] for _ in range(len(all_indices))]
        
        return all_results
    
    def _process_single_image_results(self, filename_root: str, group_data: Dict,
                                    similar_results: List[Dict],
                                    matching_cols: List[str],
                                    max_results_per_sku: int,
                                    exclude_same_model: bool,
                                    allowed_statuses: List[str],
                                    group_unisex: bool,
                                    dual_engine_enabled: bool) -> List[Dict]:
        """Process results for a single image group"""
        source_item = group_data['source_item']
        group_skus = group_data['skus']
        
        # Extract matching filters
        matching_filters = {col: source_item.get(col) for col in matching_cols if col in source_item}
        
        # Apply filters
        filtered_results = similar_results
        initial_count = len(filtered_results)
        
        # Filter by matching columns
        if matching_filters:
            new_filtered_results = []
            for item in filtered_results:
                match = True
                for col, val in matching_filters.items():
                    item_value = item.get(col)
                    
                    # Check if this column should use range filtering
                    if config_filtering.is_range_filter_column(col):
                        # Range-based filtering for numeric columns
                        min_val, max_val = config_filtering.get_range_bounds(val, col)
                        if min_val is not None and max_val is not None:
                            try:
                                # Handle European decimal format (comma instead of dot)
                                if isinstance(item_value, str):
                                    item_value = item_value.replace(',', '.')
                                item_numeric = float(item_value)
                                if not (min_val <= item_numeric <= max_val):
                                    match = False
                                    break
                            except (ValueError, TypeError):
                                # If can't convert to numeric, no match
                                match = False
                                break
                    else:
                        # Handle NaN values
                        if pd.isna(val) and pd.isna(item_value):
                            continue  # Both NaN, consider as match
                        elif pd.isna(val) or pd.isna(item_value):
                            match = False  # One is NaN, other isn't
                            break
                        else:
                            # Normalize strings for comparison - remove trailing spaces and compare case-insensitive
                            item_str = str(item_value).strip().upper()
                            filter_str = str(val).strip().upper()
                            if item_str != filter_str:
                                match = False
                                break
                
                if match:
                    new_filtered_results.append(item)
            
            filtered_results = new_filtered_results
            if initial_count > 0:
                logger.debug(f"   Matching columns filter: {initial_count} → {len(filtered_results)} items")
        
        # Apply gender filtering if unisex grouping is enabled
        if group_unisex and 'USERGENDER_DES' in matching_filters:
            source_gender = matching_filters['USERGENDER_DES']
            if source_gender in ['MAN', 'WOMAN']:
                allowed_genders = [source_gender, 'UNISEX ADULT']
                filtered_results = [
                    item for item in filtered_results 
                    if item.get('USERGENDER_DES', '') in allowed_genders
                ]
        
        # Filter out same model code if requested
        if exclude_same_model:
            before_count = len(filtered_results)
            source_model_cod = source_item.get('MODEL_COD', '')
            filtered_results = [
                item for item in filtered_results 
                if item.get('MODEL_COD', '') != source_model_cod
            ]
            if before_count > 0:
                logger.debug(f"   Exclude same model filter: {before_count} → {len(filtered_results)} items")
        
        # Filter by allowed status codes
        if allowed_statuses:
            before_count = len(filtered_results)
            filtered_results = [
                item for item in filtered_results 
                if item.get('MD_SKU_STATUS_COD', '') in allowed_statuses
            ]
            if before_count > 0:
                logger.debug(f"   Status code filter: {before_count} → {len(filtered_results)} items")
        
        # Limit to requested number of results
        if len(filtered_results) < max_results_per_sku and initial_count > 100:
            logger.warning(f"   ⚠️ Only {len(filtered_results)} results passed filters out of {initial_count} similar items")
        filtered_results = filtered_results[:max_results_per_sku]
        
        # Format results for all SKUs in this group
        formatted_results = []
        for input_sku in group_skus:
            sku_source_item = group_data.get('sku_to_source', {}).get(input_sku, source_item)
            
            for similar_item in filtered_results:
                # Note: Baseline filters (date and status) are already applied in the FAISS search
                # No need to filter here again
                
                result_row = {
                    'Input_SKU': input_sku,
                    'Similar_SKU': similar_item.get('SKU_COD', ''),
                    'Similarity_Score': round(1 - similar_item.get('similarity_score', 0), 3)
                }
                
                # Add matching column values
                for col in matching_cols:
                    result_row[f'Source_{col}'] = sku_source_item.get(col, '')
                    result_row[f'Similar_{col}'] = similar_item.get(col, '')
                
                formatted_results.append(result_row)
        
        return formatted_results
    
    def _get_image_path(self, filename_root: str) -> str:
        """Get the image path for a filename_root"""
        # Use data_loader's implementation which handles path normalization
        path = self.data_loader.get_image_path(filename_root)
        return path if path else ""

# Global instance will be created when needed
batch_processor = None 