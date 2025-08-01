#!/usr/bin/env python3
"""
Parallel Filter Processor
Handles parallel filtering operations for batch processing using multiprocessing
"""

import os
# Fix MKL threading conflicts in multiprocessing
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional, Any
import logging
import multiprocessing as mp
from functools import partial
import time
import pickle
import config_filtering

logger = logging.getLogger(__name__)

class ParallelFilterProcessor:
    """Handles parallel filtering operations for improved performance"""
    
    def __init__(self, df: pd.DataFrame, filename_to_idx: Dict, idx_to_filename: Dict, 
                 embeddings_count: int, num_workers: Optional[int] = None):
        """
        Initialize the parallel filter processor
        
        Args:
            df: The main dataframe with product data
            filename_to_idx: Mapping from filename to embedding index
            idx_to_filename: Mapping from embedding index to filename
            embeddings_count: Total number of embeddings
            num_workers: Number of worker processes (defaults to CPU count)
        """
        self.df = df
        self.filename_to_idx = filename_to_idx
        self.idx_to_filename = idx_to_filename
        self.embeddings_count = embeddings_count
        self.num_workers = num_workers or mp.cpu_count()
        
        # Pre-compute baseline filters once
        self.baseline_mask = self._compute_baseline_mask()
        self.baseline_indices = self._get_baseline_embedding_indices()
        
        logger.info(f"Initialized ParallelFilterProcessor with {self.num_workers} workers")
        logger.info(f"Baseline filter reduces {len(self.df)} items to {self.baseline_mask.sum()}")
    
    def _compute_baseline_mask(self) -> pd.Series:
        """Compute baseline filter mask"""
        mask = pd.Series([True] * len(self.df), index=self.df.index)
        
        # Apply baseline status filter
        if config_filtering.ENABLE_BASELINE_STATUS_FILTER and config_filtering.BASELINE_STATUS_CODES:
            if 'MD_SKU_STATUS_COD' in self.df.columns:
                mask &= self.df['MD_SKU_STATUS_COD'].isin(config_filtering.BASELINE_STATUS_CODES)
        
        # Apply baseline date filter
        if config_filtering.ENABLE_BASELINE_DATE_FILTER:
            if 'STARTSKU_DATE' in self.df.columns:
                date_mask = ~self.df['STARTSKU_DATE'].apply(config_filtering.should_exclude_by_baseline_date)
                mask &= date_mask
        
        return mask
    
    def _get_baseline_embedding_indices(self) -> Set[int]:
        """Get embedding indices that pass baseline filters"""
        baseline_df = self.df[self.baseline_mask]
        indices = set()
        
        if 'filename_root' in baseline_df.columns:
            for filename_root in baseline_df['filename_root'].dropna().unique():
                if filename_root in self.filename_to_idx:
                    idx = self.filename_to_idx[filename_root]
                    if 0 <= idx < self.embeddings_count:
                        indices.add(idx)
        
        return indices
    
    def parallel_filter_queries(self, query_filters: List[Tuple[str, Dict]]) -> Dict[str, np.ndarray]:
        """
        Filter multiple queries in parallel
        
        Args:
            query_filters: List of (query_id, filters) tuples
            
        Returns:
            Dict mapping query_id to numpy array of embedding indices
        """
        start_time = time.time()
        
        # Group queries by unique filter combinations to avoid duplicate work
        filter_groups = {}
        query_to_filter_key = {}
        
        for query_id, filters in query_filters:
            filter_key = self._create_filter_key(filters)
            if filter_key not in filter_groups:
                filter_groups[filter_key] = filters
            query_to_filter_key[query_id] = filter_key
        
        logger.info(f"Processing {len(query_filters)} queries with {len(filter_groups)} unique filter combinations")
        
        # Prepare data for multiprocessing
        # We'll serialize the necessary data to avoid shared memory issues
        df_subset = self.df[self.baseline_mask].copy()
        
        # Create chunks for parallel processing
        filter_items = list(filter_groups.items())
        chunk_size = max(1, len(filter_items) // self.num_workers)
        chunks = [filter_items[i:i + chunk_size] for i in range(0, len(filter_items), chunk_size)]
        
        # Process in parallel
        with mp.Pool(processes=self.num_workers) as pool:
            # Create partial function with shared data
            process_func = partial(
                self._process_filter_chunk,
                df_subset=df_subset,
                filename_to_idx=self.filename_to_idx,
                embeddings_count=self.embeddings_count,
                baseline_indices=self.baseline_indices
            )
            
            # Map chunks to workers
            chunk_results = pool.map(process_func, chunks)
        
        # Combine results
        filter_results = {}
        for chunk_result in chunk_results:
            filter_results.update(chunk_result)
        
        # Map back to original query IDs
        query_results = {}
        for query_id, filter_key in query_to_filter_key.items():
            query_results[query_id] = filter_results[filter_key]
        
        elapsed = time.time() - start_time
        logger.info(f"Parallel filtering completed in {elapsed:.2f}s ({len(query_filters)/elapsed:.1f} queries/sec)")
        
        return query_results
    
    @staticmethod
    def _process_filter_chunk(filter_chunk: List[Tuple[str, Dict]], 
                            df_subset: pd.DataFrame,
                            filename_to_idx: Dict,
                            embeddings_count: int,
                            baseline_indices: Set[int]) -> Dict[str, np.ndarray]:
        """
        Process a chunk of filters in a worker process
        
        This is a static method to ensure it's pickleable for multiprocessing
        """
        results = {}
        
        for filter_key, filters in filter_chunk:
            # Apply filters to the dataframe subset
            mask = pd.Series([True] * len(df_subset), index=df_subset.index)
            
            for col, value in filters.items():
                if col in df_subset.columns and value is not None and value != '':
                    if config_filtering.is_range_filter_column(col):
                        # Range-based filtering
                        min_val, max_val = config_filtering.get_range_bounds(value, col)
                        if min_val is not None and max_val is not None:
                            col_values = df_subset[col].copy()
                            if col_values.dtype == 'object':
                                col_values = col_values.str.replace(',', '.', regex=False)
                            col_values = pd.to_numeric(col_values, errors='coerce')
                            mask &= (col_values >= min_val) & (col_values <= max_val)
                    else:
                        # Text-based filtering
                        if isinstance(value, list):
                            normalized_values = [str(v).strip().upper() for v in value]
                            mask &= df_subset[col].fillna('').astype(str).str.strip().str.upper().isin(normalized_values)
                        else:
                            normalized_value = str(value).strip().upper()
                            mask &= (df_subset[col].fillna('').astype(str).str.strip().str.upper() == normalized_value)
            
            # Get filtered dataframe
            filtered_df = df_subset[mask]
            
            # Convert to embedding indices
            embedding_indices = []
            if 'filename_root' in filtered_df.columns:
                for filename_root in filtered_df['filename_root'].dropna().unique():
                    if filename_root in filename_to_idx:
                        idx = filename_to_idx[filename_root]
                        if 0 <= idx < embeddings_count and idx in baseline_indices:
                            embedding_indices.append(idx)
            
            results[filter_key] = np.array(embedding_indices, dtype=np.int64)
        
        return results
    
    def _create_filter_key(self, filters: Dict) -> str:
        """Create a hashable key for filter combination"""
        normalized_filters = {}
        for k, v in filters.items():
            if isinstance(v, list):
                normalized_filters[k] = tuple(sorted(v))
            else:
                normalized_filters[k] = v
        
        import json
        return json.dumps(normalized_filters, sort_keys=True)
    
    def parallel_filter_measurement_index(self, 
                                        measurement_results: List[Tuple[str, np.ndarray, np.ndarray, Dict]],
                                        measurement_path_mapping: Dict,
                                        df: pd.DataFrame) -> Dict[str, List[Tuple[int, float]]]:
        """
        Filter measurement index results in parallel
        
        Args:
            measurement_results: List of (query_id, distances, indices, filters) tuples
            measurement_path_mapping: Mapping from measurement index to path info
            df: DataFrame with product information
            
        Returns:
            Dict mapping query_id to filtered (index, distance) tuples
        """
        start_time = time.time()
        
        # Prepare chunks for parallel processing
        chunk_size = max(1, len(measurement_results) // self.num_workers)
        chunks = [measurement_results[i:i + chunk_size] 
                 for i in range(0, len(measurement_results), chunk_size)]
        
        # Process in parallel
        with mp.Pool(processes=self.num_workers) as pool:
            process_func = partial(
                self._process_measurement_chunk,
                measurement_path_mapping=measurement_path_mapping,
                df=df
            )
            chunk_results = pool.map(process_func, chunks)
        
        # Combine results
        results = {}
        for chunk_result in chunk_results:
            results.update(chunk_result)
        
        elapsed = time.time() - start_time
        logger.info(f"Parallel measurement filtering completed in {elapsed:.2f}s")
        
        return results
    
    @staticmethod
    def _process_measurement_chunk(chunk: List[Tuple[str, np.ndarray, np.ndarray, Dict]],
                                 measurement_path_mapping: Dict,
                                 df: pd.DataFrame) -> Dict[str, List[Tuple[int, float]]]:
        """Process a chunk of measurement results"""
        results = {}
        
        for query_id, distances, indices, filters in chunk:
            filtered_results = []
            
            for i, (meas_idx, dist) in enumerate(zip(indices, distances)):
                if meas_idx < 0 or meas_idx not in measurement_path_mapping:
                    continue
                
                normalized_path = measurement_path_mapping[meas_idx]['normalized']
                filename_root = os.path.basename(normalized_path).replace('.jpg', '')
                
                # Get products matching this filename_root
                matching_products = df[df['filename_root'] == filename_root]
                
                if matching_products.empty:
                    continue
                
                # Check if any product matches all filters
                matches_filters = False
                for _, product in matching_products.iterrows():
                    all_match = True
                    for col, filter_val in filters.items():
                        if col in product:
                            product_val = product[col]
                            if isinstance(filter_val, list):
                                if product_val not in filter_val:
                                    all_match = False
                                    break
                            elif str(product_val).strip().upper() != str(filter_val).strip().upper():
                                all_match = False
                                break
                    
                    if all_match:
                        matches_filters = True
                        break
                
                if matches_filters:
                    filtered_results.append((meas_idx, float(dist)))
            
            results[query_id] = filtered_results
        
        return results