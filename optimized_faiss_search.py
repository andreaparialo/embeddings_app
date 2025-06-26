import numpy as np
import pandas as pd
import faiss
from typing import Dict, List, Tuple, Optional, Set
import logging
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)

class OptimizedFAISSSearch:
    """Optimized FAISS search with pre-filtering capabilities"""
    
    def __init__(self, index: faiss.Index, embeddings: np.ndarray, metadata_df: pd.DataFrame, 
                 filename_to_idx: Dict = None, idx_to_filename: Dict = None):
        self.index = index
        self.embeddings = embeddings
        self.df = metadata_df
        self.filter_cache = {}  # Cache for filter combinations
        self.max_cache_size = 10000  # Limit cache size
        
        # CRITICAL: Store filename mappings for proper index translation
        self.filename_to_idx = filename_to_idx or {}
        self.idx_to_filename = idx_to_filename or {}
        
        # Pre-compute some statistics
        self.total_items = len(self.df)
        self.embeddings_count = len(self.embeddings)
        
        logger.info(f"Initialized OptimizedFAISSSearch:")
        logger.info(f"  - DataFrame items: {self.total_items}")
        logger.info(f"  - Embeddings count: {self.embeddings_count}")
        logger.info(f"  - Filename mappings: {len(self.filename_to_idx)}")
        
        if self.total_items != self.embeddings_count:
            logger.warning(f"⚠️ DataFrame ({self.total_items}) and embeddings ({self.embeddings_count}) size mismatch!")
            logger.warning("  This is expected if DataFrame contains SKUs and embeddings are per filename_root")
        
    def get_filtered_indices(self, filters: Dict) -> np.ndarray:
        """Get EMBEDDING indices of items that pass all filters"""
        if not filters:
            # Return all embedding indices
            return np.arange(self.embeddings_count)
        
        # Create cache key
        cache_key = json.dumps(filters, sort_keys=True)
        
        # Check cache
        if cache_key in self.filter_cache:
            return self.filter_cache[cache_key]
        
        # Apply filters to dataframe
        mask = pd.Series([True] * len(self.df), index=self.df.index)
        
        for col, value in filters.items():
            if col in self.df.columns:
                if value is not None and value != '':
                    # Handle different types of filtering
                    if isinstance(value, list):
                        mask &= self.df[col].isin(value)
                    else:
                        mask &= (self.df[col] == value)
        
        # Get filtered dataframe
        filtered_df = self.df[mask]
        
        # CRITICAL: Convert DataFrame rows to embedding indices
        embedding_indices = []
        
        if 'filename_root' in filtered_df.columns:
            # Get unique filename_roots from filtered data
            unique_filename_roots = filtered_df['filename_root'].dropna().unique()
            
            # Convert filename_roots to embedding indices
            for filename_root in unique_filename_roots:
                if filename_root in self.filename_to_idx:
                    embedding_idx = self.filename_to_idx[filename_root]
                    # Validate the embedding index
                    if 0 <= embedding_idx < self.embeddings_count:
                        embedding_indices.append(embedding_idx)
                    else:
                        logger.warning(f"Invalid embedding index {embedding_idx} for {filename_root}")
        else:
            # Fallback: assume DataFrame indices map directly (old behavior)
            logger.warning("No filename_root column found, using direct index mapping")
            df_indices = filtered_df.index.values
            # Only include indices that are within embeddings bounds
            embedding_indices = df_indices[df_indices < self.embeddings_count]
        
        filtered_indices = np.array(embedding_indices, dtype=np.int64)
        
        # Cache result if cache not too large
        if len(self.filter_cache) < self.max_cache_size:
            self.filter_cache[cache_key] = filtered_indices
        
        logger.debug(f"Filtered {len(filtered_df)}/{self.total_items} DataFrame items → {len(filtered_indices)} embedding indices")
        
        return filtered_indices
    
    def search_with_prefilter(
        self, 
        query_embedding: np.ndarray, 
        filters: Dict, 
        top_k: int,
        exclude_indices: Optional[Set[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search FAISS index with pre-filtered indices"""
        
        # Get filtered indices
        valid_indices = self.get_filtered_indices(filters)
        
        # Exclude specific indices if needed (e.g., self-matches)
        if exclude_indices:
            valid_indices = np.array([idx for idx in valid_indices if idx not in exclude_indices])
        
        if len(valid_indices) == 0:
            logger.debug(f"No items match filters: {filters}")
            # Return empty results with proper shape
            return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)
        
        # Ensure query embedding is the right shape and type
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        query_embedding = query_embedding.astype(np.float32)
        
        # If we're filtering out most items, create a temporary index
        filter_ratio = len(valid_indices) / self.total_items
        
        # For GPU indexes, prefer subset index for very selective filters to avoid k-limit issues
        gpu_index = hasattr(self.index, '__class__') and 'Gpu' in str(self.index.__class__)
        
        # Use subset index if:
        # 1. Very selective filter (< 10% of items) OR
        # 2. GPU index with highly selective filter (< 5%) to avoid k-limit issues OR
        # 3. Very few valid indices (< 100) regardless of ratio
        use_subset = (
            (filter_ratio < 0.1 and len(valid_indices) > top_k) or
            (gpu_index and filter_ratio < 0.05) or
            len(valid_indices) < 100
        )
        
        if use_subset:
            logger.debug(f"Using subset index: {len(valid_indices)} valid items ({filter_ratio*100:.1f}%)")
            return self._search_with_subset_index(query_embedding, valid_indices, top_k)
        else:
            logger.debug(f"Using IDSelector: {len(valid_indices)} valid items ({filter_ratio*100:.1f}%)")
            return self._search_with_id_selector(query_embedding, valid_indices, top_k)
    
    def _search_with_id_selector(
        self, 
        query_embedding: np.ndarray, 
        valid_indices: np.ndarray, 
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Search using FAISS IDSelector"""
        
        # Create ID selector
        id_selector = faiss.IDSelectorArray(valid_indices)
        
        # Adjust top_k if necessary
        search_k = min(top_k, len(valid_indices))
        
        # Check if index supports search_preassigned
        if hasattr(self.index, 'search_preassigned'):
            # IVF index
            params = faiss.SearchParametersIVF()
            params.sel = id_selector
            params.nprobe = getattr(self.index, 'nprobe', 32)
            
            D = np.empty((1, search_k), dtype=np.float32)
            I = np.empty((1, search_k), dtype=np.int64)
            
            self.index.search_preassigned(
                query_embedding,
                search_k,
                I, D,
                params
            )
        else:
            # For other index types, use regular search with post-filtering
            # Search for more to ensure we get enough valid results
            search_multiplier = max(3, int(self.total_items / len(valid_indices)))
            temp_k = min(search_k * search_multiplier, self.total_items)
            
            # GPU indexes have a limit of 2048 for k-selection
            gpu_max_k = 2048
            if hasattr(self.index, '__class__') and 'Gpu' in str(self.index.__class__):
                temp_k = min(temp_k, gpu_max_k)
                logger.debug(f"GPU index detected, limiting k to {temp_k}")
            
            D_temp, I_temp = self.index.search(query_embedding, temp_k)
            
            # Filter results
            valid_set = set(valid_indices)
            D, I = [], []
            
            for d, i in zip(D_temp[0], I_temp[0]):
                if i in valid_set:
                    D.append(d)
                    I.append(i)
                    if len(D) >= search_k:
                        break
            
            # Pad if necessary
            while len(D) < search_k:
                D.append(float('inf'))
                I.append(-1)
            
            D = np.array([D[:search_k]])
            I = np.array([I[:search_k]])
        
        return D, I
    
    def _search_with_subset_index(
        self, 
        query_embedding: np.ndarray, 
        valid_indices: np.ndarray, 
        top_k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create temporary subset index for very selective filters"""
        
        # CRITICAL: Bounds checking to prevent IndexError
        if len(valid_indices) > 0:
            max_idx = np.max(valid_indices)
            if max_idx >= self.embeddings_count:
                logger.error(f"Index out of bounds: max index {max_idx} >= embeddings size {self.embeddings_count}")
                # Filter out invalid indices
                valid_indices = valid_indices[valid_indices < self.embeddings_count]
                if len(valid_indices) == 0:
                    logger.warning("No valid indices after bounds filtering")
                    return np.array([[]], dtype=np.float32), np.array([[]], dtype=np.int64)
        
        # Get embeddings for valid indices
        subset_embeddings = self.embeddings[valid_indices]
        
        # Create temporary flat index
        if subset_embeddings.shape[1] == self.embeddings.shape[1]:
            if hasattr(self.index, 'metric_type'):
                if self.index.metric_type == faiss.METRIC_INNER_PRODUCT:
                    temp_index = faiss.IndexFlatIP(subset_embeddings.shape[1])
                else:
                    temp_index = faiss.IndexFlatL2(subset_embeddings.shape[1])
            else:
                temp_index = faiss.IndexFlatL2(subset_embeddings.shape[1])
            
            # Add embeddings
            temp_index.add(subset_embeddings.astype(np.float32))
            
            # Search
            search_k = min(top_k, len(valid_indices))
            D, I_local = temp_index.search(query_embedding, search_k)
            
            # Map back to original indices
            I = np.array([[valid_indices[i] if i >= 0 else -1 for i in I_local[0]]])
            
            return D, I
        else:
            logger.error(f"Embedding dimension mismatch: {subset_embeddings.shape[1]} vs {self.embeddings.shape[1]}")
            return np.array([[]]), np.array([[]])
    
    def batch_search_with_prefilter(
        self,
        query_embeddings: List[Tuple[str, np.ndarray, Dict]],  # (id, embedding, filters)
        top_k: int,
        max_workers: int = 4
    ) -> Dict[str, List[Tuple[float, int]]]:
        """Batch search with different filters per query"""
        
        start_time = time.time()
        results = {}
        
        # Group queries by filter combination for efficiency
        filter_groups = defaultdict(list)
        for query_id, embedding, filters in query_embeddings:
            filter_key = json.dumps(filters, sort_keys=True)
            filter_groups[filter_key].append((query_id, embedding))
        
        logger.info(f"Processing {len(query_embeddings)} queries with {len(filter_groups)} unique filter combinations")
        
        # Process each filter group in parallel
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for filter_key, queries in filter_groups.items():
                filters = json.loads(filter_key)
                future = executor.submit(
                    self._process_filter_group,
                    queries, filters, top_k
                )
                futures.append(future)
            
            # Collect results
            for future in as_completed(futures):
                group_results = future.result()
                results.update(group_results)
        
        elapsed = time.time() - start_time
        logger.info(f"Batch search completed in {elapsed:.2f}s ({len(query_embeddings)/elapsed:.1f} queries/sec)")
        
        return results
    
    def _process_filter_group(
        self,
        queries: List[Tuple[str, np.ndarray]],
        filters: Dict,
        top_k: int
    ) -> Dict[str, List[Tuple[float, int]]]:
        """Process all queries with the same filter requirements"""
        
        # Get filtered indices once for this group
        valid_indices = self.get_filtered_indices(filters)
        
        if len(valid_indices) == 0:
            return {query_id: [] for query_id, _ in queries}
        
        results = {}
        
        # Process each query
        for query_id, embedding in queries:
            D, I = self.search_with_prefilter(embedding, filters, top_k)
            
            # Convert to list of (distance, index) tuples
            query_results = []
            for d, i in zip(D[0], I[0]):
                if i >= 0:  # Valid result
                    query_results.append((float(d), int(i)))
            
            results[query_id] = query_results
        
        return results
    
    def clear_cache(self):
        """Clear the filter cache"""
        self.filter_cache.clear()
        logger.info("Filter cache cleared")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        return {
            "cache_size": len(self.filter_cache),
            "max_cache_size": self.max_cache_size,
            "total_items": self.total_items
        } 