import os
# Fix MKL threading conflicts in multiprocessing
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Set, Optional
import logging
import concurrent.futures
from threading import Lock
import torch
from collections import defaultdict
import time
from optimized_faiss_search import OptimizedFAISSSearch
import config_filtering

logger = logging.getLogger(__name__)

class OptimizedBatchProcessor:
    """Optimized batch processing with pre-filtering for faster searches"""
    
    def __init__(self, search_engine, data_loader, gme_model):
        self.search_engine = search_engine
        self.data_loader = data_loader
        self.gme_model = gme_model
        self.gpu_lock = Lock()
        
        # Initialize optimized search with proper filename mappings
        if not hasattr(data_loader, 'filename_to_idx') or not data_loader.filename_to_idx:
            logger.warning("⚠️ Data loader missing filename_to_idx mappings!")
            logger.warning("  Pre-filtering may not work correctly for SKU-based data")
        
        self.optimized_search = OptimizedFAISSSearch(
            index=data_loader.index,
            embeddings=data_loader.embeddings,
            metadata_df=data_loader.df,
            filename_to_idx=getattr(data_loader, 'filename_to_idx', {}),
            idx_to_filename=getattr(data_loader, 'idx_to_filename_root', {})
        )
        
    def process_image_groups_with_prefilter(self, image_groups: Dict[str, Dict], 
                                           matching_cols: List[str],
                                           max_results_per_sku: int = 50,
                                           exclude_same_model: bool = False,
                                           allowed_statuses: List[str] = None,
                                           group_unisex: bool = False,
                                           dual_engine_enabled: bool = False,
                                           batch_size: int = 8) -> List[Dict]:
        """
        Process multiple image groups with pre-filtering for optimal performance
        """
        start_time = time.time()
        logger.info(f"🚀 Starting optimized batch processing with PRE-FILTERING")
        logger.info(f"📊 Processing {len(image_groups)} unique images")
        logger.info(f"🔧 Matching columns: {matching_cols}")
        
        # Define which columns to use for pre-filtering vs post-filtering
        prefilter_columns = config_filtering.get_prefilter_columns()
        # IMPORTANT: Only columns that are BOTH in PREFILTER_COLUMNS AND selected in the UI 
        # as matching columns will be used for pre-filtering
        prefilter_columns = [col for col in prefilter_columns if col in matching_cols]
        
        logger.info(f"🔍 Pre-filter columns (applied before similarity search): {prefilter_columns}")
        logger.info(f"📋 Post-filter columns (applied after similarity search): {[col for col in matching_cols if col not in prefilter_columns]}")
        
        # Prepare queries with their filters
        queries = []
        query_metadata = {}
        
        for filename_root, group_data in image_groups.items():
            source_item = group_data['source_item']
            
            # CRITICAL FIX: Only use essential filters for pre-filtering
            # The rest will be applied AFTER similarity search
            prefilters = {}
            for col in prefilter_columns:
                if col in matching_cols and col in source_item and source_item[col] is not None:
                    prefilters[col] = source_item[col]
            
            # Note: Status filter is now a baseline filter applied automatically in FAISS search
            # No need to add it to prefilters
            
            # Handle gender filtering for unisex grouping
            if group_unisex and 'USERGENDER_DES' in prefilters:
                source_gender = prefilters['USERGENDER_DES']
                if source_gender in ['MAN', 'WOMAN']:
                    prefilters['USERGENDER_DES'] = [source_gender, 'UNISEX ADULT']
            
            # Get embedding
            embedding = self._get_embedding_for_filename(filename_root)
            if embedding is not None:
                query_id = f"query_{len(queries)}"
                queries.append((query_id, embedding, prefilters))
                query_metadata[query_id] = {
                    'filename_root': filename_root,
                    'group_data': group_data,
                    'source_item': source_item,
                    'exclude_model': source_item.get('MODEL_COD') if exclude_same_model else None,
                    'postfilter_columns': [col for col in matching_cols if col not in prefilter_columns],  # Store columns for post-filtering
                    'all_filters': {col: source_item.get(col) for col in matching_cols if col in source_item}  # Store all filters for post-processing
                }
        
        logger.info(f"📋 Prepared {len(queries)} queries for batch search")
        
        # Group queries by filter combination for efficiency
        filter_groups = defaultdict(list)
        for query_id, embedding, filters in queries:
            filter_key = self._create_filter_key(filters)
            filter_groups[filter_key].append((query_id, embedding, filters))
        
        logger.info(f"🔍 Found {len(filter_groups)} unique filter combinations")
        
        # Log filter effectiveness
        total_queries_with_results = 0
        for i, (filter_key, group) in enumerate(filter_groups.items()):
            if i < 5:  # Show first 5 filter groups
                filters = group[0][2]  # Get filters from first query in group
                filtered_indices = self.optimized_search.get_filtered_indices(filters)
                # Note: filtered_indices are now EMBEDDING indices, not DataFrame indices
                logger.info(f"   Filter group {i+1}: {len(filtered_indices)} embedding indices ({len(filtered_indices)/self.optimized_search.embeddings_count*100:.1f}% of embeddings)")
                if len(filtered_indices) > 0:
                    total_queries_with_results += len(group)
                
                # Show sample filter for debugging
                if i == 0:
                    logger.debug(f"     Sample filter: {filters}")
        
        if total_queries_with_results == 0:
            logger.warning("⚠️ WARNING: Filters are too restrictive! Consider relaxing some filter criteria.")
        
        # Perform batch search with pre-filtering
        logger.info("🔥 Starting batch FAISS search with PRE-FILTERING...")
        search_results = self.optimized_search.batch_search_with_prefilter(
            query_embeddings=queries,
            top_k=max_results_per_sku * 3,  # Get extra for post-filtering
            max_workers=4  # Use all 4 GPUs
        )
        
        # Process results
        all_results = []
        
        for query_id, result_indices in search_results.items():
            metadata = query_metadata[query_id]
            filename_root = metadata['filename_root']
            group_data = metadata['group_data']
            exclude_model = metadata['exclude_model']
            postfilter_columns = metadata['postfilter_columns']
            all_filters = metadata['all_filters']
            
            # Convert embedding indices to full results
            similar_items = []
            items_before_postfilter = 0
            items_filtered_out = 0
            
            # Get more results initially to account for post-filtering
            for distance, embedding_idx in result_indices:
                if embedding_idx >= 0:
                    # Convert embedding index back to filename_root
                    if embedding_idx in self.data_loader.idx_to_filename_root:
                        similar_filename_root = self.data_loader.idx_to_filename_root[embedding_idx]
                        
                        # Find all rows in DataFrame with this filename_root
                        matching_rows = self.data_loader.df[
                            self.data_loader.df['filename_root'] == similar_filename_root
                        ]
                        
                        # Process each matching row (multiple SKUs can have same filename_root)
                        for _, row in matching_rows.iterrows():
                            item = row.to_dict()
                            item['similarity_score'] = distance
                            
                            # Apply model exclusion if needed
                            if exclude_model and item.get('MODEL_COD') == exclude_model:
                                continue
                            
                            items_before_postfilter += 1
                            
                            # CRITICAL: Apply post-filters here
                            skip_item = False
                            for col in postfilter_columns:
                                if col in all_filters and all_filters[col] is not None:
                                    item_value = item.get(col)
                                    filter_value = all_filters[col]
                                    
                                    # Check if this column should use range filtering
                                    if config_filtering.is_range_filter_column(col):
                                        # Range-based filtering for numeric columns
                                        min_val, max_val = config_filtering.get_range_bounds(filter_value, col)
                                        if min_val is not None and max_val is not None:
                                            try:
                                                # Handle European decimal format (comma instead of dot)
                                                if isinstance(item_value, str):
                                                    item_value = item_value.replace(',', '.')
                                                item_numeric = float(item_value)
                                                
                                                if not (min_val <= item_numeric <= max_val):
                                                    # Debug logging for range filter mismatches
                                                    if col == 'FRONT_HEIGHT_VAL' and abs(item_numeric - filter_value) / filter_value > 0.2:
                                                        logger.debug(f"Range filter mismatch on {col}: source={filter_value}, item={item_numeric}, range=[{min_val:.2f}, {max_val:.2f}]")
                                                    skip_item = True
                                                    break
                                            except (ValueError, TypeError) as e:
                                                # If can't convert to numeric, skip
                                                logger.debug(f"Failed to convert {col} value '{item_value}' to numeric: {e}")
                                                skip_item = True
                                                break
                                    else:
                                        # Handle different comparison types
                                        if pd.isna(item_value) and pd.isna(filter_value):
                                            continue  # Both NaN, consider as match
                                        elif pd.isna(item_value) or pd.isna(filter_value):
                                            skip_item = True  # One is NaN, other isn't
                                            break
                                        else:
                                            # Normalize strings for comparison - remove trailing spaces and compare case-insensitive
                                            item_str = str(item_value).strip().upper()
                                            filter_str = str(filter_value).strip().upper()
                                            if item_str != filter_str:
                                                skip_item = True
                                                break
                            
                            if not skip_item:
                                similar_items.append(item)
                            else:
                                items_filtered_out += 1
                            
                            if len(similar_items) >= max_results_per_sku:
                                break
                    else:
                        logger.warning(f"Embedding index {embedding_idx} not found in idx_to_filename_root mapping")
                
                if len(similar_items) >= max_results_per_sku:
                    break
            
            # Debug: Log how many items passed post-filtering
            if postfilter_columns:
                if items_filtered_out > 0 or len(similar_items) < 20:
                    logger.info(f"Image {filename_root}: {items_before_postfilter} → {len(similar_items)} items (filtered out {items_filtered_out} by post-filters: {postfilter_columns})")
            
            # Format results for all SKUs in this group
            for input_sku in group_data['skus']:
                for similar_item in similar_items:
                    # Note: Baseline filters (date and status) are already applied in the FAISS search
                    # No need to filter here again
                    
                    result_row = {
                        'Input_SKU': input_sku,
                        'Similar_SKU': similar_item.get('SKU_COD', ''),
                        'Similarity_Score': round(1 - similar_item.get('similarity_score', 0), 3)
                    }
                    
                    # Add matching column values
                    for col in matching_cols:
                        result_row[f'Source_{col}'] = group_data['source_item'].get(col, '')
                        result_row[f'Similar_{col}'] = similar_item.get(col, '')
                    
                    all_results.append(result_row)
        
        # Sort results by Input_SKU and then by Similarity_Score (descending)
        if all_results:
            results_df = pd.DataFrame(all_results)
            results_df = results_df.sort_values(
                by=['Input_SKU', 'Similarity_Score'], 
                ascending=[True, False]
            )
            all_results = results_df.to_dict('records')
        
        # Performance summary
        total_time = time.time() - start_time
        logger.info(f"✅ Optimized batch processing complete in {total_time:.1f} seconds")
        logger.info(f"⚡ Performance: {len(image_groups)/total_time:.1f} images/sec")
        logger.info(f"📊 Total results: {len(all_results)}")
        
        # Log baseline filters that are active
        logger.info("🛡️ Baseline filters (applied to all searches):")
        if config_filtering.ENABLE_BASELINE_STATUS_FILTER:
            logger.info(f"  ✅ Status filter: Only including {config_filtering.BASELINE_STATUS_CODES}")
        if config_filtering.ENABLE_BASELINE_DATE_FILTER:
            logger.info(f"  🚫 Date filter: Excluding {config_filtering.BASELINE_EXCLUDE_YEARS} years and {config_filtering.BASELINE_EXCLUDE_DATES} dates")
        
        # Show cache stats
        cache_stats = self.optimized_search.get_cache_stats()
        logger.info(f"💾 Filter cache: {cache_stats['cache_size']} entries cached")
        
        return all_results
    
    def _get_embedding_for_filename(self, filename_root: str) -> Optional[np.ndarray]:
        """Get pre-computed embedding for filename"""
        # Try variations
        variations = [
            filename_root,
            filename_root.lower(),
            filename_root.upper(),
        ]
        
        # If it ends with a letter, also try without it
        if filename_root and filename_root[-1].isalpha():
            variations.extend([
                filename_root[:-1],
                filename_root[:-1].lower()
            ])
        
        for variant in variations:
            if variant in self.data_loader.filename_to_idx:
                idx = self.data_loader.filename_to_idx[variant]
                return self.data_loader.embeddings[idx]
        
        # Check filename mappings
        for variant in variations:
            if variant in self.data_loader.filename_mappings:
                mapped_root = self.data_loader.filename_mappings[variant]
                if mapped_root in self.data_loader.filename_to_idx:
                    idx = self.data_loader.filename_to_idx[mapped_root]
                    return self.data_loader.embeddings[idx]
        
        logger.warning(f"⚠️ No embedding found for {filename_root}")
        return None
    
    def _create_filter_key(self, filters: Dict) -> str:
        """Create a hashable key for filter combination"""
        # Convert lists to tuples for hashability
        normalized_filters = {}
        for k, v in filters.items():
            if isinstance(v, list):
                normalized_filters[k] = tuple(sorted(v))
            else:
                normalized_filters[k] = v
        
        import json
        return json.dumps(normalized_filters, sort_keys=True)
    
    def clear_filter_cache(self):
        """Clear the filter cache to free memory"""
        self.optimized_search.clear_cache()
        logger.info("🧹 Filter cache cleared") 