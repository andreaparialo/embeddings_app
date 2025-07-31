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
                                           batch_size: int = 8,
                                           main_weight: float = 0.7,
                                           measurement_weight: float = 0.3,
                                           search_mode: str = "global") -> List[Dict]:
        """
        Process multiple image groups with pre-filtering for optimal performance
        """
        start_time = time.time()
        logger.info(f"🚀 Starting optimized batch processing with PRE-FILTERING")
        logger.info(f"📊 Processing {len(image_groups)} unique images")
        logger.info(f"🔧 Matching columns: {matching_cols}")
        if dual_engine_enabled:
            logger.info(f"🔍 Search mode: {search_mode}")
            logger.info(f"⚖️ Weights: GME={main_weight:.1%}, Technical={measurement_weight:.1%}")
        
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
        
        if dual_engine_enabled:
            logger.info("🎭 Using DUAL INDEX search with pre-computed embeddings!")
            logger.info(f"⚖️ Weights: GME={main_weight:.1%}, Technical={measurement_weight:.1%}")
            # Use dual index search - both GME and measurement indexes
            search_results = self._batch_dual_index_search(
                queries, max_results_per_sku * 3, query_metadata,
                main_weight=main_weight, measurement_weight=measurement_weight,
                search_mode=search_mode
            )
        else:
            # Use standard GME-only search with pre-filtering
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
            for item in result_indices:
                if len(item) == 3:  # New format with scoring details
                    distance, embedding_idx, scoring_info = item
                else:  # Legacy format
                    distance, embedding_idx = item
                    scoring_info = None
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
                            
                            # Add scoring details if available
                            if scoring_info:
                                item['gme_score'] = scoring_info.get('main_similarity', 0.0)
                                item['technical_score'] = scoring_info.get('measurement_similarity', 0.0)
                                item['final_score'] = scoring_info.get('combined_score', 0.0)
                                item['index_coverage'] = scoring_info.get('source', 'unknown')
                            
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
                    
                    # Add scoring details if dual engine was used
                    if dual_engine_enabled and 'gme_score' in similar_item:
                        result_row.update({
                            'GME_Score': round(similar_item.get('gme_score', 0), 3),
                            'Technical_Score': round(similar_item.get('technical_score', 0), 3),
                            'Final_Score': round(similar_item.get('final_score', 0), 3),
                            'Score_Formula': f"{main_weight:.0%}×GME + {measurement_weight:.0%}×Technical",
                            'Index_Coverage': similar_item.get('index_coverage', 'unknown')
                        })
                    
                    # Add ALL columns from source item (prefixed with Source_)
                    for col, value in group_data['source_item'].items():
                        if col not in ['Input_SKU', 'Similar_SKU', 'Similarity_Score']:  # Avoid duplicates
                            result_row[f'Source_{col}'] = value
                    
                    # Add ALL columns from similar item (prefixed with Similar_)
                    for col, value in similar_item.items():
                        if col not in ['similarity_score', 'SKU_COD']:  # These are already included
                            result_row[f'Similar_{col}'] = value
                    
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
        
        if dual_engine_enabled:
            logger.info(f"🎭 Dual Engine Mode Summary:")
            logger.info(f"  Search mode: {search_mode}")
            logger.info(f"  Weights: GME={main_weight:.1%}, Technical={measurement_weight:.1%}")
        
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
    
    def _batch_dual_index_search(self, queries: List[Tuple], top_k: int, query_metadata: Dict, 
                                main_weight: float = 0.7, measurement_weight: float = 0.3,
                                search_mode: str = "global") -> Dict[str, List[Tuple[float, int, Dict]]]:
        """
        Perform TRUE dual index search with weighted scoring
        
        Args:
            queries: List of (query_id, embedding, filters) tuples
            top_k: Number of results per query
            query_metadata: Dict mapping query_id to metadata including filename_root
            main_weight: Weight for GME similarity (default 0.7)
            measurement_weight: Weight for technical similarity (default 0.3)
            
        Returns:
            Dict mapping query_id to list of (distance, embedding_idx, scoring_info) tuples
        """
        from dual_index_data_loader import dual_index_loader
        import numpy as np
        
        logger.info("🔍 Performing TRUE dual index search with weighted scoring...")
        logger.info(f"📊 Processing {len(queries)} queries")
        logger.info(f"⚖️ Formula: {main_weight:.1%} × GME + {measurement_weight:.1%} × Technical")
        logger.info(f"🔍 Search mode: {search_mode}")
        
        # Set weights in dual loader
        dual_index_loader.set_scoring_weights(main_weight, measurement_weight)
        
        results = {}
        
        for query_id, gme_embedding, filters in queries:
            try:
                # For batch search, we use the GME embedding from the query (filename_root's embedding)
                # and search for the corresponding measurement embedding in the dual index system
                
                # Debug: Log filters being applied
                logger.debug(f"🔍 Debug [{query_id}]: Search mode={search_mode}, Filters={filters}")
                
                if search_mode == "global":
                    # Mode 1: Global Search - Search all products with FAISS, then apply filters
                    # Search WITHOUT filters first to get global best matches
                    main_distances, main_indices = dual_index_loader.search_main_index(
                        gme_embedding, top_k * 2, filters=None  # No filters for global search
                    )
                else:
                    # Mode 2: Filtered Search - Apply filters first, then search within subset
                    # Search WITH filters to get best matches within filtered subset
                    main_distances, main_indices = dual_index_loader.search_main_index(
                        gme_embedding, top_k * 2, filters
                    )
                
                # 2. Get measurement embedding for this query
                filename_root = query_metadata[query_id]['filename_root']
                measurement_embedding = dual_index_loader.get_measurement_embedding_by_filename(filename_root)
                
                if measurement_embedding is not None:
                    # 3. Search measurement index with appropriate filtering based on mode
                    if search_mode == "global":
                        # Mode 1: Global Search - No filters during search
                        meas_distances, meas_indices = dual_index_loader.search_measurement_index(
                            measurement_embedding, top_k * 2
                        )
                    else:
                        # Mode 2: Filtered Search - Apply same filters as main index
                        meas_distances, meas_indices = dual_index_loader.search_measurement_index_with_filters(
                            measurement_embedding, top_k * 2, filters
                        )
                    logger.debug(f"Query {filename_root}: Found measurement embedding, searching both indexes")
                else:
                    # No measurement embedding available
                    meas_distances = np.array([])
                    meas_indices = np.array([])
                    logger.debug(f"Query {filename_root}: No measurement embedding, using GME only")
                
                # 4. Use the EXISTING combine_search_results for proper weighted scoring!
                combined_results = dual_index_loader.combine_search_results(
                    main_distances, main_indices,
                    meas_distances, meas_indices,
                    top_k
                )
                
                # 5. Convert combined results back to the format expected by batch processor
                query_results = []
                
                # For global mode, we need to filter the results after scoring
                if search_mode == "global" and filters:
                    # Apply filters post-search for global mode
                    filtered_results = []
                    for result in combined_results:
                        # Check if result matches filters
                        result_filename = result.get('filename_root', '')
                        if result_filename:
                            # Find matching rows in dataframe
                            matching_rows = self.data_loader.df[
                                self.data_loader.df['filename_root'] == result_filename
                            ]
                            
                            # Check if any row matches all filters
                            matches_filters = False
                            for _, row in matching_rows.iterrows():
                                all_match = True
                                for col, filter_val in filters.items():
                                    if col in row:
                                        if isinstance(filter_val, list):
                                            if row[col] not in filter_val:
                                                all_match = False
                                                break
                                        elif row[col] != filter_val:
                                            all_match = False
                                            break
                                if all_match:
                                    matches_filters = True
                                    break
                            
                            if matches_filters:
                                filtered_results.append(result)
                    
                    original_count = len(combined_results)
                    combined_results = filtered_results
                    logger.info(f"🎯 Global mode post-filtering for {query_id}: {original_count} → {len(filtered_results)} results after filtering")
                
                for result in combined_results:
                    # Get the filename_root from the result
                    result_filename = result.get('filename_root', '')
                    if result_filename and result_filename in self.data_loader.filename_to_idx:
                        embedding_idx = self.data_loader.filename_to_idx[result_filename]
                        # Use the combined similarity score (already weighted!)
                        similarity_score = result.get('similarity_score', 0.0)
                        # Convert similarity to distance for consistency
                        distance = 1.0 - similarity_score
                        
                        # Store additional scoring info in the result
                        query_results.append((
                            distance, 
                            embedding_idx,
                            {
                                'main_similarity': result.get('main_similarity', 0.0),
                                'measurement_similarity': result.get('measurement_similarity', 0.0),
                                'combined_score': similarity_score,
                                'source': result.get('score_source', 'unknown')
                            }
                        ))
                
                results[query_id] = query_results
                
            except Exception as e:
                logger.warning(f"Error in dual index search for {query_id}: {e}")
                # Fallback to empty results
                results[query_id] = []
        
        logger.info(f"✅ True dual index search completed for {len(queries)} queries")
        return results 