#!/usr/bin/env python3
"""
Simple test to debug why batch processor returns 0 results
"""

import logging
import pandas as pd
from batch_processor_optimized import OptimizedBatchProcessor
from data_loader import DataLoader
from gme_model import GMEModel
from dual_index_data_loader import dual_index_loader

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize
data_loader = DataLoader()
gme_model = GMEModel()

# Load data
csv_path = "database_results/DB_ACTIVE.csv"
df = data_loader.load_csv(csv_path)
data_loader.load_faiss_index("indexes", "1095", "1095")
dual_index_loader.initialize(csv_path)

# Create batch processor
batch_processor = OptimizedBatchProcessor(None, data_loader, gme_model)

# Test with one SKU
test_sku = "2056085NC52UC"
sku_data = df[df['SKU_COD'] == test_sku].iloc[0]

image_groups = {
    sku_data['filename_root']: {
        'skus': [test_sku],
        'source_item': sku_data.to_dict()
    }
}

# Test filtered mode
logger.info("\n" + "="*60)
logger.info("TESTING FILTERED MODE")
logger.info("="*60)

results = batch_processor.process_image_groups_with_prefilter(
    image_groups=image_groups,
    matching_cols=['BRAND_DES'],  # Just one filter for simplicity
    max_results_per_sku=10,
    dual_engine_enabled=True,
    main_weight=0.7,
    measurement_weight=0.3,
    search_mode='filtered'
)

logger.info(f"\nGOT {len(results)} RESULTS")
if results:
    df_results = pd.DataFrame(results)
    logger.info(f"Columns: {list(df_results.columns)}")
    if 'Index_Membership' in df_results.columns:
        logger.info(f"Index membership: {df_results['Index_Membership'].value_counts().to_dict()}")
    logger.info(f"\nFirst result:")
    for key, value in results[0].items():
        if key not in ['Source_' + k for k in sku_data.keys()] and key not in ['Similar_' + k for k in sku_data.keys()]:
            logger.info(f"  {key}: {value}")