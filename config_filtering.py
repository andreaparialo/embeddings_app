"""
Configuration for batch processing filter behavior
"""

import pandas as pd

# ============================================================================
# BASELINE FILTERS - Applied to ALL searches before any other filtering
# ============================================================================
# These filters create the baseline search space that applies to all queries
# They significantly reduce the total embeddings available for search

# SKU STATUS BASELINE FILTER
# Only products with these status codes will be searchable
BASELINE_STATUS_CODES = [
    'IL',  # Only include "In Line" products
]
ENABLE_BASELINE_STATUS_FILTER = True

# RELEASE DATE BASELINE FILTER
# Exclude specific releases from the searchable index
# Years to exclude (will exclude any STARTSKU_DATE containing these years)
BASELINE_EXCLUDE_YEARS = [
    2026,  # Exclude all 2026 releases
]

# Specific dates to exclude (format: YYYY-MM-DD)
BASELINE_EXCLUDE_DATES = [
    '2025-08-01',  # Exclude specific August 2025 release
]
ENABLE_BASELINE_DATE_FILTER = True

# ============================================================================
# PRE-FILTERS - Applied before similarity search (after baseline filters)
# ============================================================================
# Columns that should be used for pre-filtering (before similarity search)
# These should be high-level category columns that significantly reduce the search space
# but still leave enough products for meaningful similarity comparison
PREFILTER_COLUMNS = [
    'BRAND_DES',           # Brand is critical for category definition
    'USERGENDER_DES',      # Gender significantly affects product style
    'PRODUCT_TYPE_COD',
    'GRANULAR_SHAPE_AWS',
    'MACRO_SHAPE_AWS',
    'FITTING_DES',
    'RIM_TYPE_DES',
    'CTM_FIRST_FRONT_MATERIAL_DES',
]

# Additional columns that can be added to pre-filtering if needed
# (uncomment to activate)
OPTIONAL_PREFILTER_COLUMNS = [
    # 'FITTING_DES',       # Could be used if fitting is critical
    # 'RIM_TYPE_DES',      # Could be used if rim type is essential
]

# ============================================================================
# POST-FILTERS - Applied after similarity search
# ============================================================================
# All other selected columns will be applied as post-filters
# (after the similarity search)

# ============================================================================
# FILTER PARAMETERS
# ============================================================================
# Minimum number of embeddings required after pre-filtering
# If pre-filtering results in fewer embeddings than this, a warning is logged
MIN_EMBEDDINGS_AFTER_PREFILTER = 50

# Maximum number of columns allowed for pre-filtering
# (to prevent over-restrictive filtering)
MAX_PREFILTER_COLUMNS = 10  # Increased to allow more pre-filter columns

# ============================================================================
# RANGE FILTERING - For numeric columns that should use range matching
# ============================================================================
# Columns that should use range-based filtering instead of exact matching
# Note: Values must be numeric or convertible to numeric (European comma decimals are supported)
RANGE_FILTER_COLUMNS = {
    'ACT_SKU_PRICE_VAL': 0.25,      # ±25% range for price
    'FRONT_LENGTH_VAL': 0.10,        # ±10% range for front length
    'TEMPLE_LENGTH_VAL': 0.10,       # ±10% range for temple length  
    'SIZE_COD': 0.05,                # ±5% range for size
    'FRONT_HEIGHT_VAL': 0.10,        # ±10% range for front height
}

# Default range percentage if column not specified above
DEFAULT_RANGE_PERCENTAGE = 0.25  # ±25%

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_prefilter_columns():
    """Get the list of columns to use for pre-filtering"""
    columns = PREFILTER_COLUMNS.copy()
    columns.extend(OPTIONAL_PREFILTER_COLUMNS)
    return columns[:MAX_PREFILTER_COLUMNS]

def is_prefilter_column(column_name):
    """Check if a column should be used for pre-filtering"""
    return column_name in get_prefilter_columns()

def should_exclude_by_baseline_date(startsku_date):
    """Check if a product should be excluded based on baseline date filter"""
    if not ENABLE_BASELINE_DATE_FILTER:
        return False
    
    if not startsku_date:
        return False
    
    date_str = str(startsku_date)
    
    # Check year exclusions
    for year in BASELINE_EXCLUDE_YEARS:
        if str(year) in date_str:
            return True
    
    # Check specific date exclusions
    for exclude_date in BASELINE_EXCLUDE_DATES:
        if exclude_date in date_str or date_str == exclude_date:
            return True
    
    return False

def should_exclude_by_baseline_status(status_code):
    """Check if a product should be excluded based on baseline status filter"""
    if not ENABLE_BASELINE_STATUS_FILTER:
        return False
    
    if not BASELINE_STATUS_CODES:
        return False  # If no allowed codes specified, include everything
    
    if not status_code:
        return True  # Exclude products with no status
    
    # Include only if status is in allowed list
    return status_code not in BASELINE_STATUS_CODES

def is_baseline_excluded(row):
    """Check if a product should be excluded by any baseline filter"""
    # Check status filter
    if should_exclude_by_baseline_status(row.get('MD_SKU_STATUS_COD')):
        return True
    
    # Check date filter
    if should_exclude_by_baseline_date(row.get('STARTSKU_DATE')):
        return True
    
    return False

# Legacy functions for backward compatibility
def should_exclude_by_date(startsku_date):
    """Legacy function - redirects to baseline filter"""
    return should_exclude_by_baseline_date(startsku_date)

def should_exclude_by_status(status_code):
    """Legacy function - redirects to baseline filter"""
    return should_exclude_by_baseline_status(status_code)

# Legacy variables for backward compatibility
ENABLE_RELEASE_FILTERING = ENABLE_BASELINE_DATE_FILTER
EXCLUDE_YEARS = BASELINE_EXCLUDE_YEARS
EXCLUDE_DATES = BASELINE_EXCLUDE_DATES
ENABLE_STATUS_FILTERING = ENABLE_BASELINE_STATUS_FILTER
ALLOWED_STATUS_CODES = BASELINE_STATUS_CODES 

def is_range_filter_column(column_name):
    """Check if a column should use range filtering"""
    return column_name in RANGE_FILTER_COLUMNS

def get_range_percentage(column_name):
    """Get the range percentage for a column"""
    return RANGE_FILTER_COLUMNS.get(column_name, DEFAULT_RANGE_PERCENTAGE)

def get_range_bounds(value, column_name):
    """Get the min/max bounds for range filtering"""
    if pd.isna(value) or value is None:
        return None, None
    
    try:
        # Handle European decimal format (comma instead of dot)
        if isinstance(value, str):
            value = value.replace(',', '.')
        value = float(value)
        range_pct = get_range_percentage(column_name)
        min_val = value * (1 - range_pct)
        max_val = value * (1 + range_pct)
        return min_val, max_val
    except (ValueError, TypeError):
        return None, None 