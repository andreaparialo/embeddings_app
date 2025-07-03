# Filter Architecture Changes

## Overview
The filtering system has been reorganized to properly distinguish between filename_root level filters (pre-filters) and SKU-specific filters (post-filters).

## Pre-Filters (Applied BEFORE FAISS Search)
These filters operate at the **filename_root level** - attributes shared by all SKUs with the same product image.
All pre-filter columns are **treated as TEXT** for string comparison.

### Pre-Filter Columns:
- `BRAND_DES` - Brand name
- `MD_SKU_STATUS_COD` - SKU status code  
- `PRODUCT_TYPE_COD` - Product type
- `COLOR` - Color code
- `COLOR_FAMILY_1_DES` - Color family description
- `ZLENSCODE` - Lens code
- `USERGENDER_DES` - Gender designation
- `CTM_FIRST_FRONT_MATERIAL_DES` - Front material
- `CTM_FIRST_TEMPLE_MATERIAL_DES` - Temple material
- `MACRO_SHAPE_AWS` - Macro shape classification
- `GRANULAR_SHAPE_AWS` - Granular shape classification
- `SHAPE_SEMI_GROUPED` - Semi-grouped shape
- `FlatTop_FlatTop_1` - Flat top feature
- `browline_browline_1` - Browline feature
- `bridge_Bridge_1` - Bridge feature
- `RIM_TYPE_DES` - Rim type description

## Post-Filters (Applied AFTER FAISS Search)
These filters operate at the **SKU level** - attributes that can vary between SKUs even if they share the same image.
All post-filter columns are **treated as FLOAT** for numeric range comparison.

### Post-Filter Columns:
- `ACT_SKU_PRICE_VAL` - Price (±25% range)
- `SIZE_COD` - Size code (±5% range)
- `FRONT_LENGTH_VAL` - Front length (±10% range)
- `BRIDGE_LENGTH_VAL` - Bridge length (±10% range)
- `FRONT_HEIGHT_VAL` - Front height (±10% range)

## Data Type Handling

### Automatic Conversion
When loading the CSV data:
1. **Pre-filter columns** are automatically converted to string type
2. **Post-filter columns** are automatically converted to float type
   - European decimal format (comma) is converted to standard format (dot)
   - Invalid numeric values become NaN

### Comparison Logic
- **Pre-filters**: Case-insensitive string comparison with whitespace normalization
- **Post-filters**: Numeric range comparison with configurable tolerance

## Performance Impact
This architecture ensures:
1. **Efficient search**: Pre-filters reduce the FAISS search space dramatically
2. **Accurate results**: SKU-specific attributes are only applied to already-similar products
3. **Type safety**: Automatic data type conversion prevents comparison errors

## Example Flow
```
Input SKU: 100075PJP5417
1. Find filename_root: 1000750PJP00
2. Apply pre-filters: BRAND_DES='BRAND X', USERGENDER_DES='MAN', etc.
   → Reduces search space from 11,531 to ~500 embeddings
3. FAISS similarity search on reduced space
4. Apply post-filters: ACT_SKU_PRICE_VAL within ±25% of source
   → Final filtered results
``` 