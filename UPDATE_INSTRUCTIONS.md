
# Database Migration Instructions

## Manual Steps Required:

1. **Update app.py**:
   - Replace `final_with_aws_shapes_enriched.csv` with `DB_FINAL_SIMILARIT_270615.csv`
   - Update filter logic to remove old columns and add new ones
   - Ensure COLOR column is treated as text (not numeric)

2. **Update templates/index.html**:
   - Remove filter UI elements for: FITTING_DES, LENS_BASE_DES, TEMPLE_LENGTH_VAL
   - Add new filter elements for: COLOR, CTM_FIRST_TEMPLE_MATERIAL_DES, SHAPE_SEMI_GROUPED, BRIDGE_LENGTH_VAL
   - Update filter labels and placeholders

3. **Test the following**:
   - Image search with new index
   - SKU search with new database
   - All filter combinations
   - Batch processing Excel upload
   - Performance with new reduced dataset

4. **Monitor for issues**:
   - Check for any missing column errors
   - Verify COLOR codes display correctly (3 digits)
   - Test BRIDGE_LENGTH_VAL range filtering
   - Ensure new shape/material filters work

## Automated Updates Applied:
- ✅ data_loader.py - Updated paths
- ✅ config_filtering_new.py - Created new filter configuration
- ✅ Backup created in: backups/backup_20250627_201408
