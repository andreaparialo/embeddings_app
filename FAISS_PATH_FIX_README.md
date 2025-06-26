# FAISS Index Path Fix Documentation

## Issue
When using the Excel Bulk Batch processor without selecting STARTSKU_DATE as a filter, all search results were incorrectly showing the same STARTSKU_DATE as the input SKUs. This should not happen - only selected columns should be used for filtering.

## Root Cause
The FAISS index (`v11_complete_merged_20250625_115302`) was created by merging two separate indexing processes:
1. **Old index**: Used paths like `../o00_images/`
2. **Delta index**: Used paths like `pictures_delta_prepared/`

However, all images are now located in the `pictures/` directory. This path mismatch was causing incorrect mappings between embeddings and products.

## Solution Applied
1. **Path Normalization**: Updated all image paths in the FAISS metadata to point to `pictures/` directory
2. **Updated Path Resolution**: Modified `data_loader.py`, `search_engine.py`, `openclip_search_engine.py`, and `batch_processor.py` to handle path normalization correctly
3. **Web App Updates**: Updated `app.py`, `app_minimal.py`, and `app_openclip.py` to resolve image paths correctly

## Files Modified
- `data_loader.py`: Added `get_image_path()` method with path normalization
- `search_engine.py`: Updated to use data_loader's path resolution
- `openclip_search_engine.py`: Added path normalization logic
- `batch_processor.py`: Updated to use data_loader's path resolution
- `app.py`, `app_minimal.py`, `app_openclip.py`: Updated image path resolution

## Backup Created
- Original metadata backed up to: `../indexes/v11_complete_merged_20250625_115302_metadata_backup.json`
- Path mapping saved to: `../indexes/path_mapping.json`

## Verification
After the fix, FAISS search results now correctly return products with diverse STARTSKU_DATE values instead of all having the same date as the input.

## Date
Fixed on: December 2024 