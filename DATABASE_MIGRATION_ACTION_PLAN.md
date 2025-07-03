# Database Migration Action Plan
## From `final_with_aws_shapes_enriched.csv` to `DB_FINAL_SIMILARIT_270615.csv`

---

## 📊 Database Changes Summary

### New Database Stats:
- **Rows**: 12,601 (reduced from 34,431) - ✅ Intentional reduction
- **Columns**: 25 (reduced from 40)
- **File**: `database_results/DB_FINAL_SIMILARIT_270615.csv`

### Removed Columns (19):
1. ❌ `ACT_SKU_PRICE_RANGE_DES`
2. ❌ `CONCEPT_01_DES` 
3. ❌ `CONCEPT_02_DES`
4. ❌ `FIRST_FRONT_MAT_DES`
5. ❌ `FITTING_DES` ⚠️ (Was used in pre-filtering - will be removed)
6. ❌ `FLG_SECOND_CHOICE`
7. ❌ `FlatTop_Confidence_1`
8. ❌ `GRANULAR_SHAPE_DES`
9. ❌ `LENS_BASE_DES` ⚠️ (Was priority filter - will be removed)
10. ❌ `MACRO_SHAPE_DES`
11. ❌ `MATERIALGROUP_DES`
12. ❌ `PORTFOLIO_PRICE_RANGE_DES`
13. ❌ `SKU_STATUS_HIST_DAILY_COD`
14. ❌ `SKU_URL_MEDIUM`
15. ❌ `SPECIAL_SKU_FLG`
16. ❌ `TEMPLE_LENGTH_VAL` ⚠️ (Was range filter - will be removed)
17. ❌ `ZCOLFAM1`
18. ❌ `bridge_Confidence_1`
19. ❌ `browline_Confidence_1`

### Added Columns (4):
1. ✅ `BRIDGE_LENGTH_VAL` (int64) - Will use ±10% range filtering
2. ✅ `COLOR` (object) - 3-digit codes, all verified
3. ✅ `CTM_FIRST_TEMPLE_MATERIAL_DES` (object) - Will be added as filter
4. ✅ `SHAPE_SEMI_GROUPED` (object) - Will be added as filter

### Critical Columns Status:
- ✅ All critical columns present (SKU_COD, filename_root, MODEL_COD, etc.)

---

## 🔍 FAISS Index Analysis

### Current Situation:
- **Existing index**: 29,136 embeddings
- **Products in new DB**: 11,979 unique filename_roots
- **Coverage**: Only 82.9% (9,927 products) have embeddings
- **Missing embeddings**: 2,052 products (17.1%)

### ⚠️ Critical Decision Required:
**2,052 products in the new database don't have embeddings in the current FAISS index!**

Options:
1. **Option A**: Create reduced index with only 9,927 products (faster but incomplete)
2. **Option B**: Generate embeddings for 2,052 missing products first (complete but slower)

**Recommendation**: Option B - Generate missing embeddings to ensure all products are searchable

---

## 🔧 Updated Action Items

### Phase 0: FAISS Index Preparation
- [ ] Generate embeddings for 2,052 missing products
- [ ] Create new reduced FAISS index with all 11,979 products
- [ ] Update metadata files to match new index

### Phase 1: Configuration Updates

#### File: `config_filtering.py`
- [x] Remove `FITTING_DES` from `PREFILTER_COLUMNS`
- [x] Remove `TEMPLE_LENGTH_VAL` from `RANGE_FILTER_COLUMNS`
- [x] Add `BRIDGE_LENGTH_VAL` to `RANGE_FILTER_COLUMNS` with 10% tolerance
- [x] Update any references to removed columns

#### File: `data_loader.py`
- [x] Update priority filter columns list (remove LENS_BASE_DES)
- [ ] Update CSV path to use new database
- [ ] Test loading with reduced row count (12,601 vs 34,431)

### Phase 2: Frontend Updates

#### File: `templates/index.html`
- [ ] Remove filter dropdowns for all 19 removed columns
- [ ] Add new filter dropdowns:
  - COLOR (3-digit codes)
  - CTM_FIRST_TEMPLE_MATERIAL_DES
  - SHAPE_SEMI_GROUPED
  - BRIDGE_LENGTH_VAL (as range filter)

### Phase 3: Backend Updates

#### File: `app.py`
- [ ] Update CSV path in startup_event() to use new database
- [ ] Update any hardcoded column references
- [ ] Verify batch processing works with new columns

#### File: `search_engine.py`
- [ ] Update filter options generation
- [ ] Ensure no references to removed columns

#### File: `batch_processor_optimized.py`
- [ ] Update matching column logic for removed columns
- [ ] Add handling for new columns

### Phase 4: Data Validation

- [x] COLOR codes verified - all are 3 digits
- [ ] Verify filename_root mappings with new FAISS index
- [ ] Test SKU derivation logic with new data
- [ ] Validate range filtering for BRIDGE_LENGTH_VAL

### Phase 5: Testing Plan

1. [ ] Test single image search
2. [ ] Test SKU search
3. [ ] Test filter-only search
4. [ ] Test batch Excel processing
5. [ ] Test all new filter combinations
6. [ ] Test BRIDGE_LENGTH_VAL range filter (±10%)

---

## 📋 Implementation Order

1. **Phase 0**: FAISS Index (CRITICAL - Do First!)
   - Generate missing embeddings
   - Create reduced index
   - Test index alignment

2. **Phase 1**: Backend Configuration
   - Update config_filtering.py ✅
   - Update data_loader.py
   - Test database loading

3. **Phase 2**: Core Functionality
   - Update app.py startup
   - Update search_engine.py
   - Test basic searches

4. **Phase 3**: Batch Processing
   - Update batch processors
   - Test Excel uploads

5. **Phase 4**: Frontend
   - Update filter dropdowns
   - Update UI labels
   - Test all interactions

6. **Phase 5**: Validation
   - Run full test suite
   - Performance testing
   - Error handling

---

## 🚨 Risk Assessment

1. **High Risk**: 17.1% of products missing embeddings - must be addressed
2. **Medium Risk**: Broken filters due to removed columns
3. **Low Risk**: UI inconsistencies with new columns

---

## 📝 User Decisions Summary

1. ✅ Database reduction is intentional
2. ✅ Create new FAISS index with reduced products
3. ✅ Remove old columns without replacement
4. ✅ Add all new columns as filters
5. ✅ BRIDGE_LENGTH_VAL uses ±10% tolerance
6. ✅ COLOR codes are text (3 digits)
7. ✅ Permanent migration (no backward compatibility)

---

## 🚀 Next Steps

1. **Immediate**: Decide on FAISS index strategy (recommend Option B)
2. **Then**: Generate missing embeddings if needed
3. **Finally**: Start Phase 1 implementation 