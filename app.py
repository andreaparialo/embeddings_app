from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import pandas as pd
import os
import tempfile
import shutil
from typing import Dict, List, Optional
import logging
from old_app.search_engine import search_engine
from old_app.data_loader import data_loader
from old_app.dual_engine import dual_engine
from old_app.gme_model import gme_model
import json
import math
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="Hybrid Product Search Engine", version="1.0.0")

# Create directories if they don't exist
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
os.makedirs("uploads", exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
app.mount("/pictures", StaticFiles(directory="pictures", follow_symlink=True), name="pictures")

# Setup templates
templates = Jinja2Templates(directory="templates")

# Global variables
INITIALIZATION_STATUS = {"initialized": False, "message": "Not initialized"}

def sanitize_json_data(data):
    """Sanitize data for JSON serialization by handling NaN and infinity values"""
    if isinstance(data, dict):
        return {key: sanitize_json_data(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [sanitize_json_data(item) for item in data]
    elif isinstance(data, float):
        if math.isnan(data):
            return None  # Convert NaN to null
        elif math.isinf(data):
            return None  # Convert infinity to null
        else:
            return data
    else:
        return data

def get_image_path(filename_root):
    """Get the correct image path, handling both .JPG and .jpg extensions"""
    if not filename_root:
        return None
    
    # Try both .JPG and .jpg extensions
    image_path_jpg_upper = f"pictures/{filename_root}_O00.JPG"
    image_path_jpg_lower = f"pictures/{filename_root}_O00.jpg"
    
    if os.path.exists(image_path_jpg_upper):
        return image_path_jpg_upper
    elif os.path.exists(image_path_jpg_lower):
        return image_path_jpg_lower
    else:
        return None

def derive_filename_root_from_sku(sku_cod, product_type=None):
    """
    Derive filename_root from SKU_COD based on product type rules.
    
    If PRODUCT_TYPE != 1:
        SKU_COD = MODEL_COD (6) + COLOR_COD (3) + SIZE_COD (2) + ZLENSCODE (2)
        filename_root = MODEL_COD (6) + "0" + COLOR_COD (3) + ZLENSCODE (2)
    
    If PRODUCT_TYPE == 1:
        SKU_COD = MODEL_COD (6) + COLOR_COD (3) + SIZE_COD (2) + Something_else (2)
        filename_root = MODEL_COD (6) + "0" + COLOR_COD (3) + "00"
        Note: SKUs starting with "1" might be truncated and missing last 2 digits
    """
    if not sku_cod:
        return None
    
    try:
        # Handle truncated SKUs (especially those starting with "1")
        if len(sku_cod) < 9:  # Too short to have MODEL_COD + COLOR_COD
            return None
        
        model_cod = sku_cod[:6]  # First 6 digits
        color_cod = sku_cod[6:9] if len(sku_cod) >= 9 else sku_cod[6:]  # Next 3 digits (or whatever's available)
        
        # Pad color_cod if it's shorter than 3 digits
        if len(color_cod) < 3:
            color_cod = color_cod.ljust(3, '0')
        
        if product_type == 1 or sku_cod.startswith('1'):
            # For PRODUCT_TYPE == 1, ZLENSCODE is empty so we use "00"
            filename_root = f"{model_cod}0{color_cod}00"
        else:
            # For other product types, use the last 2 digits as ZLENSCODE if available
            if len(sku_cod) >= 13:
                zlenscode = sku_cod[11:13]
            elif len(sku_cod) >= 12:
                zlenscode = sku_cod[11:] + "0"  # Pad with one zero
            else:
                zlenscode = "00"  # Default to "00" if not enough digits
            
            filename_root = f"{model_cod}0{color_cod}{zlenscode}"
        
        return filename_root
    except Exception as e:
        logger.error(f"Error deriving filename_root from SKU {sku_cod}: {e}")
        return None

def find_sku_with_fallback(sku_cod, data_loader):
    """
    Find SKU in database with multiple fallback strategies:
    1. Direct SKU search
    2. Exact filename_root derivation
    3. Partial filename_root matching
    """
    try:
        # Strategy 1: Direct SKU search (existing behavior)
        direct_results = search_engine.search_by_sku(sku_cod, top_k=1)
        if direct_results:
            return direct_results[0]
        
        # Strategy 2: Try to find by derived filename_root
        logger.info(f"   🔄 Direct search failed for {sku_cod}, trying filename_root derivation...")
        
        # Try to determine product type from SKU pattern
        product_type = 1 if sku_cod.startswith('1') else 0
        derived_filename_root = derive_filename_root_from_sku(sku_cod, product_type)
        
        if derived_filename_root:
            logger.info(f"   📍 Derived filename_root: {derived_filename_root}")
            
            # Look for exact filename_root match
            exact_matches = data_loader.df[
                data_loader.df['filename_root'].astype(str).str.upper() == derived_filename_root.upper()
            ]
            
            if len(exact_matches) > 0:
                logger.info(f"   ✅ Found exact filename_root match!")
                return exact_matches.iloc[0].to_dict()
        
        # Strategy 3: Partial matching for truncated SKUs
        if len(sku_cod) < 13 and sku_cod.startswith('1'):
            logger.info(f"   🔄 Trying partial match for truncated SKU {sku_cod}...")
            
            # For truncated SKUs starting with "1", try partial filename_root matching
            if len(sku_cod) >= 9:
                model_cod = sku_cod[:6]
                color_cod = sku_cod[6:9]
                
                # Look for filename_root that starts with MODEL_COD + "0" + COLOR_COD
                partial_pattern = f"{model_cod}0{color_cod}"
                logger.info(f"   🔍 Looking for filename_root starting with: {partial_pattern}")
                
                partial_matches = data_loader.df[
                    data_loader.df['filename_root'].astype(str).str.upper().str.startswith(partial_pattern.upper())
                ]
                
                if len(partial_matches) > 0:
                    logger.info(f"   ✅ Found {len(partial_matches)} partial matches, using first one")
                    return partial_matches.iloc[0].to_dict()
        
        # Strategy 4: Try searching by MODEL_COD for very short SKUs
        if len(sku_cod) >= 6:
            model_cod = sku_cod[:6]
            logger.info(f"   🔄 Trying MODEL_COD search for: {model_cod}")
            
            model_matches = data_loader.df[
                data_loader.df['MODEL_COD'].astype(str).str.upper() == model_cod.upper()
            ]
            
            if len(model_matches) > 0:
                logger.info(f"   ✅ Found {len(model_matches)} MODEL_COD matches, using first one")
                return model_matches.iloc[0].to_dict()
        
        logger.warning(f"   ❌ No matches found for SKU {sku_cod} with any strategy")
        return None
        
    except Exception as e:
        logger.error(f"❌ Error in find_sku_with_fallback for {sku_cod}: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    """Initialize the search engine on startup"""
    global INITIALIZATION_STATUS
    try:
        # GPU configuration
        use_gpu = os.getenv("USE_FAISS_GPU", "true").lower() == "true"
        if not use_gpu:
            logger.info("🖥️  FAISS GPU disabled by environment variable")
            data_loader.use_gpu = False
        else:
            # Check if pip-installed faiss-gpu might cause issues
            try:
                import faiss
                if hasattr(faiss, '__version__'):
                    logger.info(f"📦 FAISS version: {faiss.__version__}")
                # Try to detect if it's conda or pip installed
                import site
                site_packages = site.getsitepackages()
                for path in site_packages:
                    if 'conda' in path.lower():
                        logger.info("✅ FAISS appears to be conda-installed")
                        break
                else:
                    logger.warning("⚠️  FAISS appears to be pip-installed. This may cause slow GPU transfers.")
                    logger.warning("   Consider installing via: conda install -c pytorch faiss-gpu")
            except Exception as e:
                logger.debug(f"Could not check FAISS installation: {e}")
        
        csv_path = "database_results/final_with_aws_shapes_20250625_155822.csv"
        if search_engine.initialize(csv_path):
            INITIALIZATION_STATUS = {"initialized": True, "message": "Search engine initialized successfully"}
            logger.info("Search engine initialized on startup")
        else:
            INITIALIZATION_STATUS = {"initialized": False, "message": "Failed to initialize search engine"}
    except Exception as e:
        INITIALIZATION_STATUS = {"initialized": False, "message": f"Startup error: {str(e)}"}
        logger.error(f"Startup error: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    filter_options = search_engine.get_filter_options() if INITIALIZATION_STATUS["initialized"] else {}
    checkpoints = data_loader.get_available_checkpoints() if INITIALIZATION_STATUS["initialized"] else []
    
    return templates.TemplateResponse("index.html", {
        "request": request,
        "initialization_status": INITIALIZATION_STATUS,
        "filter_options": filter_options,
        "checkpoints": checkpoints
    })

@app.get("/api/status")
async def get_status():
    """Get system status"""
    stats = search_engine.get_stats() if INITIALIZATION_STATUS["initialized"] else {}
    return {
        "initialization": INITIALIZATION_STATUS,
        "stats": stats
    }

@app.post("/test")
async def test_endpoint():
    """Simple test endpoint to check if requests are working"""
    logger.info("🧪 Test endpoint called!")
    return {"status": "success", "message": "Test endpoint working!"}

@app.post("/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    filters: str = Form("{}"),
    top_k: int = Form(50)
):
    """Search by uploaded image"""
    try:
        logger.info(f"🔍 Image search request received")
        logger.info(f"📁 File: {file.filename}")
        logger.info(f"📊 Size: {file.size} bytes" if file.size else "📊 Size: unknown")
        logger.info(f"🔧 Raw filters: {filters}")
        logger.info(f"🎯 Top K: {top_k}")
        
        # Check if search engine is initialized
        if not INITIALIZATION_STATUS["initialized"]:
            logger.error("❌ Search engine not initialized")
            return {"error": "Search engine not initialized"}
        
        # Parse filters
        try:
            filter_dict = json.loads(filters) if filters != "{}" else {}
            logger.info(f"🔧 Parsed filters: {list(filter_dict.keys())}")
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid filter JSON format: {e}")
            return {"error": "Invalid filter format"}
        
        # Check file
        if not file.filename:
            logger.error("❌ No filename provided")
            return {"error": "No file provided"}
        
        # Save uploaded file temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"temp_{timestamp}_{file.filename}"
        temp_path = os.path.join("uploads", temp_filename)
        
        logger.info(f"💾 Saving to: {temp_path}")
        try:
            with open(temp_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)
            logger.info(f"✅ File saved successfully, size: {len(content)} bytes")
        except Exception as e:
            logger.error(f"❌ Error saving file: {e}")
            return {"error": f"Error saving file: {e}"}
        
        # Perform search
        logger.info("🚀 Starting image similarity search...")
        try:
            results = search_engine.search_by_image_similarity(temp_path, filter_dict, top_k)
            logger.info(f"✅ Search completed, got {len(results)} results")
        except Exception as e:
            logger.error(f"❌ Error in search: {e}")
            return {"error": f"Search failed: {e}"}
        
        # Cleanup
        try:
            os.remove(temp_path)
            logger.info("🧹 Temporary file cleaned up")
        except Exception as e:
            logger.warning(f"⚠️ Could not clean up temp file: {e}")
        
        logger.info(f"✅ Image search completed - returned {len(results)} results")
        
        # Sanitize results for JSON serialization
        sanitized_results = sanitize_json_data(results)
        
        return {
            "results": sanitized_results,
            "total": len(results),
            "search_type": "image_similarity",
            "filters_applied": list(filter_dict.keys())
        }
        
    except Exception as e:
        logger.error(f"❌ Error in image search endpoint: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return {"error": str(e)}

@app.post("/search/sku")
async def search_by_sku_endpoint(sku: str = Form(...), top_k: int = Form(50)):
    """Search by SKU code"""
    try:
        logger.info(f"🔍 SKU search request received")
        logger.info(f"🏷️  SKU query: '{sku}'")
        
        # Perform search
        results = search_engine.search_by_sku(sku, top_k=top_k)
        
        logger.info(f"✅ SKU search completed - returned {len(results)} results")
        return {
            "results": sanitize_json_data(results),
            "total": len(results),
            "search_type": "sku",
            "query": sku
        }
        
    except Exception as e:
        logger.error(f"❌ Error in SKU search endpoint: {e}")
        return {"error": str(e)}

@app.post("/search/filters")
async def search_by_filters_endpoint(filters: str = Form(...), top_k: int = Form(50)):
    """Search by filters only"""
    try:
        logger.info(f"🔍 Filter search request received")
        
        # Parse filters
        try:
            filter_dict = json.loads(filters)
            logger.info(f"🔧 Parsed filters: {list(filter_dict.keys())}")
        except json.JSONDecodeError:
            logger.error("❌ Invalid filter JSON format")
            return {"error": "Invalid filter format"}
        
        # Perform search
        results = search_engine.search_by_filters(filter_dict, top_k=top_k)
        
        logger.info(f"✅ Filter search completed - returned {len(results)} results")
        return {
            "results": sanitize_json_data(results),
            "total": len(results),
            "search_type": "filters",
            "filters_applied": list(filter_dict.keys())
        }
        
    except Exception as e:
        logger.error(f"❌ Error in filter search endpoint: {e}")
        return {"error": str(e)}

@app.post("/search/batch")
async def search_by_sku_batch(file: UploadFile = File(...)):
    """Batch search by uploading Excel file with SKUs"""
    try:
        logger.info(f"🔍 Batch SKU search request received")
        logger.info(f"📁 File: {file.filename}")
        
        if not file.filename.endswith(('.xlsx', '.xls')):
            logger.error("❌ Invalid file format - only Excel files allowed")
            return {"error": "Only Excel files (.xlsx, .xls) are allowed"}
        
        # Save uploaded file temporarily  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"batch_{timestamp}_{file.filename}"
        temp_path = os.path.join("uploads", temp_filename)
        
        logger.info(f"💾 Saving Excel file to: {temp_path}")
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Read Excel file
        logger.info("📖 Reading Excel file...")
        import pandas as pd
        try:
            df = pd.read_excel(temp_path)
            logger.info(f"📊 Excel contains {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"🔍 Columns: {list(df.columns)}")
            
            # Extract SKUs (assume first column contains SKUs)
            sku_list = df.iloc[:, 0].astype(str).tolist()
            sku_list = [sku.strip() for sku in sku_list if sku.strip()]
            logger.info(f"📦 Extracted {len(sku_list)} SKUs")
            
        except Exception as e:
            logger.error(f"❌ Error reading Excel file: {e}")
            return {"error": f"Error reading Excel file: {e}"}
        
        # Perform batch search
        logger.info("🚀 Starting batch SKU search...")
        results = search_engine.search_by_sku_list(sku_list)
        
        # Cleanup
        try:
            os.remove(temp_path)
            logger.info("🧹 Temporary Excel file cleaned up")
        except:
            pass
        
        logger.info(f"✅ Batch search completed - returned {len(results)} results")
        return {
            "results": sanitize_json_data(results),
            "total": len(results),
            "search_type": "batch_sku",
            "skus_processed": len(sku_list)
        }
        
    except Exception as e:
        logger.error(f"❌ Error in batch search endpoint: {e}")
        return {"error": str(e)}

@app.post("/search/batch-enhanced")
async def enhanced_batch_search(
    file: UploadFile = File(...),
    matching_columns: str = Form(...),
    max_results_per_sku: int = Form(10),
    exclude_same_model: bool = Form(False),
    allowed_status_codes: str = Form('["IL", "NS", "NF", "OB", "AA"]'),
    group_unisex: bool = Form(False),
    dual_engine: bool = Form(False)
):
    """Enhanced batch search with column matching and Excel export"""
    try:
        logger.info(f"🔍 Enhanced batch search request received")
        logger.info(f"📁 File: {file.filename}")
        logger.info(f"⚙️ Max results per SKU: {max_results_per_sku}")
        logger.info(f"🚫 Exclude same model code: {exclude_same_model}")
        logger.info(f"👥 Group unisex: {group_unisex}")
        logger.info(f"🚀 Dual engine mode: {dual_engine}")
        
        if not file.filename.endswith(('.xlsx', '.xls')):
            logger.error("❌ Invalid file format - only Excel files allowed")
            return {"error": "Only Excel files (.xlsx, .xls) are allowed"}
        
        # Parse matching columns
        try:
            matching_cols = json.loads(matching_columns)
            logger.info(f"🔧 Matching columns: {matching_cols}")
        except json.JSONDecodeError:
            logger.error("❌ Invalid matching columns JSON")
            return {"error": "Invalid matching columns format"}
        
        # Parse allowed status codes
        try:
            allowed_statuses = json.loads(allowed_status_codes)
            logger.info(f"📋 Allowed status codes: {allowed_statuses}")
        except json.JSONDecodeError:
            logger.error("❌ Invalid allowed status codes JSON")
            return {"error": "Invalid allowed status codes format"}
        
        # Initialize dual engine if requested
        if dual_engine:
            logger.info("🚀 Initializing dual engine for this search...")
            csv_path = "database_results/final_with_aws_shapes_20250625_155822.csv"
            if not dual_engine.initialize_dual_engine(csv_path, "680", "1095"):
                logger.warning("⚠️ Dual engine initialization failed, falling back to single engine")
                dual_engine_enabled = False
            else:
                dual_engine_enabled = True
                logger.info("✅ Dual engine ready")
        else:
            dual_engine_enabled = False
        
        # Save uploaded file temporarily  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"enhanced_batch_{timestamp}_{file.filename}"
        temp_path = os.path.join("uploads", temp_filename)
        
        logger.info(f"💾 Saving Excel file to: {temp_path}")
        with open(temp_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Read Excel file
        logger.info("📖 Reading Excel file...")
        import pandas as pd
        try:
            df = pd.read_excel(temp_path)
            logger.info(f"📊 Excel contains {len(df)} rows, {len(df.columns)} columns")
            
            # Extract SKUs (assume first column contains SKUs)
            sku_list = df.iloc[:, 0].astype(str).tolist()
            sku_list = [sku.strip() for sku in sku_list if sku.strip()]
            logger.info(f"📦 Extracted {len(sku_list)} SKUs")
            
        except Exception as e:
            logger.error(f"❌ Error reading Excel file: {e}")
            return {"error": f"Error reading Excel file: {e}"}
        
        # Group SKUs by filename_root for optimized search
        logger.info("🚀 Starting enhanced batch search...")
        logger.info("🔧 Using BULK operations for SKU lookup and grouping...")
        
        # BULK OPTIMIZATION: Process all SKUs in bulk instead of one by one
        sku_groups = {}  # filename_root -> {source_item, skus: [list of input SKUs]}
        sku_to_source = {}  # input_sku -> source_item mapping
        
        # Step 1: Find all exact SKU matches in ONE query
        logger.info("📊 Step 1: Bulk exact SKU match...")
        exact_matches_df = data_loader.df[data_loader.df['SKU_COD'].isin(sku_list)]
        logger.info(f"✅ Found {len(exact_matches_df)} exact SKU matches in bulk")
        
        # Process exact matches
        found_skus = set()
        for _, row in exact_matches_df.iterrows():
            sku = row['SKU_COD']
            source_item = row.to_dict()
            sku_to_source[sku] = source_item
            found_skus.add(sku)
            
            # Get filename_root
            filename_root = source_item.get('filename_root', '')
            if filename_root:
                if filename_root not in sku_groups:
                    sku_groups[filename_root] = {
                        'source_item': source_item,
                        'skus': []
                    }
                sku_groups[filename_root]['skus'].append(sku)
        
        # Step 2: Process missing SKUs - derive filename_roots in bulk
        missing_skus = [sku for sku in sku_list if sku not in found_skus]
        if missing_skus:
            logger.info(f"📊 Step 2: Processing {len(missing_skus)} missing SKUs...")
            
            # Derive all filename_roots upfront
            derived_mappings = {}  # sku -> (derived_filename_root, product_type)
            for sku in missing_skus:
                product_type = 1 if sku.startswith('1') else 0
                derived_root = derive_filename_root_from_sku(sku, product_type)
                if derived_root:
                    derived_mappings[sku] = (derived_root, product_type)
            
            # Get unique derived roots for bulk search
            unique_derived_roots = list(set(root for root, _ in derived_mappings.values()))
            logger.info(f"📍 Derived {len(unique_derived_roots)} unique filename_roots")
            
            # Bulk search for all derived roots
            if unique_derived_roots:
                root_matches_df = data_loader.df[
                    data_loader.df['filename_root'].str.upper().isin([r.upper() for r in unique_derived_roots])
                ]
                logger.info(f"✅ Found {len(root_matches_df)} matches for derived roots")
                
                # Create a mapping of filename_root -> row data
                root_to_data = {}
                for _, row in root_matches_df.iterrows():
                    root = row['filename_root'].upper()
                    if root not in root_to_data:
                        root_to_data[root] = row.to_dict()
                
                # Process each missing SKU using the bulk results
                for sku, (derived_root, product_type) in derived_mappings.items():
                    upper_root = derived_root.upper()
                    if upper_root in root_to_data:
                        source_item = root_to_data[upper_root]
                        sku_to_source[sku] = source_item
                        found_skus.add(sku)
                        
                        # Group by filename_root
                        if derived_root not in sku_groups:
                            sku_groups[derived_root] = {
                                'source_item': source_item,
                                'skus': []
                            }
                        sku_groups[derived_root]['skus'].append(sku)
            
            # Step 3: Try partial matches for remaining truncated SKUs
            still_missing = [sku for sku in missing_skus if sku not in found_skus]
            if still_missing:
                truncated_skus = [sku for sku in still_missing if len(sku) < 13 and sku.startswith('1')]
                if truncated_skus:
                    logger.info(f"📊 Step 3: Processing {len(truncated_skus)} truncated SKUs...")
                    
                    # Build partial patterns for bulk search
                    partial_patterns = []
                    pattern_to_sku = {}
                    for sku in truncated_skus:
                        if len(sku) >= 9:
                            model_cod = sku[:6]
                            color_cod = sku[6:9]
                            pattern = f"{model_cod}0{color_cod}"
                            partial_patterns.append(pattern)
                            if pattern not in pattern_to_sku:
                                pattern_to_sku[pattern] = []
                            pattern_to_sku[pattern].append(sku)
                    
                    if partial_patterns:
                        # Use regex for bulk partial matching
                        import re
                        pattern_regex = '|'.join([f'^{p}' for p in partial_patterns])
                        partial_matches_df = data_loader.df[
                            data_loader.df['filename_root'].str.upper().str.contains(pattern_regex, regex=True, na=False)
                        ]
                        logger.info(f"✅ Found {len(partial_matches_df)} partial matches")
                        
                        # Process partial matches
                        for _, row in partial_matches_df.iterrows():
                            root_upper = row['filename_root'].upper()
                            for pattern, skus in pattern_to_sku.items():
                                if root_upper.startswith(pattern.upper()):
                                    source_item = row.to_dict()
                                    for sku in skus:
                                        if sku not in found_skus:
                                            sku_to_source[sku] = source_item
                                            found_skus.add(sku)
                                            
                                            # Group by filename_root
                                            filename_root = row['filename_root']
                                            if filename_root not in sku_groups:
                                                sku_groups[filename_root] = {
                                                    'source_item': source_item,
                                                    'skus': []
                                                }
                                            sku_groups[filename_root]['skus'].append(sku)
                                            break
        
        # Log final statistics
        not_found = [sku for sku in sku_list if sku not in found_skus]
        logger.info(f"📊 Bulk processing complete:")
        logger.info(f"   ✅ Found: {len(found_skus)}/{len(sku_list)} SKUs")
        logger.info(f"   🖼️ Grouped into: {len(sku_groups)} unique images")
        if not_found:
            logger.info(f"   ❌ Not found: {len(not_found)} SKUs")
            if len(not_found) <= 5:
                logger.info(f"      Missing: {', '.join(not_found)}")
        
        # Add sku_to_source mapping to each group
        for filename_root, group_data in sku_groups.items():
            group_data['sku_to_source'] = sku_to_source
        
        # Check if we should use parallel processing
        use_parallel = len(sku_groups) > 10  # Use parallel for more than 10 images
        
        if use_parallel:
            # Use parallel batch processor for many images
            logger.info("🚀 Using parallel batch processor for faster processing...")
            
                        # Initialize batch processor if needed
            from old_app.batch_processor import BatchImageProcessor
            from old_app.batch_processor_optimized import OptimizedBatchProcessor
            
            # Use optimized processor with pre-filtering for better performance
            use_optimized = True  # Can make this configurable later
            
            if use_optimized:
                logger.info("🚀 Using OPTIMIZED batch processor with pre-filtering")
                batch_proc = OptimizedBatchProcessor(search_engine, data_loader, gme_model)
            else:
                logger.info("📦 Using standard batch processor")
                batch_proc = BatchImageProcessor(search_engine, data_loader, gme_model)
            
            # Process in parallel
            if use_optimized and hasattr(batch_proc, 'process_image_groups_with_prefilter'):
                all_results = batch_proc.process_image_groups_with_prefilter(
                    sku_groups, matching_cols, max_results_per_sku,
                    exclude_same_model, allowed_statuses, group_unisex,
                    dual_engine_enabled, batch_size=16  # Process 16 images at once
                )
            else:
                all_results = batch_proc.process_image_groups_parallel(
                    sku_groups, matching_cols, max_results_per_sku,
                    exclude_same_model, allowed_statuses, group_unisex,
                    dual_engine_enabled, batch_size=16  # Process 16 images at once
                )
        else:
            # Use sequential processing for small batches
            logger.info("📋 Using sequential processing for small batch...")
            all_results = []
            processed_images = 0
            
            for filename_root, group_data in sku_groups.items():
                try:
                    processed_images += 1
                    source_item = group_data['source_item']
                    group_skus = group_data['skus']
                    
                    logger.info(f"🖼️ Processing image {processed_images}/{len(sku_groups)}: {filename_root}")
                    logger.info(f"   📦 SKUs in this group: {len(group_skus)}")
                    
                    # Get image path
                    image_path = get_image_path(filename_root)
                    if not image_path:
                        logger.warning(f"⚠️ Image not found for filename_root: {filename_root}")
                        continue
                    
                    # Extract matching column values from source item
                    matching_filters = {}
                    for col in matching_cols:
                        if col in source_item:
                            matching_filters[col] = source_item[col]
                            logger.info(f"   🏷️ {col}: {source_item[col]}")
                    
                    # Apply unisex grouping logic if enabled
                    if group_unisex and 'USERGENDER_DES' in matching_filters:
                        source_gender = matching_filters['USERGENDER_DES']
                        if source_gender in ['MAN', 'WOMAN']:
                            logger.info(f"   👥 Unisex grouping enabled for {source_gender}")
                    
                    # Determine search parameters
                    search_multiplier = max_results_per_sku * 2
                    if group_unisex and 'USERGENDER_DES' in matching_filters:
                        source_gender = matching_filters['USERGENDER_DES']
                        if source_gender in ['MAN', 'WOMAN']:
                            search_multiplier = max_results_per_sku * 3
                    
                    # Perform ONE image similarity search for all SKUs in this group
                    logger.info(f"   🔍 Performing image similarity search...")
                    
                    if dual_engine_enabled:
                        logger.info("   🚀 Using dual engine search...")
                        
                        if group_unisex and 'USERGENDER_DES' in matching_filters:
                            source_gender = matching_filters['USERGENDER_DES']
                            if source_gender in ['MAN', 'WOMAN']:
                                temp_filters = {k: v for k, v in matching_filters.items() if k != 'USERGENDER_DES'}
                                similar_results = dual_engine.search_by_image_similarity_dual(
                                    image_path, temp_filters, top_k=search_multiplier
                                )
                            else:
                                similar_results = dual_engine.search_by_image_similarity_dual(
                                    image_path, matching_filters, top_k=search_multiplier
                                )
                        else:
                            similar_results = dual_engine.search_by_image_similarity_dual(
                                image_path, matching_filters, top_k=search_multiplier
                            )
                    else:
                        logger.info("   🔍 Using single engine search...")
                        
                        if group_unisex and 'USERGENDER_DES' in matching_filters:
                            source_gender = matching_filters['USERGENDER_DES']
                            if source_gender in ['MAN', 'WOMAN']:
                                temp_filters = {k: v for k, v in matching_filters.items() if k != 'USERGENDER_DES'}
                                similar_results = search_engine.search_by_image_similarity(
                                    image_path, temp_filters, top_k=search_multiplier
                                )
                            else:
                                similar_results = search_engine.search_by_image_similarity(
                                    image_path, matching_filters, top_k=search_multiplier
                                )
                        else:
                            similar_results = search_engine.search_by_image_similarity(
                                image_path, matching_filters, top_k=search_multiplier
                            )
                    
                    logger.info(f"   🎯 Found {len(similar_results)} similar items")
                    
                    # Apply filters to the search results
                    # Apply gender filtering if unisex grouping is enabled
                    if group_unisex and 'USERGENDER_DES' in matching_filters:
                        source_gender = matching_filters['USERGENDER_DES']
                        if source_gender in ['MAN', 'WOMAN']:
                            allowed_genders = [source_gender, 'UNISEX ADULT']
                            similar_results = [
                                item for item in similar_results 
                                if item.get('USERGENDER_DES', '') in allowed_genders
                            ]
                            logger.info(f"   👥 After unisex filtering: {len(similar_results)} items")
                    
                    # Filter out same model code if requested
                    if exclude_same_model:
                        source_model_cod = source_item.get('MODEL_COD', '')
                        similar_results = [
                            item for item in similar_results 
                            if item.get('MODEL_COD', '') != source_model_cod
                        ]
                        logger.info(f"   🚫 After excluding same MODEL_COD: {len(similar_results)} items")
                    
                    # Filter by allowed status codes
                    if allowed_statuses:
                        similar_results = [
                            item for item in similar_results 
                            if item.get('MD_SKU_STATUS_COD', '') in allowed_statuses
                        ]
                        logger.info(f"   📋 After status code filtering: {len(similar_results)} items")
                    
                    # Limit to requested number of results
                    similar_results = similar_results[:max_results_per_sku]
                    
                    # Now apply these results to ALL SKUs in this group
                    for input_sku in group_skus:
                        # Get the specific source item for this SKU
                        sku_source_item = sku_to_source.get(input_sku, source_item)
                        
                        # Add results for this specific SKU
                        for similar_item in similar_results:
                            result_row = {
                                'Input_SKU': input_sku,
                                'Similar_SKU': similar_item.get('SKU_COD', ''),
                                'Similarity_Score': round(1 - similar_item.get('similarity_score', 0), 3)
                            }
                            
                            # Add matching column values (using SKU-specific source)
                            for col in matching_cols:
                                result_row[f'Source_{col}'] = sku_source_item.get(col, '')
                                result_row[f'Similar_{col}'] = similar_item.get(col, '')
                            
                            # Add dual engine information if applicable
                            if dual_engine_enabled:
                                result_row['Dual_Engine_Boost'] = similar_item.get('dual_engine_boost', False)
                                if similar_item.get('dual_engine_boost', False):
                                    result_row['Primary_Similarity'] = similar_item.get('primary_similarity', '')
                                    result_row['Secondary_Similarity'] = similar_item.get('secondary_similarity', '')
                                else:
                                    result_row['Source_Engine'] = similar_item.get('source_engine', 'primary')
                            
                            all_results.append(result_row)
                    
                    logger.info(f"   ✅ Added {len(similar_results) * len(group_skus)} results for this image group")
                    
                except Exception as e:
                    logger.error(f"❌ Error processing filename_root {filename_root}: {e}")
                    continue
        
        logger.info(f"🎉 Optimization complete: Processed {len(sku_list)} SKUs with only {len(sku_groups)} image searches!")
        logger.info(f"⚡ Using BULK operations saved significant time in the preparation phase!")
        
        # Cleanup temporary file
        try:
            os.remove(temp_path)
            logger.info("🧹 Temporary Excel file cleaned up")
        except:
            pass
        
        # Create Excel file with results
        if all_results:
            results_df = pd.DataFrame(all_results)
            
            # Create Excel file in memory
            from io import BytesIO
            output = BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                results_df.to_excel(writer, index=False, sheet_name='Batch_Search_Results')
            
            output.seek(0)
            
            logger.info(f"✅ Enhanced batch search completed - {len(all_results)} results")
            
            # Return Excel file as download
            from fastapi.responses import StreamingResponse
            return StreamingResponse(
                BytesIO(output.read()),
                media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                headers={"Content-Disposition": "attachment; filename=batch_search_results.xlsx"}
            )
        else:
            logger.warning("⚠️ No results found")
            return {"error": "No results found for any of the input SKUs"}
        
    except Exception as e:
        logger.error(f"❌ Error in enhanced batch search endpoint: {e}")
        import traceback
        logger.error(f"❌ Traceback: {traceback.format_exc()}")
        return {"error": str(e)}

@app.post("/api/search/filename")
async def search_by_filename(
    filename_root: str = Form(...),
    filters: str = Form("{}"),
    top_k: int = Form(50)
):
    """Search by filename similarity (using existing indexed image)"""
    if not INITIALIZATION_STATUS["initialized"]:
        return JSONResponse(
            status_code=503,
            content={"error": "Search engine not initialized"}
        )
    
    try:
        import json
        filter_dict = json.loads(filters) if filters else {}
        
        results = search_engine.search_by_filename_similarity(
            filename_root, filters=filter_dict, top_k=top_k
        )
        return {"results": sanitize_json_data(results), "count": len(results)}
        
    except Exception as e:
        logger.error(f"Error in filename search: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/filters")
async def get_filter_options():
    """Get available filter options"""
    if not INITIALIZATION_STATUS["initialized"]:
        return JSONResponse(
            status_code=503,
            content={"error": "Search engine not initialized"}
        )
    
    return search_engine.get_filter_options()

@app.get("/api/checkpoints")
async def get_checkpoints():
    """Get available LoRA checkpoints"""
    return data_loader.get_available_checkpoints()

@app.post("/api/change-checkpoint")
async def change_checkpoint(checkpoint: str = Form(...)):
    """Change LoRA checkpoint"""
    global INITIALIZATION_STATUS
    try:
        csv_path = "database_results/final_with_aws_shapes_20250625_155822.csv"
        if search_engine.initialize(csv_path, checkpoint=checkpoint):
            INITIALIZATION_STATUS = {"initialized": True, "message": f"Switched to checkpoint {checkpoint}"}
            return {"success": True, "message": f"Switched to checkpoint {checkpoint}"}
        else:
            return JSONResponse(
                status_code=500,
                content={"error": f"Failed to switch to checkpoint {checkpoint}"}
            )
    except Exception as e:
        logger.error(f"Error changing checkpoint: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools_manifest():
    """Handle Chrome DevTools manifest request to prevent 404 errors"""
    return {"version": "1.0.0", "name": "Hybrid Product Search Engine"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080) 