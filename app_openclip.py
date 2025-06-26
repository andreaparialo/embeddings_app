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
from openclip_search_engine import openclip_search_engine
from openclip_data_loader import openclip_data_loader
from openclip_model import openclip_model
from openclip_batch_processor import openclip_batch_processor
import json
import math
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(title="OpenCLIP Product Search Engine", version="1.0.0")

# Create directories if they don't exist
os.makedirs("uploads", exist_ok=True)

# Mount static files - use parent directory's static folder
static_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")
if os.path.exists(static_path):
    app.mount("/static", StaticFiles(directory=static_path), name="static")
else:
    # Create local static directory if parent doesn't exist
    os.makedirs("static", exist_ok=True)
    app.mount("/static", StaticFiles(directory="static"), name="static")
# Use the parent directory's pictures folder
pictures_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pictures")
if os.path.exists(pictures_path):
    app.mount("/pictures", StaticFiles(directory=pictures_path, follow_symlink=True), name="pictures")
else:
    logger.warning(f"Pictures directory not found at {pictures_path}")

# Setup templates - use parent directory's templates
templates_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "templates")
templates = Jinja2Templates(directory=templates_path)

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
    
    # Get the parent directory's pictures folder
    pictures_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pictures")
    
    # Try both .JPG and .jpg extensions
    image_path_jpg_upper = os.path.join(pictures_dir, f"{filename_root}_O00.JPG")
    image_path_jpg_lower = os.path.join(pictures_dir, f"{filename_root}_O00.jpg")
    
    # Return the path relative to the mount point for web serving
    if os.path.exists(image_path_jpg_upper):
        return f"/pictures/{filename_root}_O00.JPG"
    elif os.path.exists(image_path_jpg_lower):
        return f"/pictures/{filename_root}_O00.jpg"
    else:
        return None

@app.on_event("startup")
async def startup_event():
    """Initialize the OpenCLIP search engine on startup"""
    global INITIALIZATION_STATUS
    try:
        csv_path = "/home/ubuntu/SPEEDINGTHEPROCESS/database_results/final_with_aws_shapes_20250625_155822.csv"
        index_dir = "/home/ubuntu/SPEEDINGTHEPROCESS/indexes/openclip"
        checkpoint = "epoch_008_model.pth"  # Default checkpoint
        
        if openclip_search_engine.initialize(csv_path, index_dir, checkpoint):
            INITIALIZATION_STATUS = {"initialized": True, "message": "OpenCLIP search engine initialized successfully"}
            logger.info("OpenCLIP search engine initialized on startup")
        else:
            INITIALIZATION_STATUS = {"initialized": False, "message": "Failed to initialize OpenCLIP search engine"}
    except Exception as e:
        INITIALIZATION_STATUS = {"initialized": False, "message": f"Startup error: {str(e)}"}
        logger.error(f"Startup error: {e}")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    filter_options = openclip_search_engine.get_filter_options() if INITIALIZATION_STATUS["initialized"] else {}
    checkpoints = ["epoch_008_model.pth", "epoch_011_model.pth"]  # Available OpenCLIP checkpoints
    
    # Create a modified template context that indicates this is the OpenCLIP version
    return templates.TemplateResponse("index.html", {
        "request": request,
        "initialization_status": INITIALIZATION_STATUS,
        "filter_options": filter_options,
        "checkpoints": checkpoints,
        "engine_type": "OpenCLIP"  # Add this to distinguish from GME version
    })

@app.get("/api/status")
async def get_status():
    """Get system status"""
    stats = openclip_search_engine.get_stats() if INITIALIZATION_STATUS["initialized"] else {}
    return {
        "initialization": INITIALIZATION_STATUS,
        "stats": stats,
        "engine": "OpenCLIP"
    }

@app.post("/test")
async def test_endpoint():
    """Simple test endpoint to check if requests are working"""
    logger.info("🧪 Test endpoint called!")
    return {"status": "success", "message": "OpenCLIP test endpoint working!"}

@app.post("/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    filters: str = Form("{}"),
    top_k: int = Form(50),
    search_pool_size: int = Form(1000)
):
    """Search by uploaded image using OpenCLIP"""
    try:
        logger.info(f"🔍 Image search request received (OpenCLIP)")
        logger.info(f"📁 File: {file.filename}")
        logger.info(f"📊 Size: {file.size} bytes" if file.size else "📊 Size: unknown")
        logger.info(f"🔧 Raw filters: {filters}")
        logger.info(f"🎯 Top K: {top_k}")
        
        # Check if search engine is initialized
        if not INITIALIZATION_STATUS["initialized"]:
            logger.error("❌ OpenCLIP search engine not initialized")
            return {"error": "OpenCLIP search engine not initialized"}
        
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
        logger.info("🚀 Starting OpenCLIP image similarity search...")
        try:
            results = openclip_search_engine.search_by_image_similarity(temp_path, filter_dict, top_k, search_pool_size)
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
        
        logger.info(f"✅ OpenCLIP image search completed - returned {len(results)} results")
        
        # Sanitize results for JSON serialization
        sanitized_results = sanitize_json_data(results)
        
        return {
            "results": sanitized_results,
            "total": len(results),
            "search_type": "image_similarity",
            "filters_applied": list(filter_dict.keys()),
            "engine": "OpenCLIP"
        }
        
    except Exception as e:
        logger.error(f"❌ Error in OpenCLIP image search endpoint: {e}")
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
        results = openclip_search_engine.search_by_sku(sku, top_k=top_k)
        
        logger.info(f"✅ SKU search completed - returned {len(results)} results")
        return {
            "results": sanitize_json_data(results),
            "total": len(results),
            "search_type": "sku",
            "query": sku,
            "engine": "OpenCLIP"
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
        results = openclip_search_engine.search_by_filters(filter_dict, top_k=top_k)
        
        logger.info(f"✅ Filter search completed - returned {len(results)} results")
        return {
            "results": sanitize_json_data(results),
            "total": len(results),
            "search_type": "filters",
            "filters_applied": list(filter_dict.keys()),
            "engine": "OpenCLIP"
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
        results = openclip_search_engine.search_by_sku_list(sku_list)
        
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
            "skus_processed": len(sku_list),
            "engine": "OpenCLIP"
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
    group_unisex: bool = Form(False)
):
    """Enhanced batch search with column matching and Excel export using OpenCLIP"""
    try:
        logger.info(f"🔍 Enhanced batch search request received (OpenCLIP)")
        logger.info(f"📁 File: {file.filename}")
        logger.info(f"⚙️ Max results per SKU: {max_results_per_sku}")
        logger.info(f"🚫 Exclude same model code: {exclude_same_model}")
        logger.info(f"👥 Group unisex: {group_unisex}")
        
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
        
        # Use the batch processor for parallel GPU processing
        logger.info("🚀 Using OpenCLIP batch processor with parallel GPU processing...")
        
        # Check if pre-filtering method is available
        if hasattr(openclip_batch_processor, 'process_batch_search_with_prefilter'):
            logger.info("✨ Using OPTIMIZED pre-filtering approach")
            all_results = openclip_batch_processor.process_batch_search_with_prefilter(
                sku_list=sku_list,
                matching_columns=matching_cols,
                max_results_per_sku=max_results_per_sku,
                exclude_same_model=exclude_same_model,
                allowed_status_codes=allowed_statuses,
                group_unisex=group_unisex,
                search_pool_size=1000
            )
        else:
            all_results = openclip_batch_processor.process_batch_search(
                sku_list=sku_list,
                matching_columns=matching_cols,
                max_results_per_sku=max_results_per_sku,
                exclude_same_model=exclude_same_model,
                allowed_status_codes=allowed_statuses,
                group_unisex=group_unisex,
                search_pool_size=1000
            )
        
        processed_count = len(set([r['input_sku'] for r in all_results]))
        
        # Cleanup
        try:
            os.remove(temp_path)
            logger.info("🧹 Temporary Excel file cleaned up")
        except:
            pass
        
        logger.info(f"✅ Enhanced batch search completed")
        logger.info(f"   📊 Processed: {processed_count}/{len(sku_list)} SKUs")
        logger.info(f"   📋 Total results: {len(all_results)}")
        
        return {
            "results": sanitize_json_data(all_results),
            "total": len(all_results),
            "search_type": "enhanced_batch",
            "skus_processed": len(sku_list),
            "skus_found": processed_count,
            "engine": "OpenCLIP"
        }
        
    except Exception as e:
        logger.error(f"❌ Error in enhanced batch search endpoint: {e}")
        return {"error": str(e)}

@app.post("/api/search/filename")
async def search_by_filename(
    filename_root: str = Form(...),
    filters: str = Form("{}"),
    top_k: int = Form(50),
    search_pool_size: int = Form(1000)
):
    """Search by filename similarity"""
    try:
        logger.info(f"🔍 Filename similarity search request received")
        logger.info(f"📁 Filename root: {filename_root}")
        
        # Parse filters
        try:
            filter_dict = json.loads(filters) if filters != "{}" else {}
        except json.JSONDecodeError:
            filter_dict = {}
        
        # Perform search
        results = openclip_search_engine.search_by_filename_similarity(filename_root, filter_dict, top_k, search_pool_size)
        
        logger.info(f"✅ Filename search completed - returned {len(results)} results")
        return {
            "results": sanitize_json_data(results),
            "total": len(results),
            "search_type": "filename_similarity",
            "query": filename_root,
            "engine": "OpenCLIP"
        }
        
    except Exception as e:
        logger.error(f"❌ Error in filename search endpoint: {e}")
        return {"error": str(e)}

@app.get("/api/filters")
async def get_filter_options():
    """Get available filter options"""
    if not INITIALIZATION_STATUS["initialized"]:
        return {}
    
    return openclip_search_engine.get_filter_options()

@app.post("/api/change-checkpoint")
async def change_checkpoint(checkpoint: str = Form(...)):
    """Change OpenCLIP checkpoint"""
    try:
        logger.info(f"🔄 Changing OpenCLIP checkpoint to: {checkpoint}")
        
        # Reload model with new checkpoint
        if openclip_model.load_model(checkpoint):
            logger.info(f"✅ Successfully loaded checkpoint: {checkpoint}")
            return {"status": "success", "message": f"Checkpoint changed to {checkpoint}"}
        else:
            logger.error(f"❌ Failed to load checkpoint: {checkpoint}")
            return {"status": "error", "message": f"Failed to load checkpoint {checkpoint}"}
            
    except Exception as e:
        logger.error(f"❌ Error changing checkpoint: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools_manifest():
    """Chrome DevTools manifest"""
    return {
        "capabilities": ["file_server"],
        "fileServerAllowlist": ["/"],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)  # Run on different port than main app 