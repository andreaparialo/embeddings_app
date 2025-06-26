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
INITIALIZATION_STATUS = {"initialized": False, "message": "Minimal mode - AI features disabled"}
df = None

def load_csv_data():
    """Load CSV data"""
    global df, INITIALIZATION_STATUS
    try:
        csv_path = "database_results/final_with_aws_shapes_20250625_155822.csv"
        if not os.path.exists(csv_path):
            INITIALIZATION_STATUS = {"initialized": False, "message": "CSV file not found"}
            return False
        
        # Try different encodings
        for encoding in ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']:
            try:
                df = pd.read_csv(csv_path, encoding=encoding)
                logger.info(f"Successfully loaded CSV with {encoding} encoding")
                logger.info(f"CSV shape: {df.shape}")
                INITIALIZATION_STATUS = {"initialized": True, "message": "CSV data loaded successfully (minimal mode)"}
                return True
            except UnicodeDecodeError:
                continue
        
        INITIALIZATION_STATUS = {"initialized": False, "message": "Could not decode CSV file"}
        return False
        
    except Exception as e:
        INITIALIZATION_STATUS = {"initialized": False, "message": f"Error loading CSV: {str(e)}"}
        logger.error(f"Error loading CSV: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    load_csv_data()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    filter_options = {}
    checkpoints = ["680", "1020", "1095"]  # Mock data
    
    # Get filter options if CSV is loaded
    if df is not None:
        filter_columns = [col for col in df.columns if col != 'filename_root']
        for column in filter_columns[:10]:  # Limit to first 10 columns for demo
            try:
                unique_values = df[column].dropna().unique()
                if len(unique_values) < 100:  # Only for reasonable number of options
                    filter_options[column] = sorted([str(val) for val in unique_values])
            except:
                pass
    
    return templates.TemplateResponse("index_minimal.html", {
        "request": request,
        "initialization_status": INITIALIZATION_STATUS,
        "filter_options": filter_options,
        "checkpoints": checkpoints
    })

@app.get("/api/status")
async def get_status():
    """Get system status"""
    stats = {
        "csv_rows": len(df) if df is not None else 0,
        "mode": "minimal",
        "ai_features": "disabled"
    }
    return {
        "initialization": INITIALIZATION_STATUS,
        "stats": stats
    }

@app.post("/api/search/sku")
async def search_by_sku(sku_cod: str = Form(...)):
    """Search by SKU code"""
    if not INITIALIZATION_STATUS["initialized"] or df is None:
        return JSONResponse(
            status_code=503,
            content={"error": "CSV data not loaded"}
        )
    
    try:
        result = df[df['SKU_COD'] == sku_cod]
        if len(result) > 0:
            result_dict = result.iloc[0].to_dict()
            # Add image path
            if 'filename_root' in result_dict:
                image_path = get_image_path(result_dict['filename_root'])
                if image_path:
                    result_dict['image_path'] = image_path
            return {"result": result_dict}
        else:
            return {"result": None, "message": "SKU not found"}
    except Exception as e:
        logger.error(f"Error in SKU search: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/search/filters")
async def search_by_filters(
    filters: str = Form("{}"),
    top_k: int = Form(50)
):
    """Search using only filters"""
    if not INITIALIZATION_STATUS["initialized"] or df is None:
        return JSONResponse(
            status_code=503,
            content={"error": "CSV data not loaded"}
        )
    
    try:
        import json
        filter_dict = json.loads(filters) if filters else {}
        
        filtered_df = df.copy()
        
        # Apply filters
        for column, value in filter_dict.items():
            if column in filtered_df.columns and value is not None and value != '':
                if isinstance(value, str):
                    filtered_df = filtered_df[
                        filtered_df[column].astype(str).str.contains(value, case=False, na=False)
                    ]
                else:
                    filtered_df = filtered_df[filtered_df[column] == value]
        
        # Get results
        results = []
        for _, row in filtered_df.head(top_k).iterrows():
            result = row.to_dict()
            # Add image path
            if 'filename_root' in result:
                image_path = get_image_path(result['filename_root'])
                if image_path:
                    result['image_path'] = image_path
            results.append(result)
        
        return {"results": results, "count": len(results)}
        
    except Exception as e:
        logger.error(f"Error in filter search: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/search/sku-batch")
async def search_by_sku_batch(file: UploadFile = File(...)):
    """Search by Excel file with SKU codes"""
    if not INITIALIZATION_STATUS["initialized"] or df is None:
        return JSONResponse(
            status_code=503,
            content={"error": "CSV data not loaded"}
        )
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
            shutil.copyfileobj(file.file, tmp_file)
            tmp_path = tmp_file.name
        
        try:
            # Read Excel file
            excel_df = pd.read_excel(tmp_path)
            
            if len(excel_df.columns) == 0:
                return JSONResponse(
                    status_code=400,
                    content={"error": "Excel file appears to be empty"}
                )
            
            sku_column = excel_df.columns[0]
            sku_list = excel_df[sku_column].astype(str).tolist()
            
            # Search for each SKU
            results = []
            for sku in sku_list:
                result = df[df['SKU_COD'] == sku]
                if len(result) > 0:
                    result_dict = result.iloc[0].to_dict()
                    if 'filename_root' in result_dict:
                        image_path = get_image_path(result_dict['filename_root'])
                        if image_path:
                            result_dict['image_path'] = image_path
                    results.append(result_dict)
            
            return {
                "results": results,
                "total_skus": len(sku_list),
                "found_skus": len(results)
            }
            
        finally:
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"Error in batch SKU search: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# Disabled endpoints for minimal version
@app.post("/api/search/image")
async def search_by_image_disabled():
    return JSONResponse(
        status_code=503,
        content={"error": "Image search disabled in minimal mode. Install AI dependencies first."}
    )

@app.post("/api/change-checkpoint")
async def change_checkpoint_disabled():
    return JSONResponse(
        status_code=503,
        content={"error": "Checkpoint switching disabled in minimal mode."}
    )

@app.get("/api/filters")
async def get_filter_options():
    """Get available filter options"""
    if not INITIALIZATION_STATUS["initialized"] or df is None:
        return {}
    
    filter_options = {}
    filter_columns = [col for col in df.columns if col != 'filename_root']
    
    for column in filter_columns[:15]:  # Limit for performance
        try:
            unique_values = df[column].dropna().unique()
            if len(unique_values) < 200:
                filter_options[column] = sorted([str(val) for val in unique_values])
        except:
            pass
    
    return filter_options

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000) 