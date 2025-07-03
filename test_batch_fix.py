#!/usr/bin/env python3
"""
Test script to verify the batch-enhanced endpoint fix
"""

import requests
import pandas as pd
import os
import tempfile
import json

# Base URL for the application
BASE_URL = "http://127.0.0.1:8080"

def test_batch_enhanced():
    """Test the batch-enhanced endpoint"""
    print("\n🔍 Testing Batch Enhanced Search...")
    
    # Create a test Excel file with SKUs that exist in the new database
    test_skus = ['100075PJP5417', '100138OO45215', '100140AJH5236', '102166ISK5418']
    df = pd.DataFrame({'SKU': test_skus})
    
    # Save to temporary Excel file
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        df.to_excel(tmp.name, index=False, engine='openpyxl')
        temp_path = tmp.name
    
    try:
        # Prepare the request
        with open(temp_path, 'rb') as f:
            files = {'file': ('test_skus.xlsx', f, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
            data = {
                'matching_columns': json.dumps([
                    # Pre-filters (filename_root level)
                    'BRAND_DES', 'USERGENDER_DES', 'PRODUCT_TYPE_COD', 
                    'COLOR_FAMILY_1_DES', 'MACRO_SHAPE_AWS',
                    'CTM_FIRST_FRONT_MATERIAL_DES',
                    # Post-filters (SKU-specific)
                    'ACT_SKU_PRICE_VAL', 'FRONT_HEIGHT_VAL'
                ]),
                'max_results_per_sku': 5,
                'exclude_same_model': False,
                'allowed_status_codes': json.dumps(['IL', 'NS', 'NF', 'OB', 'AA']),
                'group_unisex': False,
                'dual_engine': False
            }
            
            # Make the request
            print("📤 Sending batch search request...")
            response = requests.post(f'{BASE_URL}/search/batch-enhanced', files=files, data=data)
            
            print(f"📊 Response Status Code: {response.status_code}")
            print(f"📊 Response Headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                # Check if it's an Excel file response
                content_type = response.headers.get('content-type', '')
                if 'spreadsheet' in content_type:
                    # Save the Excel file to check results
                    output_path = 'test_batch_results.xlsx'
                    with open(output_path, 'wb') as f:
                        f.write(response.content)
                    print(f"✅ Success! Results saved to {output_path}")
                    
                    # Read and show summary
                    results_df = pd.read_excel(output_path)
                    print(f"📊 Total results: {len(results_df)} rows")
                    print(f"📊 Unique input SKUs: {results_df['Input_SKU'].nunique()}")
                    print(f"📊 Sample results:")
                    print(results_df.head())
                else:
                    # It's a JSON response
                    result = response.json()
                    if 'error' in result:
                        print(f"❌ Error: {result['error']}")
                    else:
                        print(f"📊 Response: {result}")
            else:
                try:
                    error_data = response.json()
                    print(f"❌ Error: {error_data}")
                except:
                    print(f"❌ Error: {response.text}")
                    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)
            print("🧹 Cleaned up temporary file")

def test_api_status():
    """Test if the API is running"""
    print("\n🔍 Testing API Status...")
    
    try:
        response = requests.get(f'{BASE_URL}/api/status')
        if response.status_code == 200:
            status = response.json()
            print(f"✅ API is running: {status}")
            return True
        else:
            print(f"❌ API returned status code: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Could not connect to API: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Batch Enhanced Test")
    
    # First check if API is running
    if test_api_status():
        # Run the batch enhanced test
        test_batch_enhanced()
    else:
        print("⚠️ Please start the server first with: ./start_gpu.sh") 