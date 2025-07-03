#!/usr/bin/env python3
"""
Test script to verify database migration and index configuration
"""

import requests
import pandas as pd
import json
import sys
import os

# Base URL for the application
BASE_URL = "http://127.0.0.1:8080"

def test_database_loading():
    """Test if the new database is loaded correctly"""
    print("\n🔍 Testing Database Loading...")
    
    # Check if new database file exists
    db_path = "database_results/DB_FINAL_SIMILARIT_270615.csv"
    if not os.path.exists(db_path):
        print(f"❌ Database file not found: {db_path}")
        return False
    
    # Load database and check columns
    df = pd.read_csv(db_path)
    print(f"✅ Database loaded: {len(df)} rows, {len(df.columns)} columns")
    
    # Check for new columns
    new_columns = ['COLOR', 'CTM_FIRST_TEMPLE_MATERIAL_DES', 'SHAPE_SEMI_GROUPED', 'BRIDGE_LENGTH_VAL']
    missing_columns = [col for col in new_columns if col not in df.columns]
    if missing_columns:
        print(f"❌ Missing new columns: {missing_columns}")
        return False
    else:
        print(f"✅ All new columns present: {new_columns}")
    
    # Check that old columns are removed
    old_columns = ['FITTING_DES', 'LENS_BASE_DES', 'TEMPLE_LENGTH_VAL']
    present_old_columns = [col for col in old_columns if col in df.columns]
    if present_old_columns:
        print(f"⚠️  Old columns still present: {present_old_columns}")
    else:
        print(f"✅ Old columns successfully removed")
    
    return True

def test_api_status():
    """Test API status endpoint"""
    print("\n🔍 Testing API Status...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ API Status: {data}")
            return data['initialization']['initialized']
        else:
            print(f"❌ API Status failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return False

def test_filter_options():
    """Test if filter options include new columns"""
    print("\n🔍 Testing Filter Options...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/filters")
        if response.status_code == 200:
            filters = response.json()
            
            # Check for new filter columns
            new_filters = ['COLOR', 'CTM_FIRST_TEMPLE_MATERIAL_DES', 'SHAPE_SEMI_GROUPED', 'BRIDGE_LENGTH_VAL']
            for filter_name in new_filters:
                if filter_name in filters:
                    options_count = len(filters[filter_name]) if isinstance(filters[filter_name], list) else 'N/A'
                    print(f"✅ {filter_name}: {options_count} options")
                else:
                    print(f"❌ {filter_name}: NOT FOUND in filters")
            
            # Check that old filters are removed
            old_filters = ['FITTING_DES', 'LENS_BASE_DES', 'TEMPLE_LENGTH_VAL']
            for filter_name in old_filters:
                if filter_name in filters:
                    print(f"⚠️  {filter_name}: Still present (should be removed)")
                else:
                    print(f"✅ {filter_name}: Successfully removed")
            
            return True
        else:
            print(f"❌ Filter options request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting filter options: {e}")
        return False

def test_indexes():
    """Test available indexes"""
    print("\n🔍 Testing Index Configuration...")
    
    try:
        response = requests.get(f"{BASE_URL}/api/indexes")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Available indexes: {len(data['indexes'])}")
            
            for index in data['indexes']:
                print(f"  - {index['id']}: {index['name']}")
                print(f"    Description: {index['description']}")
                print(f"    Image folder: {index['image_folder']}")
            
            if data['current_index']:
                print(f"\n✅ Current index: {data['current_index']['name']}")
            
            # Check if we have the two required indexes
            index_ids = [idx['id'] for idx in data['indexes']]
            required = ['v11_merged_latest', 'v11_1095_db_pictures_512']
            for req_id in required:
                if req_id in index_ids:
                    print(f"✅ Required index present: {req_id}")
                else:
                    print(f"❌ Required index missing: {req_id}")
            
            return True
        else:
            print(f"❌ Indexes request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting indexes: {e}")
        return False

def test_sku_search():
    """Test SKU search functionality"""
    print("\n🔍 Testing SKU Search...")
    
    # Get a sample SKU from the database
    df = pd.read_csv("database_results/DB_FINAL_SIMILARIT_270615.csv")
    sample_sku = df['SKU_COD'].iloc[0]
    
    try:
        response = requests.post(f"{BASE_URL}/search/sku", 
                               data={'sku': sample_sku, 'top_k': 5})
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SKU search successful for {sample_sku}")
            print(f"  Found {len(data.get('results', []))} results")
            
            # Check if results have new columns
            if data.get('results'):
                result = data['results'][0]
                new_fields = ['COLOR', 'SHAPE_SEMI_GROUPED']
                for field in new_fields:
                    if field in result:
                        print(f"  ✅ Result contains {field}: {result[field]}")
                    else:
                        print(f"  ❌ Result missing {field}")
            
            return True
        else:
            print(f"❌ SKU search failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error in SKU search: {e}")
        return False

def test_index_switching():
    """Test switching between indexes"""
    print("\n🔍 Testing Index Switching...")
    
    try:
        # Get current index
        response = requests.get(f"{BASE_URL}/api/indexes")
        if response.status_code != 200:
            print("❌ Could not get current index")
            return False
        
        current = response.json()['current_index_id']
        print(f"Current index: {current}")
        
        # Try to switch to the other index
        target_index = 'v11_1095_db_pictures_512' if current != 'v11_1095_db_pictures_512' else 'v11_merged_latest'
        
        print(f"Switching to: {target_index}")
        response = requests.post(f"{BASE_URL}/api/change-index", 
                               data={'index_id': target_index})
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Successfully switched to {target_index}")
            print(f"  Message: {result['message']}")
            
            # Switch back
            print(f"Switching back to: {current}")
            response = requests.post(f"{BASE_URL}/api/change-index", 
                                   data={'index_id': current})
            if response.status_code == 200:
                print(f"✅ Successfully switched back to {current}")
            
            return True
        else:
            print(f"❌ Index switch failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Error in index switching: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 DATABASE MIGRATION TEST SUITE")
    print("=" * 60)
    
    # Check if server is running
    try:
        response = requests.get(BASE_URL)
        if response.status_code != 200:
            print(f"❌ Server not responding at {BASE_URL}")
            print("Please start the server with: ./start_gpu.sh")
            sys.exit(1)
    except:
        print(f"❌ Cannot connect to server at {BASE_URL}")
        print("Please start the server with: ./start_gpu.sh")
        sys.exit(1)
    
    tests = [
        ("Database Loading", test_database_loading),
        ("API Status", test_api_status),
        ("Filter Options", test_filter_options),
        ("Index Configuration", test_indexes),
        ("SKU Search", test_sku_search),
        ("Index Switching", test_index_switching)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Migration successful!")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main() 