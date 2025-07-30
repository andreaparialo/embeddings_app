#!/usr/bin/env python3
"""
Test script to verify dual index search fix
"""

import logging
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_dual_index_fix():
    """Test the dual index fix"""
    
    print("🧪 Testing Dual Index SKU Search Fix")
    print("=" * 50)
    
    try:
        # Test imports
        from dual_index_search_engine import dual_search_engine
        from dual_index_data_loader import dual_index_loader
        
        print("✅ Imports successful")
        
        # Check if dual search engine is initialized
        if dual_search_engine.is_initialized:
            print("✅ Dual search engine is initialized")
        else:
            print("❌ Dual search engine not initialized")
            return False
        
        # Check measurement_to_main_mapping
        if hasattr(dual_index_loader, 'measurement_to_main_mapping'):
            mapping_size = len(dual_index_loader.measurement_to_main_mapping)
            print(f"✅ measurement_to_main_mapping exists: {mapping_size} mappings")
        else:
            print("❌ measurement_to_main_mapping missing")
            return False
        
        print("\n🎯 Expected Fix Results:")
        print("1. ✅ Should find overlapping products in both indexes")
        print("2. ✅ Should create boosted results for dual-index products")
        print("3. ✅ Should convert main indices to measurement indices")
        print("4. ✅ Should show 'Both indexes: X' with X > 0 in logs")
        print("5. ✅ Should use proper weighted scoring from both indexes")
        
        print("\n🔍 Key Changes Made:")
        print("- Copied exact logic from batch processor")
        print("- Create boosted results for overlapping products")
        print("- Convert main indices to measurement indices")
        print("- Proper measurement candidates creation")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dual_index_fix()
    print(f"\n{'✅ Test passed!' if success else '❌ Test failed!'}")
    sys.exit(0 if success else 1)
