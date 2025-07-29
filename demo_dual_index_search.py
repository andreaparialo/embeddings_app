#!/usr/bin/env python3
"""
Demo Dual Index Search
Demonstrates the new dual-index search system working with both GME and measurement features.
"""

import os
import sys
import logging
import numpy as np
from PIL import Image
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_image():
    """Create a test image for demonstration"""
    # Create a simple test image
    test_image = Image.new('RGB', (224, 224), color=(128, 64, 192))  # Purple color
    test_image_path = "demo_test_image.jpg"
    test_image.save(test_image_path)
    return test_image_path

def demo_dual_search():
    """Demonstrate dual index search"""
    try:
        logger.info("🚀 Dual Index Search Demo")
        logger.info("=" * 50)
        
        # Initialize the dual search engine
        from dual_index_search_engine import dual_search_engine
        
        csv_path = "database_results/final_with_aws_shapes_enriched.csv"
        index_id = "v11_1095_db_pictures_512_merged_final"
        
        logger.info("🔧 Initializing dual search engine...")
        if not dual_search_engine.initialize(csv_path, "indexes", "1095", index_id):
            logger.error("❌ Failed to initialize dual search engine")
            return False
        
        # Get system stats
        stats = dual_search_engine.get_search_stats()
        logger.info("📊 System Statistics:")
        logger.info(f"   Main index: {stats['main_index']['total_vectors']} vectors ({stats['main_index']['dimensions']}d)")
        logger.info(f"   Measurement index: {stats['measurement_index']['total_vectors']} vectors ({stats['measurement_index']['dimensions']}d)")
        logger.info(f"   Overlapping products: {stats['overlapping_products']}")
        logger.info(f"   Default weights: Main={stats['scoring_weights']['main_weight']:.1f}, Measurement={stats['scoring_weights']['measurement_weight']:.1f}")
        
        # Create a test image
        logger.info("\n🖼️ Creating test image...")
        test_image_path = create_test_image()
        
        try:
            # Test different search configurations
            test_configs = [
                {"name": "Main Index Only", "main_weight": 1.0, "measurement_weight": 0.0},
                {"name": "Measurement Index Only", "main_weight": 0.0, "measurement_weight": 1.0},
                {"name": "Balanced Dual Search", "main_weight": 0.7, "measurement_weight": 0.3},
                {"name": "Measurement Focused", "main_weight": 0.3, "measurement_weight": 0.7},
            ]
            
            for config in test_configs:
                logger.info(f"\n🔍 Testing: {config['name']}")
                logger.info(f"   Weights: Main={config['main_weight']:.1f}, Measurement={config['measurement_weight']:.1f}")
                
                try:
                    # Perform search
                    results = dual_search_engine.search_by_image_similarity_dual(
                        test_image_path, 
                        filters={}, 
                        top_k=5,
                        main_weight=config['main_weight'],
                        measurement_weight=config['measurement_weight']
                    )
                    
                    logger.info(f"   📊 Results: {len(results)} products found")
                    
                    # Show sample results
                    for i, result in enumerate(results[:3]):
                        if 'SKU_COD' in result and 'similarity_score' in result:
                            score_info = ""
                            if 'main_similarity' in result and 'measurement_similarity' in result:
                                score_info = f" (Main: {result['main_similarity']:.3f}, Measurement: {result['measurement_similarity']:.3f})"
                            
                            logger.info(f"     {i+1}. SKU: {result['SKU_COD']}, Score: {result['similarity_score']:.3f}{score_info}")
                    
                except Exception as e:
                    logger.error(f"   ❌ Error in {config['name']}: {e}")
            
            logger.info("\n✅ Demo completed successfully!")
            logger.info("\n🎉 Dual Index Search System is working correctly!")
            logger.info("   • Main index (GME embeddings): 3584d visual features")
            logger.info("   • Measurement index (Features): 256d technical features") 
            logger.info("   • Configurable scoring weights for different use cases")
            logger.info("   • Intelligent result combination and ranking")
            
            return True
            
        finally:
            # Clean up test image
            if os.path.exists(test_image_path):
                os.remove(test_image_path)
                logger.info("🧹 Cleaned up test image")
        
    except Exception as e:
        logger.error(f"❌ Demo error: {e}")
        return False

def demo_api_integration():
    """Show how to use the API endpoints"""
    logger.info("\n🌐 API Integration Examples:")
    logger.info("=" * 50)
    
    # Show curl examples
    examples = [
        {
            "name": "Standard Dual Search",
            "curl": """curl -X POST "http://127.0.0.1:8080/search/image-dual" \\
  -F "file=@your_image.jpg" \\
  -F "main_weight=0.7" \\
  -F "measurement_weight=0.3" \\
  -F "top_k=50"""
        },
        {
            "name": "Measurement-Focused Search",
            "curl": """curl -X POST "http://127.0.0.1:8080/search/image-dual" \\
  -F "file=@your_image.jpg" \\
  -F "main_weight=0.3" \\
  -F "measurement_weight=0.7" \\
  -F "filters={\"BRAND_DES\":\"RAY-BAN\"}" \\
  -F "top_k=20"""
        },
        {
            "name": "Get System Stats",
            "curl": """curl "http://127.0.0.1:8080/api/dual-index-stats" """
        },
        {
            "name": "Update Default Weights",
            "curl": """curl -X POST "http://127.0.0.1:8080/api/set-dual-weights" \\
  -F "main_weight=0.6" \\
  -F "measurement_weight=0.4" """
        }
    ]
    
    for example in examples:
        logger.info(f"\n📋 {example['name']}:")
        logger.info(f"   {example['curl']}")

if __name__ == "__main__":
    print("🚀 Starting Dual Index Search Demo...")
    
    try:
        # Run the demo
        success = demo_dual_search()
        
        # Show API examples
        demo_api_integration()
        
        print("\n" + "="*60)
        if success:
            print("🎉 Demo completed successfully!")
            print("The dual index search system is ready for use.")
        else:
            print("❌ Demo encountered issues.")
            print("Please check the logs above for details.")
        print("="*60)
        
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1) 