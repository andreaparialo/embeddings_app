#!/usr/bin/env python3
"""
Diagnostic Script: Analyze Index Overlap
Creates a detailed CSV report of which products exist in which indexes
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Set
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_main_metadata():
    """Load main index metadata"""
    main_paths = [
        "indexes/v11_1095_db_pictures_512_merged_final_20250703_125538_metadata.json",
        "indexes/v11_complete_merged_20250625_115302_metadata.json",
        "indexes/metadata.json"
    ]
    
    for path in main_paths:
        if Path(path).exists():
            logger.info(f"Loading main metadata from: {path}")
            with open(path, 'r') as f:
                return json.load(f), path
    
    raise FileNotFoundError("No main metadata file found")

def load_measurement_metadata():
    """Load measurement index metadata"""
    measurement_paths = [
        "indexes/index_measurements/corrected_metadata.json",
        "indexes/index_measurements/metadata.json"
    ]
    
    for path in measurement_paths:
        if Path(path).exists():
            logger.info(f"Loading measurement metadata from: {path}")
            with open(path, 'r') as f:
                return json.load(f), path
    
    raise FileNotFoundError("No measurement metadata file found")

def load_database_csv():
    """Load the main database CSV"""
    csv_paths = [
        "database_results/final_with_aws_shapes_enriched.csv",
        "database_results/DB_ACTIVE.csv"
    ]
    
    for path in csv_paths:
        if Path(path).exists():
            logger.info(f"Loading database CSV from: {path}")
            return pd.read_csv(path), path
    
    raise FileNotFoundError("No database CSV file found")

def normalize_filename(filename: str) -> str:
    """Normalize filename to match the format used in the system"""
    # Remove common suffixes
    suffixes_to_remove = ['_P02_white_bg', '_P02_white_bg.jpg', '_white_bg', '_P02']
    base_name = filename
    
    for suffix in suffixes_to_remove:
        if base_name.endswith(suffix):
            base_name = base_name[:-len(suffix)]
            break
    
    # Add the standard path format
    return f"db_pictures_512/{base_name}.jpg"

def extract_filename_root(path: str) -> str:
    """Extract filename root from path"""
    if "/" in path:
        filename = path.split("/")[-1]
    else:
        filename = path
    
    # Remove extensions
    for ext in ['.jpg', '.jpeg', '.png', '.gif']:
        if filename.lower().endswith(ext):
            filename = filename[:-len(ext)]
            break
    
    return filename

def analyze_index_overlap():
    """Analyze overlap between indexes and create detailed report"""
    logger.info("🔍 Starting comprehensive index overlap analysis...")
    
    # Load all data sources
    try:
        main_metadata, main_path = load_main_metadata()
        measurement_metadata, measurement_path = load_measurement_metadata()
        df, csv_path = load_database_csv()
    except FileNotFoundError as e:
        logger.error(f"Failed to load required files: {e}")
        return None
    
    logger.info(f"📊 Data sources loaded:")
    logger.info(f"   Main metadata: {main_path}")
    logger.info(f"   Measurement metadata: {measurement_path}")
    logger.info(f"   Database CSV: {csv_path}")
    
    # Extract filename roots from each source
    logger.info("📝 Extracting filename roots from each source...")
    
    # Main index filename roots
    main_image_paths = main_metadata.get("image_paths", [])
    main_filename_roots = set()
    for path in main_image_paths:
        if path:
            root = extract_filename_root(path)
            main_filename_roots.add(root)
    
    logger.info(f"   Main index: {len(main_filename_roots)} unique filename roots")
    
    # Measurement index filename roots
    measurement_filename_roots = set()
    
    if "image_paths" in measurement_metadata:
        # Corrected format
        measurement_image_paths = measurement_metadata.get("image_paths", [])
        for path in measurement_image_paths:
            if path:
                root = extract_filename_root(path)
                measurement_filename_roots.add(root)
    elif "product_mapping" in measurement_metadata:
        # Original format - normalize on the fly
        product_mapping = measurement_metadata.get("product_mapping", {})
        for original_name in product_mapping.values():
            normalized_path = normalize_filename(original_name)
            root = extract_filename_root(normalized_path)
            measurement_filename_roots.add(root)
    
    logger.info(f"   Measurement index: {len(measurement_filename_roots)} unique filename roots")
    
    # Database filename roots
    db_filename_roots = set()
    if 'filename_root' in df.columns:
        db_filename_roots = set(df['filename_root'].dropna().astype(str).unique())
    
    logger.info(f"   Database CSV: {len(db_filename_roots)} unique filename roots")
    
    # Calculate overlaps
    logger.info("🔄 Calculating overlaps...")
    
    # All unique filename roots across all sources
    all_filename_roots = main_filename_roots | measurement_filename_roots | db_filename_roots
    
    # Create detailed analysis
    analysis_results = []
    
    for filename_root in all_filename_roots:
        in_main = filename_root in main_filename_roots
        in_measurement = filename_root in measurement_filename_roots
        in_database = filename_root in db_filename_roots
        
        # Get SKU count from database
        sku_count = 0
        sample_skus = []
        if in_database:
            matching_rows = df[df['filename_root'] == filename_root]
            sku_count = len(matching_rows)
            sample_skus = matching_rows['SKU_COD'].head(3).tolist() if 'SKU_COD' in df.columns else []
        
        # Determine category
        if in_main and in_measurement:
            category = "Both Indexes"
        elif in_main and not in_measurement:
            category = "Main Only"
        elif not in_main and in_measurement:
            category = "Measurement Only"
        else:
            category = "Neither Index"
        
        analysis_results.append({
            'filename_root': filename_root,
            'in_main_index': in_main,
            'in_measurement_index': in_measurement,
            'in_database': in_database,
            'category': category,
            'sku_count': sku_count,
            'sample_skus': ', '.join(map(str, sample_skus)) if sample_skus else ''
        })
    
    # Convert to DataFrame
    analysis_df = pd.DataFrame(analysis_results)
    
    # Generate summary statistics
    logger.info("📊 Generating summary statistics...")
    
    summary_stats = {
        'total_unique_filename_roots': len(all_filename_roots),
        'main_index_count': len(main_filename_roots),
        'measurement_index_count': len(measurement_filename_roots),
        'database_count': len(db_filename_roots),
        'both_indexes': len(analysis_df[analysis_df['category'] == 'Both Indexes']),
        'main_only': len(analysis_df[analysis_df['category'] == 'Main Only']),
        'measurement_only': len(analysis_df[analysis_df['category'] == 'Measurement Only']),
        'neither_index': len(analysis_df[analysis_df['category'] == 'Neither Index']),
        'overlap_percentage': len(analysis_df[analysis_df['category'] == 'Both Indexes']) / len(all_filename_roots) * 100
    }
    
    # Print summary
    logger.info("📋 SUMMARY REPORT:")
    logger.info(f"   Total unique filename roots: {summary_stats['total_unique_filename_roots']}")
    logger.info(f"   Main index: {summary_stats['main_index_count']}")
    logger.info(f"   Measurement index: {summary_stats['measurement_index_count']}")
    logger.info(f"   Database: {summary_stats['database_count']}")
    logger.info(f"   Both indexes: {summary_stats['both_indexes']} ({summary_stats['overlap_percentage']:.1f}%)")
    logger.info(f"   Main only: {summary_stats['main_only']}")
    logger.info(f"   Measurement only: {summary_stats['measurement_only']}")
    logger.info(f"   Neither index: {summary_stats['neither_index']}")
    
    # Save detailed CSV report
    output_file = "index_overlap_analysis.csv"
    analysis_df.to_csv(output_file, index=False)
    logger.info(f"📄 Detailed report saved to: {output_file}")
    
    # Save summary statistics
    summary_file = "index_overlap_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_stats, f, indent=2)
    logger.info(f"📊 Summary statistics saved to: {summary_file}")
    
    # Show sample problematic cases
    logger.info("\n🔍 SAMPLE PROBLEMATIC CASES:")
    
    main_only_samples = analysis_df[analysis_df['category'] == 'Main Only'].head(5)
    if not main_only_samples.empty:
        logger.info("📋 Main index only (missing from measurement):")
        for _, row in main_only_samples.iterrows():
            logger.info(f"   {row['filename_root']} (SKUs: {row['sku_count']})")
    
    measurement_only_samples = analysis_df[analysis_df['category'] == 'Measurement Only'].head(5)
    if not measurement_only_samples.empty:
        logger.info("📋 Measurement index only (missing from main):")
        for _, row in measurement_only_samples.iterrows():
            logger.info(f"   {row['filename_root']} (SKUs: {row['sku_count']})")
    
    return analysis_df, summary_stats

def check_specific_batch_queries():
    """Check the specific queries from the batch search logs"""
    logger.info("\n🎯 Checking specific queries from recent batch search...")
    
    # These are filename roots from the logs
    batch_query_roots = [
        "1034040KJ100", "1043470RHL00", "1065600PJP00", "1070260OIT00", 
        "1070780XYO00", "1078960PJP00", "1081000N9P00", "1084100OIT00",
        "2067570M9LUZ", "2073500G3IHA", "2075760L7Q70", "2080520AOZMT"
    ]
    
    try:
        main_metadata, _ = load_main_metadata()
        measurement_metadata, _ = load_measurement_metadata()
    except FileNotFoundError as e:
        logger.error(f"Failed to load metadata: {e}")
        return
    
    # Extract filename roots
    main_paths = main_metadata.get("image_paths", [])
    main_roots = {extract_filename_root(path) for path in main_paths if path}
    
    measurement_paths = []
    if "image_paths" in measurement_metadata:
        measurement_paths = measurement_metadata.get("image_paths", [])
    elif "product_mapping" in measurement_metadata:
        for original_name in measurement_metadata.get("product_mapping", {}).values():
            normalized_path = normalize_filename(original_name)
            measurement_paths.append(normalized_path)
    
    measurement_roots = {extract_filename_root(path) for path in measurement_paths if path}
    
    logger.info("🔍 Checking batch query roots:")
    for root in batch_query_roots[:10]:  # Check first 10
        in_main = root in main_roots
        in_measurement = root in measurement_roots
        status = "✅ Both" if (in_main and in_measurement) else f"⚠️  Main: {in_main}, Measurement: {in_measurement}"
        logger.info(f"   {root}: {status}")

if __name__ == "__main__":
    try:
        # Run comprehensive analysis
        analysis_df, summary_stats = analyze_index_overlap()
        
        # Check specific batch queries
        check_specific_batch_queries()
        
        print("\n" + "="*60)
        print("🎉 Analysis complete!")
        print(f"📄 Detailed CSV report: index_overlap_analysis.csv")
        print(f"📊 Summary statistics: index_overlap_summary.json")
        print("="*60)
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        sys.exit(1) 