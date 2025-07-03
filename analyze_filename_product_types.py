#!/usr/bin/env python3
"""
Analyze filename_root distribution across product types
"""

import pandas as pd
import json
import os

def analyze_filename_product_types():
    print("🔍 Analyzing Filename Root Distribution by Product Type")
    print("=" * 80)
    
    # Load new database
    new_df = pd.read_csv('database_results/DB_FINAL_SIMILARIT_270615.csv')
    
    # Load reduction plan to get missing roots
    with open('faiss_reduction_plan.json', 'r') as f:
        reduction_plan = json.load(f)
    
    missing_roots = set(reduction_plan['missing_roots'])
    
    # Add flag for missing products
    new_df['is_missing_embedding'] = new_df['filename_root'].isin(missing_roots)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total products: {len(new_df):,}")
    print(f"  Unique filename_roots: {new_df['filename_root'].nunique():,}")
    print(f"  Missing embeddings: {len(missing_roots):,}")
    
    # Analyze by product type
    print(f"\n📊 Product Type Distribution:")
    product_type_stats = new_df.groupby('PRODUCT_TYPE_COD').agg({
        'SKU_COD': 'count',
        'filename_root': 'nunique',
        'is_missing_embedding': 'sum'
    }).rename(columns={
        'SKU_COD': 'total_skus',
        'filename_root': 'unique_roots',
        'is_missing_embedding': 'missing_embeddings'
    })
    
    product_type_stats['has_embeddings'] = product_type_stats['unique_roots'] - product_type_stats['missing_embeddings']
    product_type_stats['missing_percentage'] = (product_type_stats['missing_embeddings'] / product_type_stats['unique_roots'] * 100).round(1)
    
    print(product_type_stats)
    
    # Detailed analysis for each product type
    print(f"\n📋 Detailed Analysis by Product Type:")
    for product_type in sorted(new_df['PRODUCT_TYPE_COD'].unique()):
        print(f"\n--- Product Type: {product_type} ---")
        
        # Get data for this product type
        type_df = new_df[new_df['PRODUCT_TYPE_COD'] == product_type]
        type_missing = type_df[type_df['is_missing_embedding']]
        
        print(f"  Total SKUs: {len(type_df):,}")
        print(f"  Unique filename_roots: {type_df['filename_root'].nunique():,}")
        print(f"  Missing embeddings: {len(type_missing['filename_root'].unique()):,}")
        
        if len(type_missing) > 0:
            # Show sample missing products
            print(f"\n  Sample missing products:")
            sample_missing = type_missing.groupby('filename_root').first().head(5)
            for idx, row in sample_missing.iterrows():
                print(f"    - {idx} | SKU: {row['SKU_COD']} | Brand: {row['BRAND_DES']} | Model: {row['MODEL_COD']}")
            
            # Analyze patterns in missing products
            print(f"\n  Missing by brand:")
            brand_counts = type_missing.groupby('BRAND_DES')['filename_root'].nunique().sort_values(ascending=False).head(10)
            for brand, count in brand_counts.items():
                print(f"    - {brand}: {count} missing")
            
            # Check date patterns
            type_missing['STARTSKU_DATE'] = pd.to_datetime(type_missing['STARTSKU_DATE'])
            date_stats = type_missing.groupby(type_missing['STARTSKU_DATE'].dt.year)['filename_root'].nunique()
            if len(date_stats) > 0:
                print(f"\n  Missing by year:")
                for year, count in date_stats.sort_index().items():
                    print(f"    - {year}: {count} missing")
    
    # Check filename patterns for missing products
    print(f"\n🔍 Filename Pattern Analysis for Missing Products:")
    missing_df = new_df[new_df['is_missing_embedding']]
    
    # Check if missing filenames follow different patterns
    print(f"\n  Filename root length distribution:")
    missing_df['root_length'] = missing_df['filename_root'].str.len()
    length_counts = missing_df['root_length'].value_counts().sort_index()
    for length, count in length_counts.items():
        print(f"    {length} chars: {count} products")
    
    # Check prefixes
    print(f"\n  Common prefixes (first 6 chars):")
    missing_df['prefix'] = missing_df['filename_root'].str[:6]
    prefix_counts = missing_df.groupby('prefix')['filename_root'].nunique().sort_values(ascending=False).head(10)
    for prefix, count in prefix_counts.items():
        print(f"    {prefix}: {count} filename_roots")
    
    # Check if MODEL_COD matches filename prefix
    print(f"\n  Checking MODEL_COD vs filename_root consistency:")
    missing_df['model_str'] = missing_df['MODEL_COD'].astype(str)
    missing_df['filename_prefix'] = missing_df['filename_root'].str[:6]
    missing_df['model_matches'] = missing_df['model_str'] == missing_df['filename_prefix']
    
    match_stats = missing_df['model_matches'].value_counts()
    print(f"    Matches: {match_stats.get(True, 0):,}")
    print(f"    Mismatches: {match_stats.get(False, 0):,}")
    
    if match_stats.get(False, 0) > 0:
        print(f"\n  Sample mismatches:")
        mismatches = missing_df[~missing_df['model_matches']].head(5)
        for idx, row in mismatches.iterrows():
            print(f"    SKU: {row['SKU_COD']} | MODEL: {row['MODEL_COD']} | filename_root: {row['filename_root']}")
    
    # Save detailed report
    with open('filename_product_type_analysis.json', 'w') as f:
        report = {
            'summary': product_type_stats.to_dict(),
            'missing_by_type': {
                str(pt): int(missing_df[missing_df['PRODUCT_TYPE_COD'] == pt]['filename_root'].nunique())
                for pt in sorted(new_df['PRODUCT_TYPE_COD'].unique())
            },
            'total_missing': len(missing_roots),
            'coverage_percentage': reduction_plan['coverage']
        }
        json.dump(report, f, indent=2)
    
    print(f"\n✅ Analysis complete! Report saved to filename_product_type_analysis.json")
    
    return missing_df

if __name__ == "__main__":
    analyze_filename_product_types() 