#!/usr/bin/env python3
"""
Verify COLOR column data quality
"""

import pandas as pd

def verify_color_codes():
    # Read the CSV
    df = pd.read_csv('database_results/DB_FINAL_SIMILARIT_270615.csv')
    
    print("🎨 COLOR Column Verification")
    print("=" * 80)
    
    # Ensure COLOR is treated as string
    df['COLOR'] = df['COLOR'].astype(str)
    
    # Check length of color codes
    print("\n📏 Color Code Lengths:")
    df['color_length'] = df['COLOR'].str.len()
    length_counts = df['color_length'].value_counts().sort_index()
    for length, count in length_counts.items():
        print(f"  {length} characters: {count:,} codes")
    
    # Find non-3-digit codes
    non_3_digit = df[df['color_length'] != 3]
    if len(non_3_digit) > 0:
        print(f"\n⚠️  Found {len(non_3_digit)} non-3-digit color codes:")
        # Show unique non-3-digit codes
        unique_non_3 = non_3_digit['COLOR'].unique()
        print(f"  Unique values: {list(unique_non_3[:20])}")  # Show first 20
        
        # Group by length
        for length in sorted(non_3_digit['color_length'].unique()):
            codes = non_3_digit[non_3_digit['color_length'] == length]['COLOR'].unique()
            print(f"\n  {length}-digit codes ({len(codes)} unique):")
            for code in codes[:10]:  # Show first 10
                count = df[df['COLOR'] == code].shape[0]
                print(f"    '{code}' ({count} occurrences)")
    else:
        print("\n✅ All color codes are exactly 3 digits!")
    
    # Check if numeric codes preserve leading zeros
    print("\n🔢 Checking Leading Zeros:")
    # Find codes that start with 0
    zero_start = df[df['COLOR'].str.startswith('0')]
    print(f"  Codes starting with '0': {len(zero_start):,}")
    if len(zero_start) > 0:
        print("  Examples:", list(zero_start['COLOR'].unique()[:10]))
    
    # Find codes that start with 00
    double_zero_start = df[df['COLOR'].str.startswith('00')]
    print(f"  Codes starting with '00': {len(double_zero_start):,}")
    if len(double_zero_start) > 0:
        print("  Examples:", list(double_zero_start['COLOR'].unique()[:10]))
    
    # Check if any codes are purely numeric
    print("\n🔤 Code Type Analysis:")
    numeric_pattern = df['COLOR'].str.match(r'^\d+$')
    alpha_pattern = df['COLOR'].str.match(r'^[A-Za-z]+$')
    mixed_pattern = ~(numeric_pattern | alpha_pattern)
    
    print(f"  Purely numeric (e.g., '123'): {numeric_pattern.sum():,}")
    print(f"  Purely alphabetic (e.g., 'ABC'): {alpha_pattern.sum():,}")
    print(f"  Mixed (e.g., '1A2'): {mixed_pattern.sum():,}")
    
    # Show top color codes
    print("\n🎯 Top 20 Color Codes:")
    top_colors = df['COLOR'].value_counts().head(20)
    for color, count in top_colors.items():
        print(f"  '{color}': {count:,} products")
    
    # Save detailed report
    with open('color_verification_report.txt', 'w') as f:
        f.write("COLOR CODE VERIFICATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total unique color codes: {df['COLOR'].nunique()}\n")
        f.write(f"Total products: {len(df)}\n\n")
        
        f.write("Length distribution:\n")
        for length, count in length_counts.items():
            f.write(f"  {length} chars: {count}\n")
        
        if len(non_3_digit) > 0:
            f.write(f"\nNon-3-digit codes: {len(non_3_digit)}\n")
            f.write("All non-3-digit values:\n")
            for code in unique_non_3:
                count = df[df['COLOR'] == code].shape[0]
                f.write(f"  '{code}' ({count} occurrences)\n")
    
    print("\n✅ Verification complete! Report saved to color_verification_report.txt")
    
    return df

if __name__ == "__main__":
    verify_color_codes() 