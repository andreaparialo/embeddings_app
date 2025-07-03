#!/usr/bin/env python3
"""
Update app.py and related files to use the new database and indexes
"""

import os
import json
import pandas as pd
from datetime import datetime

def update_app_files():
    print("🔧 Updating App Configuration for New Database")
    print("=" * 80)
    
    # Configuration
    new_db_path = "database_results/DB_FINAL_SIMILARIT_270615.csv"
    new_index_path = "indexes/v11_merged_latest.faiss"
    
    # Load the new database to analyze columns
    print("\n📊 Analyzing new database structure...")
    new_db = pd.read_csv(new_db_path)
    new_columns = set(new_db.columns)
    
    # Columns to remove from filtering logic
    removed_columns = {
        'FITTING_DES', 'LENS_BASE_DES', 'TEMPLE_LENGTH_VAL',
        'ACT_SKU_PRICE_RANGE_DES', 'CONCEPT_01_DES', 'CONCEPT_02_DES',
        'FIRST_FRONT_MAT_DES', 'FLG_SECOND_CHOICE', 'FlatTop_Confidence_1',
        'GRANULAR_SHAPE_DES', 'MACRO_SHAPE_DES', 'MATERIALGROUP_DES',
        'PORTFOLIO_PRICE_RANGE_DES', 'SKU_STATUS_HIST_DAILY_COD',
        'SKU_URL_MEDIUM', 'SPECIAL_SKU_FLG', 'VAR_LENS_BASE_VAL',
        'VAR_LENS_HEIGHT_VAL', 'VAR_TEMPLE_LENGTH_VAL'
    }
    
    # New columns to add to filtering
    new_filter_columns = {
        'COLOR': 'text',  # Color codes
        'CTM_FIRST_TEMPLE_MATERIAL_DES': 'select',  # 4 options
        'SHAPE_SEMI_GROUPED': 'select',  # 24 options
        'BRIDGE_LENGTH_VAL': 'range'  # Numeric range with ±10%
    }
    
    # Create backup directory
    backup_dir = f"backups/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(backup_dir, exist_ok=True)
    
    # 1. Update data_loader.py
    print("\n📝 Updating data_loader.py...")
    with open("data_loader.py", 'r') as f:
        content = f.read()
    
    # Backup original
    with open(f"{backup_dir}/data_loader.py.bak", 'w') as f:
        f.write(content)
    
    # Update database path
    content = content.replace(
        'final_with_aws_shapes_enriched.csv',
        'DB_FINAL_SIMILARIT_270615.csv'
    )
    
    # Update index references
    content = content.replace(
        'v11-20250620-105815/indexes/v11-20250620-105815_embeddings.npy',
        'v11_merged_latest_embeddings.npy'
    )
    content = content.replace(
        'v11-20250620-105815/indexes/v11-20250620-105815_metadata.json',
        'v11_merged_latest_metadata.json'
    )
    content = content.replace(
        'v11-20250620-105815/indexes/v11-20250620-105815.faiss',
        'v11_merged_latest.faiss'
    )
    
    with open("data_loader.py", 'w') as f:
        f.write(content)
    print("  ✅ Updated data_loader.py")
    
    # 2. Update config_filtering.py
    print("\n📝 Updating config_filtering.py...")
    
    # Create new filter configuration
    filter_config = {
        "filters": {
            "pre_filters": {
                "MODEL_COD": {
                    "type": "text",
                    "label": "Model Code",
                    "placeholder": "Enter model code...",
                    "description": "Exact model code match"
                },
                "BRAND_DES": {
                    "type": "select",
                    "label": "Brand",
                    "placeholder": "Select brand...",
                    "description": "Filter by brand",
                    "multiple": True
                },
                "GENDER": {
                    "type": "select",
                    "label": "Gender",
                    "placeholder": "Select gender...",
                    "description": "Filter by gender",
                    "multiple": True
                },
                "AGE_GROUP": {
                    "type": "select",
                    "label": "Age Group",
                    "placeholder": "Select age group...",
                    "description": "Filter by age group",
                    "multiple": True
                },
                "PRODUCT_TYPE": {
                    "type": "select",
                    "label": "Product Type",
                    "placeholder": "Select product type...",
                    "description": "Filter by product type",
                    "multiple": True
                },
                "STATUS": {
                    "type": "select",
                    "label": "Status",
                    "placeholder": "Select status...",
                    "description": "Filter by status",
                    "multiple": True
                }
            },
            "range_filters": {
                "LENS_WIDTH_VAL": {
                    "type": "range",
                    "label": "Lens Width",
                    "unit": "mm",
                    "tolerance": 10,
                    "description": "Filter by lens width (±10%)"
                },
                "LENS_HEIGHT_VAL": {
                    "type": "range",
                    "label": "Lens Height",
                    "unit": "mm",
                    "tolerance": 10,
                    "description": "Filter by lens height (±10%)"
                },
                "BRIDGE_LENGTH_VAL": {
                    "type": "range",
                    "label": "Bridge Length",
                    "unit": "mm",
                    "tolerance": 10,
                    "description": "Filter by bridge length (±10%)"
                },
                "STARTSKU_DATE": {
                    "type": "date_range",
                    "label": "Start Date",
                    "description": "Filter by product start date"
                }
            },
            "additional_filters": {
                "COLOR": {
                    "type": "text",
                    "label": "Color Code",
                    "placeholder": "Enter 3-digit color code...",
                    "description": "Filter by color code (e.g., 807, PJP, 086)"
                },
                "CTM_FIRST_TEMPLE_MATERIAL_DES": {
                    "type": "select",
                    "label": "Temple Material",
                    "placeholder": "Select material...",
                    "description": "Filter by temple material",
                    "multiple": True
                },
                "SHAPE_SEMI_GROUPED": {
                    "type": "select",
                    "label": "Shape Category",
                    "placeholder": "Select shape...",
                    "description": "Filter by shape category",
                    "multiple": True
                }
            }
        },
        "priority_columns": [
            "MODEL_COD",
            "BRAND_DES",
            "GENDER",
            "AGE_GROUP",
            "PRODUCT_TYPE",
            "SHAPE_SEMI_GROUPED",
            "CTM_FIRST_TEMPLE_MATERIAL_DES",
            "COLOR"
        ],
        "exact_match_columns": ["MODEL_COD", "SKU_COD", "COLOR"],
        "removed_columns": list(removed_columns),
        "new_columns": list(new_filter_columns.keys())
    }
    
    # Write new config
    with open("config_filtering_new.py", 'w') as f:
        f.write('"""Filter configuration for the new database"""\n\n')
        f.write('FILTER_CONFIG = ')
        f.write(json.dumps(filter_config, indent=4))
        f.write('\n')
    
    print("  ✅ Created config_filtering_new.py")
    
    # 3. Create migration summary
    print("\n📝 Creating migration summary...")
    
    migration_summary = {
        "timestamp": datetime.now().isoformat(),
        "database": {
            "old": "final_with_aws_shapes_enriched.csv",
            "new": "DB_FINAL_SIMILARIT_270615.csv",
            "old_rows": 34431,
            "new_rows": 12601,
            "old_columns": 40,
            "new_columns": 25
        },
        "index": {
            "old": "v11-20250620-105815",
            "new": "v11_merged_latest",
            "old_embeddings": 29136,
            "new_embeddings": "TBD (9927 + delta)"
        },
        "columns": {
            "removed": list(removed_columns),
            "added": list(new_filter_columns.keys()),
            "kept": list(new_columns - removed_columns - set(new_filter_columns.keys()))
        },
        "filters": {
            "removed_filters": ["FITTING_DES", "LENS_BASE_DES", "TEMPLE_LENGTH_VAL"],
            "added_filters": list(new_filter_columns.keys()),
            "updated_logic": [
                "COLOR column ensured as 3-digit text",
                "BRIDGE_LENGTH_VAL replaces TEMPLE_LENGTH_VAL with ±10% tolerance",
                "New shape and material filters added"
            ]
        }
    }
    
    with open("migration_summary.json", 'w') as f:
        json.dump(migration_summary, f, indent=2)
    
    print("  ✅ Created migration_summary.json")
    
    # 4. Create update instructions
    print("\n📝 Creating update instructions...")
    
    instructions = """
# Database Migration Instructions

## Manual Steps Required:

1. **Update app.py**:
   - Replace `final_with_aws_shapes_enriched.csv` with `DB_FINAL_SIMILARIT_270615.csv`
   - Update filter logic to remove old columns and add new ones
   - Ensure COLOR column is treated as text (not numeric)

2. **Update templates/index.html**:
   - Remove filter UI elements for: FITTING_DES, LENS_BASE_DES, TEMPLE_LENGTH_VAL
   - Add new filter elements for: COLOR, CTM_FIRST_TEMPLE_MATERIAL_DES, SHAPE_SEMI_GROUPED, BRIDGE_LENGTH_VAL
   - Update filter labels and placeholders

3. **Test the following**:
   - Image search with new index
   - SKU search with new database
   - All filter combinations
   - Batch processing Excel upload
   - Performance with new reduced dataset

4. **Monitor for issues**:
   - Check for any missing column errors
   - Verify COLOR codes display correctly (3 digits)
   - Test BRIDGE_LENGTH_VAL range filtering
   - Ensure new shape/material filters work

## Automated Updates Applied:
- ✅ data_loader.py - Updated paths
- ✅ config_filtering_new.py - Created new filter configuration
- ✅ Backup created in: {backup_dir}
"""
    
    with open("UPDATE_INSTRUCTIONS.md", 'w') as f:
        f.write(instructions.format(backup_dir=backup_dir))
    
    print("  ✅ Created UPDATE_INSTRUCTIONS.md")
    
    print(f"\n✅ Update preparation complete!")
    print(f"📁 Backups saved to: {backup_dir}")
    print(f"📋 Next steps: Follow UPDATE_INSTRUCTIONS.md")

if __name__ == "__main__":
    update_app_files() 