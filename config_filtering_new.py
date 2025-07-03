"""Filter configuration for the new database"""

FILTER_CONFIG = {
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
                "multiple": true
            },
            "GENDER": {
                "type": "select",
                "label": "Gender",
                "placeholder": "Select gender...",
                "description": "Filter by gender",
                "multiple": true
            },
            "AGE_GROUP": {
                "type": "select",
                "label": "Age Group",
                "placeholder": "Select age group...",
                "description": "Filter by age group",
                "multiple": true
            },
            "PRODUCT_TYPE": {
                "type": "select",
                "label": "Product Type",
                "placeholder": "Select product type...",
                "description": "Filter by product type",
                "multiple": true
            },
            "STATUS": {
                "type": "select",
                "label": "Status",
                "placeholder": "Select status...",
                "description": "Filter by status",
                "multiple": true
            }
        },
        "range_filters": {
            "LENS_WIDTH_VAL": {
                "type": "range",
                "label": "Lens Width",
                "unit": "mm",
                "tolerance": 10,
                "description": "Filter by lens width (\u00b110%)"
            },
            "LENS_HEIGHT_VAL": {
                "type": "range",
                "label": "Lens Height",
                "unit": "mm",
                "tolerance": 10,
                "description": "Filter by lens height (\u00b110%)"
            },
            "BRIDGE_LENGTH_VAL": {
                "type": "range",
                "label": "Bridge Length",
                "unit": "mm",
                "tolerance": 10,
                "description": "Filter by bridge length (\u00b110%)"
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
                "multiple": true
            },
            "SHAPE_SEMI_GROUPED": {
                "type": "select",
                "label": "Shape Category",
                "placeholder": "Select shape...",
                "description": "Filter by shape category",
                "multiple": true
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
    "exact_match_columns": [
        "MODEL_COD",
        "SKU_COD",
        "COLOR"
    ],
    "removed_columns": [
        "MATERIALGROUP_DES",
        "CONCEPT_01_DES",
        "SKU_URL_MEDIUM",
        "SPECIAL_SKU_FLG",
        "TEMPLE_LENGTH_VAL",
        "VAR_LENS_HEIGHT_VAL",
        "LENS_BASE_DES",
        "MACRO_SHAPE_DES",
        "SKU_STATUS_HIST_DAILY_COD",
        "VAR_LENS_BASE_VAL",
        "GRANULAR_SHAPE_DES",
        "VAR_TEMPLE_LENGTH_VAL",
        "FlatTop_Confidence_1",
        "FLG_SECOND_CHOICE",
        "CONCEPT_02_DES",
        "ACT_SKU_PRICE_RANGE_DES",
        "FITTING_DES",
        "FIRST_FRONT_MAT_DES",
        "PORTFOLIO_PRICE_RANGE_DES"
    ],
    "new_columns": [
        "COLOR",
        "CTM_FIRST_TEMPLE_MATERIAL_DES",
        "SHAPE_SEMI_GROUPED",
        "BRIDGE_LENGTH_VAL"
    ]
}
