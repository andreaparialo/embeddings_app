# Dual Engine Search Logic

## Overview

The dual engine search combines results from two FAISS indexes to provide more accurate product matching by leveraging both visual and technical features.

## Indexes

### 1. GME Index (Visual Features)
- **Type**: IndexFlatIP (Inner Product)
- **Dimensions**: 3584
- **Model**: GME-Qwen2-VL-7B with LoRA fine-tuning
- **Purpose**: Captures visual similarity between products based on image embeddings

### 2. Measurement Index (Technical Features)
- **Type**: IndexFlatIP (Inner Product)  
- **Dimensions**: 256
- **Purpose**: Captures technical/measurement-based similarities between products

Both indexes use cosine similarity via normalized embeddings.

## Search Modes

### Mode 1: Global Search
- Searches across the entire product database
- No pre-filtering applied
- Best for exploratory searches where specific criteria aren't known
- More comprehensive but potentially slower with large datasets

### Mode 2: Filtered Search
- Pre-filters products based on selected column criteria
- Only searches within the filtered subset
- Faster and more targeted results
- Ideal when specific product attributes are known (e.g., category, brand, size)

## Search Process

### 1. Pre-filtering (Mode 2 only)
```python
# Apply column-based filters before FAISS search
filtered_indices = apply_filters(selected_columns, filter_values)
```

### 2. Dual Index Search
```python
# Search both indexes with the same query
gme_scores, gme_indices = gme_index.search(query_embedding, k)
measurement_scores, measurement_indices = measurement_index.search(query_embedding, k)
```

### 3. Score Combination
```python
# Weighted combination of scores
final_score = (main_weight × gme_score) + (measurement_weight × measurement_score)
```

### 4. Result Ranking
- Results are ranked by combined score
- Duplicate products are merged, keeping the highest score
- Top-k results are returned

## Weight Configurations

### Preset Options
1. **Visual Focus (70/30)**
   - 70% GME (visual) weight
   - 30% Measurement weight
   - Best for visually similar products

2. **Balanced (50/50)**
   - 50% GME weight
   - 50% Measurement weight
   - Equal importance to visual and technical features

3. **Technical Focus (30/70)**
   - 30% GME weight
   - 70% Measurement weight
   - Prioritizes technical specifications

### Custom Weights
Users can adjust weight sliders for fine-tuned control over the importance of each index.

## Implementation Details

### Batch Search
- Processes multiple query images simultaneously
- Applies the same dual engine logic to each image
- Results can be exported to Excel with combined scores

### Performance Optimizations
- Pre-filtering reduces search space significantly
- GPU acceleration for FAISS operations
- Batch processing achieves 100+ images/second
- Normalized embeddings enable fast cosine similarity

## Score Normalization

Since both indexes use cosine similarity with normalized embeddings:
- Scores range from 0 to 1
- Higher scores indicate better matches
- Combined scores maintain the same range due to weighted averaging

## Use Cases

1. **Visual-Heavy Search**: Finding products that look similar
   - Use Mode 1 (Global) + Visual Focus weights

2. **Category-Specific Search**: Finding similar products within a category
   - Use Mode 2 (Filtered) + Balanced weights

3. **Technical Specification Match**: Finding products with similar measurements
   - Use Mode 2 (Filtered) + Technical Focus weights

4. **Comprehensive Search**: Maximum accuracy across all features
   - Use Mode 1 (Global) + Balanced weights