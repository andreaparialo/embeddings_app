# 🎛️ SKU Search Weight Controls - Complete Implementation

## 📊 **Current Dual Index Scoring Formula**

### **Step 1: Distance to Similarity Conversion**
```python
main_similarity = 1.0 / (1.0 + main_distance)
measurement_similarity = 1.0 / (1.0 + measurement_distance)
```

### **Step 2: Normalization (0-1 range)**
```python
main_similarities = normalize(main_similarities)
measurement_similarities = normalize(measurement_similarities)
```

### **Step 3: Weighted Combination**
```python
# For products in both indexes:
combined_score = main_similarity * main_weight + measurement_similarity * measurement_weight

# For products in main only:
combined_score = main_similarity * main_weight

# For products in measurement only:
combined_score = measurement_similarity * measurement_weight
```

## 🎛️ **Weight Controls Added**

### **UI Controls (SKU Search Section)**
```html
<!-- Weight Controls for SKU Search -->
<div id="sku-dual-weights-controls" style="display: none;">
    <div class="row">
        <div class="col-6">
            <label class="form-label small">Visual Weight</label>
            <input type="range" id="sku-main-weight" min="0" max="1" step="0.1" value="0.7">
            <div class="text-center small" id="sku-main-weight-display">70%</div>
        </div>
        <div class="col-6">
            <label class="form-label small">Technical Weight</label>
            <input type="range" id="sku-measurement-weight" min="0" max="1" step="0.1" value="0.3">
            <div class="text-center small" id="sku-measurement-weight-display">30%</div>
        </div>
    </div>
    <div class="mt-2">
        <small class="text-muted">
            <strong>Visual:</strong> GME image similarity | <strong>Technical:</strong> Measurement features
        </small>
    </div>
</div>
```

### **JavaScript Behavior**
- **Linked sliders**: When you move Visual weight, Technical weight auto-adjusts
- **Always sum to 100%**: Weights are normalized to always total 1.0
- **Show/Hide**: Controls appear when "Enable Dual-Index Search" is checked
- **Real-time display**: Percentage updates as you move sliders

### **Backend Integration**
```python
@app.post("/search/sku")
async def search_by_sku(
    # ... other parameters
    main_weight: float = Form(0.7),           # Visual weight
    measurement_weight: float = Form(0.3)     # Technical weight
):
    # Weights are passed to dual index search
    similar_results = dual_search_engine.search_by_filename_similarity_dual(
        filename_root,
        main_weight=main_weight,
        measurement_weight=measurement_weight
    )
```

## 🎯 **How to Use**

### **Step 1: Enable Dual-Index Search**
- ✅ Check "Enable Dual-Index Search" checkbox
- 🎛️ Weight controls appear below

### **Step 2: Adjust Weights**
- **Visual Weight (0-100%)**: How much GME image similarity matters
- **Technical Weight (0-100%)**: How much measurement features matter
- **Examples**:
  - `Visual: 90%, Technical: 10%` → Prioritize visual similarity
  - `Visual: 50%, Technical: 50%` → Equal balance
  - `Visual: 30%, Technical: 70%` → Prioritize technical features

### **Step 3: Search**
- Enter SKU and search
- Results ranked by combined weighted score
- Console shows: `SKU dual-index search with weights: Visual=0.7, Technical=0.3`

## 🧪 **Weight Impact Examples**

### **High Visual Weight (90% Visual, 10% Technical)**
```
Product A: visual_sim=0.9, technical_sim=0.3
→ combined_score = 0.9 * 0.9 + 0.3 * 0.1 = 0.84

Product B: visual_sim=0.7, technical_sim=0.8  
→ combined_score = 0.7 * 0.9 + 0.8 * 0.1 = 0.71

Result: Product A ranks higher (looks more similar)
```

### **High Technical Weight (30% Visual, 70% Technical)**
```
Product A: visual_sim=0.9, technical_sim=0.3
→ combined_score = 0.9 * 0.3 + 0.3 * 0.7 = 0.48

Product B: visual_sim=0.7, technical_sim=0.8
→ combined_score = 0.7 * 0.3 + 0.8 * 0.7 = 0.77

Result: Product B ranks higher (technically more similar)
```

## 🎯 **Response Data**

When dual engine is used, response includes:
```json
{
  "dual_engine": true,
  "main_weight": 0.7,
  "measurement_weight": 0.3,
  "scoring_formula": "visual_similarity * 0.70 + technical_similarity * 0.30",
  "results": [...],
  "source_sku": {...}
}
```

## ✅ **Complete Features**

- 🎛️ **Interactive sliders** with real-time percentage display
- 🔄 **Auto-balancing** weights (always sum to 100%)
- 📱 **Show/hide** based on dual engine checkbox
- 🔧 **Backend integration** with custom weights
- 📊 **Logging** of weights used
- 📈 **Formula transparency** in response

**Users can now fine-tune the balance between visual and technical similarity!** 🎛️
