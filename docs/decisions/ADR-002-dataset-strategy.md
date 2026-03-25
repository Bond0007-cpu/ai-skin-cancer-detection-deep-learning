# ADR-002: Dataset Strategy

## Status: Accepted

## Context
Skin lesion datasets are heavily imbalanced. HAM10000 has 67% nv (benign)
and only 1.1% df class. Need a strategy to handle this without data leakage.

## Decision
Use **class-weighted loss + SMOTE on training split only** (not before splitting).

## Rationale
- Applying SMOTE before splitting causes data leakage into val/test sets
- Class weights in loss function are computationally free and effective
- SMOTE on minority classes brings all classes to ~1500 samples minimum
- Stratified split preserves class distribution in val/test sets

## Class Weight Formula
```python
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
```

## Data Sources Priority
1. HAM10000 — 10,015 validated dermoscopic images (primary)
2. ISIC 2019 Archive — 25,331 additional images (supplementary)
3. PH² Database — 200 dermoscopic images (test validation only)
4. Dermnet NZ — Clinical web images (low priority, noisy labels)
