# src/data — Data Pipeline Module

## Responsibility
All data loading, preprocessing, augmentation, and splitting logic lives here.

## Files
- `preprocess.py` — Main preprocessing pipeline (hair removal, resize, normalize)
- `augmentation.py` — Augmentation transforms (training only)
- `dataset.py` — Custom tf.data / PyTorch Dataset classes
- `split.py` — Stratified train/val/test splitting with SMOTE
- `visualize.py` — Class distribution plots, sample grid visualization

## Rules
1. NEVER apply augmentation to val or test sets
2. ALWAYS apply SMOTE AFTER splitting, not before
3. Use ImageNet normalization for all pretrained model inputs
4. Hair removal must be applied before any augmentation
5. Log class distribution to `data/processed/class_distribution.json`

## Usage
```bash
python src/data/preprocess.py --input data/raw/HAM10000 --output data/processed
```
