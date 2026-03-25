# src/models — Model Architecture Module

## Responsibility
All model definitions, layer configurations, and ensemble logic.

## Files
- `efficientnet.py`  — EfficientNetB4 with custom classification head
- `resnet.py`        — ResNet50 with custom classification head
- `densenet.py`      — DenseNet121 with custom classification head
- `ensemble.py`      — Weighted ensemble of multiple models
- `gradcam.py`       — Grad-CAM explainability implementation
- `base_model.py`    — Abstract base class all models inherit from

## Architecture Convention
Every model MUST follow this head structure:
```
BaseModel (pretrained, frozen initially)
  → GlobalAveragePooling2D
  → Dense(512, activation='relu')
  → BatchNormalization
  → Dropout(0.5)
  → Dense(7, activation='softmax')   # 7 skin lesion classes
```

## Class Labels (index → label)
```python
CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
```

## Rules
1. All models must accept input shape (None, 224, 224, 3)
2. Grad-CAM MUST target the last convolutional layer
3. Ensemble weights must be stored in `.claude/settings.json`
4. Always save model summary to `models/architecture_summary.txt`
5. Use `model.save()` in SavedModel format, not legacy HDF5

## Usage
```python
from src.models.efficientnet import build_efficientnet
model = build_efficientnet(num_classes=7, input_shape=(224, 224, 3))
model.summary()
```
