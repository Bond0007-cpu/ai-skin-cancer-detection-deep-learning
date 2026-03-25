# System Architecture

## Pipeline Overview

```
Raw Images → Preprocessing → Augmentation → Model Training → Evaluation → Deployment
```

## Phase 1 — Data Collection
- **Sources**: HAM10000 (primary), ISIC Archive, PH², Dermnet
- **Classes**: 7 (nv, mel, bkl, bcc, akiec, vasc, df)
- **Format**: Dermoscopic JPEG/PNG, 224×224 to 512×512 px

## Phase 2 — Preprocessing
- Hair removal (Inpainting + Morphological ops)
- Normalization (ImageNet mean/std)
- Resize to 224×224×3
- SMOTE + class-weighted oversampling for imbalance
- Split: 70% train / 15% val / 15% test

## Phase 3 — Model Architecture
- Base: EfficientNetB4 (pretrained on ImageNet)
- Head: GlobalAveragePooling → Dense(512, ReLU) → Dropout(0.5) → Dense(7, Softmax)
- Ensemble: EfficientNetB4 + DenseNet121 + ResNet50 (weighted avg)

## Phase 4 — Training
- Optimizer: Adam (lr=1e-4) with ReduceLROnPlateau
- Loss: Categorical Cross-Entropy with class weights
- Regularization: Dropout(0.5), L2, EarlyStopping(patience=10)
- Epochs: 100, Batch: 32, GPU: CUDA/MPS

## Phase 5 — Evaluation
- Metrics: Accuracy, Precision, Recall, F1-Score, AUC-ROC, Cohen's Kappa
- Validation: Stratified 5-fold cross-validation
- Explainability: Grad-CAM heatmaps per prediction

## Phase 6 — Deployment
- Backend: FastAPI (Python) serving model via REST endpoints
- Frontend: React web UI with drag-and-drop image upload
- Containerization: Docker (multi-stage build)
- Cloud: AWS EC2 / GCP Cloud Run with auto-scaling
- Security: HTTPS, JWT auth, HIPAA-compliant logging
