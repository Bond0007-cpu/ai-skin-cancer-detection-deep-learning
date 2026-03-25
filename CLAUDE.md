# CLAUDE.md — AI Skin Cancer Detection System

## Project Overview
An AI-powered skin cancer detection system using Machine Learning and Deep Learning.
Classifies dermoscopic images into 7 skin lesion categories with high clinical accuracy.

## Project Goal
Build a production-ready deep learning pipeline that:
- Achieves 95%+ accuracy on skin lesion classification
- Supports 7 cancer types (Melanoma, BCC, SCC, AK, BKL, DF, NV)
- Provides Grad-CAM explainability for clinical use
- Deploys as a HIPAA-compliant REST API + web interface

## Tech Stack
- Python 3.10+
- TensorFlow 2.x / Keras + PyTorch (ensemble)
- OpenCV, NumPy, Pandas, Scikit-learn
- FastAPI (backend) + React (frontend)
- Docker + AWS/GCP deployment
- MLflow for experiment tracking

## Dataset
- Primary: HAM10000 (10,015 images, 7 classes)
- Supplementary: ISIC Archive, PH², Dermnet
- Input size: 224×224×3 (RGB dermoscopic images)

## Model Architecture
- Transfer Learning: EfficientNetB4 (primary), ResNet50, DenseNet121
- Custom head: GAP → Dense(512, ReLU) → Dropout(0.5) → Output(7, Softmax)
- Ensemble: Weighted averaging of top 3 models

## Class Labels
```
0: nv   — Melanocytic nevi (benign)
1: mel  — Melanoma (malignant)
2: bkl  — Benign keratosis-like lesions
3: bcc  — Basal cell carcinoma
4: akiec— Actinic keratoses / Intraepithelial carcinoma
5: vasc — Vascular lesions
6: df   — Dermatofibroma
```

## Key Commands
```bash
# Setup environment
pip install -r requirements.txt

# Download & prepare dataset
python tools/scripts/download_dataset.py
python src/data/preprocess.py

# Train model
python src/training/train.py --model efficientnetb4 --epochs 100

# Evaluate
python src/evaluation/evaluate.py --checkpoint models/best_model.h5

# Run API server
uvicorn src/api/main:app --reload --port 8000

# Docker
docker build -t skin-cancer-api .
docker run -p 8000:8000 skin-cancer-api
```

## Performance Targets
| Metric    | Target |
|-----------|--------|
| Accuracy  | ≥ 95%  |
| Precision | ≥ 93%  |
| Recall    | ≥ 94%  |
| F1-Score  | ≥ 93%  |
| AUC-ROC   | ≥ 0.97 |

## Claude Context Notes
- Always use class weights to handle class imbalance (melanoma is rare)
- Prefer EfficientNetB4 as the base model unless experimenting
- Grad-CAM visualization is required for any prediction output
- All model outputs must include confidence scores and uncertainty estimates
- Follow src/models/CLAUDE.md for architecture conventions
- Follow src/data/CLAUDE.md for all data pipeline work
