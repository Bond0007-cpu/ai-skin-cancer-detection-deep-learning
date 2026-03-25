"""
AI Skin Cancer Detection — FastAPI Service
Endpoints: /predict, /health, /classes
"""

import base64
import io
import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from PIL import Image
from pydantic import BaseModel
from typing import Dict

from src.models.gradcam import predict_with_explanation

# ── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Skin Cancer Detection API",
    description="Dermoscopic image classifier — 7 lesion classes with Grad-CAM explainability",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

security = HTTPBearer()

CLASS_NAMES   = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
HIGH_RISK     = {'mel', 'bcc', 'akiec'}
MODEL_PATH    = os.getenv('MODEL_PATH', 'models/exported/efficientnetb4_savedmodel')
MAX_IMG_BYTES = int(os.getenv('MAX_IMAGE_SIZE_MB', 10)) * 1024 * 1024
INPUT_SHAPE   = (224, 224)

# Load model at startup
_model: tf.keras.Model = None


@app.on_event("startup")
def load_model():
    global _model
    print(f"Loading model from {MODEL_PATH}...")
    _model = tf.keras.models.load_model(MODEL_PATH)
    print("✅ Model loaded successfully")


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    prediction:           str
    confidence:           float
    class_probabilities:  Dict[str, float]
    risk_level:           str
    recommendation:       str
    gradcam_image:        str  # base64-encoded PNG


class HealthResponse(BaseModel):
    status:  str
    model:   str
    version: str


# ── Helpers ───────────────────────────────────────────────────────────────────
def preprocess_image(img_bytes: bytes) -> np.ndarray:
    """Decode, resize and normalize uploaded image for model input."""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize(INPUT_SHAPE)
    arr = np.array(img, dtype=np.float32) / 255.0
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    arr  = (arr - mean) / std
    return arr


def get_recommendation(predicted_class: str, confidence: float) -> str:
    if predicted_class in HIGH_RISK:
        return "Immediate dermatologist consultation strongly advised."
    if confidence < 0.75:
        return "Low confidence prediction — clinical review recommended."
    return "Low-risk lesion detected. Monitor for any changes."


def array_to_base64(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('utf-8')


# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    """Classify a dermoscopic skin lesion image and return Grad-CAM explanation."""
    if file.content_type not in ('image/jpeg', 'image/png'):
        raise HTTPException(status_code=422, detail="Only JPEG and PNG images are accepted.")

    img_bytes = await file.read()
    if len(img_bytes) > MAX_IMG_BYTES:
        raise HTTPException(status_code=422, detail=f"Image exceeds {MAX_IMG_BYTES // (1024*1024)}MB limit.")

    try:
        image = preprocess_image(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid image: {str(e)}")

    pred_class, confidence, gradcam_overlay, class_probs = predict_with_explanation(
        _model, image, model_name='efficientnetb4'
    )

    return PredictionResponse(
        prediction=pred_class,
        confidence=round(confidence, 4),
        class_probabilities=class_probs,
        risk_level='HIGH' if pred_class in HIGH_RISK else 'LOW',
        recommendation=get_recommendation(pred_class, confidence),
        gradcam_image=array_to_base64(gradcam_overlay),
    )


@app.get("/health", response_model=HealthResponse)
def health():
    """Check API and model health."""
    return HealthResponse(status="ok", model="efficientnetb4", version="1.0.0")


@app.get("/classes")
def get_classes():
    """List all supported skin lesion classes."""
    return {
        "classes": CLASS_NAMES,
        "descriptions": {
            "nv":    "Melanocytic nevi (benign mole)",
            "mel":   "Melanoma (malignant)",
            "bkl":   "Benign keratosis-like lesions",
            "bcc":   "Basal cell carcinoma",
            "akiec": "Actinic keratoses / Intraepithelial carcinoma",
            "vasc":  "Vascular lesions",
            "df":    "Dermatofibroma",
        }
    }
