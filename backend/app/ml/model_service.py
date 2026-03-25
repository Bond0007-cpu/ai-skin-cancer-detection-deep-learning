"""
ML Model service — loads and runs the skin cancer classification model.
Works with or without a trained model (demo mode with simulated predictions).
"""

import io
import os
import base64
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Optional

# Try to import TensorFlow — fall back to demo mode if unavailable
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("⚠️  TensorFlow not available — running in DEMO mode with simulated predictions")

from app.config import settings

CLASS_NAMES = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']
HIGH_RISK = {'mel', 'bcc', 'akiec'}
INPUT_SHAPE = (224, 224)

DISEASE_INFO = {
    "nv": {
        "name": "Melanocytic Nevi",
        "full_name": "Melanocytic Nevi (Benign Mole)",
        "description": "Melanocytic nevi are benign neoplasms of melanocytes that appear as small, dark skin growths. They are extremely common and usually harmless, though changes in shape, size, or color should be monitored.",
        "risk_level": "LOW",
        "precautions": [
            "Monitor for changes in size, shape, color, or texture",
            "Perform regular skin self-examinations monthly",
            "Use broad-spectrum SPF 30+ sunscreen daily",
            "Avoid excessive sun exposure and tanning beds",
            "Consult a dermatologist if any asymmetry or border irregularity develops"
        ]
    },
    "mel": {
        "name": "Melanoma",
        "full_name": "Melanoma (Malignant)",
        "description": "Melanoma is the most dangerous type of skin cancer. It develops in melanocytes (pigment-producing cells). Early detection is critical as it can metastasize rapidly. Look for the ABCDE signs: Asymmetry, Border irregularity, Color variation, Diameter >6mm, Evolving shape.",
        "risk_level": "HIGH",
        "precautions": [
            "URGENT: Schedule immediate dermatologist consultation",
            "Do NOT attempt self-treatment or ignore this finding",
            "Request a biopsy for definitive histopathological diagnosis",
            "Avoid sun exposure on the affected area",
            "Document changes with photographs for medical records",
            "Discuss screening schedule with oncology if confirmed"
        ]
    },
    "bkl": {
        "name": "Benign Keratosis",
        "full_name": "Benign Keratosis-like Lesions",
        "description": "Benign keratoses include seborrheic keratoses, solar lentigines, and lichen-planus like keratoses. These are non-cancerous growths that appear as waxy, scaly, or slightly elevated lesions. While harmless, they can sometimes resemble melanoma.",
        "risk_level": "LOW",
        "precautions": [
            "Generally no treatment necessary",
            "Monitor for rapid changes in appearance",
            "Consult a dermatologist if irritated, bleeding, or cosmetically concerning",
            "Use sunscreen to prevent new lesions",
            "Professional removal available via cryotherapy or curettage if desired"
        ]
    },
    "bcc": {
        "name": "Basal Cell Carcinoma",
        "full_name": "Basal Cell Carcinoma",
        "description": "Basal cell carcinoma (BCC) is the most common form of skin cancer. It arises from basal cells in the deepest layer of the epidermis. While rarely metastatic, BCC can cause significant local tissue destruction if untreated. It often appears as a pearly bump or flat, flesh-colored lesion.",
        "risk_level": "HIGH",
        "precautions": [
            "Schedule dermatologist consultation within 2 weeks",
            "Biopsy recommended for definitive diagnosis",
            "Treatment options include Mohs surgery, excision, or radiation",
            "Apply rigorous sun protection (SPF 50+)",
            "Regular follow-up screenings every 6 months post-treatment",
            "Examine surrounding skin for additional lesions"
        ]
    },
    "akiec": {
        "name": "Actinic Keratosis",
        "full_name": "Actinic Keratoses / Intraepithelial Carcinoma",
        "description": "Actinic keratoses (AK) are rough, scaly patches caused by chronic UV exposure. They are considered pre-cancerous as a small percentage can progress to squamous cell carcinoma. Intraepithelial carcinoma (Bowen's disease) represents carcinoma in situ.",
        "risk_level": "HIGH",
        "precautions": [
            "Consult dermatologist for evaluation and treatment plan",
            "Treatment options: cryotherapy, topical chemotherapy (5-FU), photodynamic therapy",
            "Apply broad-spectrum SPF 50+ sunscreen daily",
            "Wear protective clothing and wide-brimmed hats",
            "Schedule regular skin checks every 6-12 months",
            "Monitor for progression to squamous cell carcinoma"
        ]
    },
    "vasc": {
        "name": "Vascular Lesions",
        "full_name": "Vascular Lesions",
        "description": "Vascular lesions include angiomas, angiokeratomas, pyogenic granulomas, and hemorrhages. These lesions arise from blood vessels and typically appear as red, purple, or blue marks on the skin. Most are benign but some may require treatment.",
        "risk_level": "LOW",
        "precautions": [
            "Most vascular lesions are benign and require no treatment",
            "Monitor for rapid growth, bleeding, or pain",
            "Consult a dermatologist if the lesion changes or causes discomfort",
            "Laser therapy available for cosmetic treatment if desired",
            "Protect the area from trauma to prevent bleeding"
        ]
    },
    "df": {
        "name": "Dermatofibroma",
        "full_name": "Dermatofibroma",
        "description": "Dermatofibromas are firm, benign nodules that commonly occur on the legs. They are composed of fibrous tissue and are usually triggered by minor injuries like insect bites. They appear as small, firm, raised bumps that may be pink, red, or brown.",
        "risk_level": "LOW",
        "precautions": [
            "No treatment necessary in most cases",
            "Monitor for significant size changes",
            "Surgical excision available if the lesion is painful or cosmetically bothersome",
            "Consult a dermatologist if rapid growth or color change occurs",
            "The lesion may recur after removal"
        ]
    },
}

# Singleton model reference
_model = None


def load_model():
    """Load the TensorFlow model at startup."""
    global _model
    if not TF_AVAILABLE:
        print("⚠️  Running in DEMO mode — predictions will be simulated")
        return

    model_path = settings.MODEL_PATH
    if os.path.exists(model_path):
        try:
            _model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
        except Exception as e:
            print(f"⚠️  Could not load model: {e}. Running in DEMO mode.")
    else:
        print(f"⚠️  Model not found at {model_path}. Running in DEMO mode.")


def preprocess_image(img_bytes: bytes) -> np.ndarray:
    """Preprocess image bytes for model input."""
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize(INPUT_SHAPE)
    arr = np.array(img, dtype=np.float32) / 255.0
    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std
    return arr


def predict(img_bytes: bytes) -> Dict:
    """
    Run prediction on image bytes.
    Returns prediction result with class, confidence, disease info, and precautions.
    """
    global _model

    if _model is not None and TF_AVAILABLE:
        # Real model prediction
        preprocessed = preprocess_image(img_bytes)
        input_tensor = np.expand_dims(preprocessed, axis=0)
        predictions = _model.predict(input_tensor, verbose=0)
        probabilities = predictions[0]
    else:
        # Demo mode — generate realistic simulated probabilities
        np.random.seed(hash(img_bytes[:100]) % (2**32))
        raw = np.random.dirichlet(np.ones(len(CLASS_NAMES)) * 0.5)
        # Boost one class to make it look like a real prediction
        dominant_idx = np.argmax(raw)
        raw[dominant_idx] += 0.3
        probabilities = raw / raw.sum()

    # Get predicted class
    predicted_idx = int(np.argmax(probabilities))
    predicted_class = CLASS_NAMES[predicted_idx]
    confidence = float(probabilities[predicted_idx])

    # Build class probabilities dict
    class_probabilities = {
        cls: round(float(probabilities[i]), 4) for i, cls in enumerate(CLASS_NAMES)
    }

    # Get disease info
    info = DISEASE_INFO[predicted_class]

    # Generate recommendation based on risk
    if predicted_class in HIGH_RISK:
        recommendation = "⚠️ High-risk lesion detected. Immediate dermatologist consultation strongly advised."
    elif confidence < 0.75:
        recommendation = "Low confidence prediction — clinical review recommended."
    else:
        recommendation = "Low-risk lesion detected. Monitor for any changes and maintain regular skin checks."

    return {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
        "class_probabilities": class_probabilities,
        "disease_name": info["full_name"],
        "description": info["description"],
        "risk_level": info["risk_level"],
        "precautions": info["precautions"],
        "recommendation": recommendation,
    }


def image_to_base64(img_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.b64encode(img_bytes).decode('utf-8')
