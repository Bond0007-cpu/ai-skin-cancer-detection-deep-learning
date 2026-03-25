# API Reference Runbook

## Base URL
`https://api.skincancer-ai.com/v1`

## Authentication
```
Authorization: Bearer <JWT_TOKEN>
```

## Endpoints

### POST /predict
Upload a skin lesion image for classification.

**Request**
```json
{
  "image": "<base64_encoded_image>",
  "return_gradcam": true
}
```

**Response**
```json
{
  "prediction": "melanoma",
  "confidence": 0.923,
  "class_probabilities": {
    "nv": 0.031, "mel": 0.923, "bkl": 0.018,
    "bcc": 0.012, "akiec": 0.008, "vasc": 0.004, "df": 0.004
  },
  "gradcam_image": "<base64_heatmap>",
  "risk_level": "HIGH",
  "recommendation": "Immediate dermatologist consultation advised"
}
```

### GET /health
Returns API health status.

### GET /classes
Returns list of supported skin lesion classes.

## Running Locally
```bash
uvicorn src/api/main:app --reload --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
```
