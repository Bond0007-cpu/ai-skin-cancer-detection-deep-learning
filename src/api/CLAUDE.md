# src/api — FastAPI REST Service

## Responsibility
REST API endpoints for serving model predictions to web/mobile clients.

## Files
- `main.py`       — FastAPI app entry point, route registration
- `routes.py`     — Endpoint definitions (/predict, /health, /classes)
- `schemas.py`    — Pydantic request/response models
- `auth.py`       — JWT authentication middleware
- `inference.py`  — Model loading, prediction, Grad-CAM orchestration
- `config.py`     — Environment variables and app settings

## Endpoints
| Method | Path          | Auth | Description                    |
|--------|---------------|------|--------------------------------|
| POST   | /predict      | ✅   | Classify uploaded skin image   |
| GET    | /health       | ❌   | API and model health status    |
| GET    | /classes      | ❌   | List 7 supported lesion classes|
| GET    | /docs         | ❌   | Swagger UI (dev only)          |

## Rules
1. NEVER log uploaded images — HIPAA compliance
2. Always return Grad-CAM heatmap with every /predict response
3. Validate image: accept only JPEG/PNG, max 10MB, min 100×100px
4. Return HTTP 422 for invalid inputs with descriptive error messages
5. Prediction must include confidence score AND top-3 class probabilities
6. Use async endpoints (async def) for all I/O-bound operations

## Running
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
# Swagger: http://localhost:8000/docs
# ReDoc:   http://localhost:8000/redoc
```

## Environment Variables (.env)
```
MODEL_PATH=models/exported/efficientnetb4_savedmodel
JWT_SECRET=<your-secret>
MAX_IMAGE_SIZE_MB=10
LOG_LEVEL=INFO
```
