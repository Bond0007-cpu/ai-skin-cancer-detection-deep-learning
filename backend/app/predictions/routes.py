"""
Prediction routes — analyze images, get history, stats.
"""

import os
import uuid
from datetime import datetime, timezone
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Query
from bson import ObjectId
from typing import Optional

from app.database import get_database
from app.auth.jwt_handler import get_current_user
from app.ml.model_service import predict, image_to_base64, DISEASE_INFO
from app.predictions.models import PredictionResult, PredictionListResponse, StatsResponse
from app.config import settings

router = APIRouter(prefix="/api/predictions", tags=["Predictions"])

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/jpg", "image/webp"}


def prediction_doc_to_response(doc: dict) -> PredictionResult:
    """Convert MongoDB doc to PredictionResult."""
    return PredictionResult(
        id=str(doc["_id"]),
        user_id=str(doc["user_id"]),
        image_filename=doc.get("image_filename", ""),
        image_base64=doc.get("image_base64"),
        predicted_class=doc["predicted_class"],
        confidence=doc["confidence"],
        class_probabilities=doc["class_probabilities"],
        disease_name=doc["disease_name"],
        description=doc["description"],
        risk_level=doc["risk_level"],
        precautions=doc["precautions"],
        recommendation=doc["recommendation"],
        created_at=doc["created_at"],
    )


@router.post("/analyze", response_model=PredictionResult)
async def analyze_image(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user),
):
    """Upload an image and get skin cancer prediction."""
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid file type '{file.content_type}'. Accepted: JPEG, PNG, WebP",
        )

    # Read file
    img_bytes = await file.read()
    max_size = settings.MAX_IMAGE_SIZE_MB * 1024 * 1024
    if len(img_bytes) > max_size:
        raise HTTPException(
            status_code=422,
            detail=f"Image exceeds {settings.MAX_IMAGE_SIZE_MB}MB limit",
        )

    # Run prediction
    try:
        result = predict(img_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Save image file
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    filename = f"{uuid.uuid4().hex}_{file.filename}"
    filepath = os.path.join(settings.UPLOAD_DIR, filename)
    with open(filepath, "wb") as f:
        f.write(img_bytes)

    # Store prediction in database
    db = get_database()
    prediction_doc = {
        "user_id": current_user["user_id"],
        "image_filename": filename,
        "image_base64": image_to_base64(img_bytes),
        "predicted_class": result["predicted_class"],
        "confidence": result["confidence"],
        "class_probabilities": result["class_probabilities"],
        "disease_name": result["disease_name"],
        "description": result["description"],
        "risk_level": result["risk_level"],
        "precautions": result["precautions"],
        "recommendation": result["recommendation"],
        "created_at": datetime.now(timezone.utc),
    }

    insert_result = await db.predictions.insert_one(prediction_doc)
    prediction_doc["_id"] = insert_result.inserted_id

    # Update user's total scans count
    await db.users.update_one(
        {"_id": ObjectId(current_user["user_id"])},
        {"$inc": {"total_scans": 1}},
    )

    return prediction_doc_to_response(prediction_doc)


@router.get("/history", response_model=PredictionListResponse)
async def get_history(
    page: int = Query(1, ge=1),
    per_page: int = Query(10, ge=1, le=50),
    risk_level: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user),
):
    """Get current user's prediction history with pagination."""
    db = get_database()

    query = {"user_id": current_user["user_id"]}
    if risk_level:
        query["risk_level"] = risk_level.upper()

    total = await db.predictions.count_documents(query)
    skip = (page - 1) * per_page

    cursor = db.predictions.find(query).sort("created_at", -1).skip(skip).limit(per_page)
    predictions = []
    async for doc in cursor:
        predictions.append(prediction_doc_to_response(doc))

    return PredictionListResponse(
        predictions=predictions,
        total=total,
        page=page,
        per_page=per_page,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(current_user: dict = Depends(get_current_user)):
    """Get user's prediction stats for dashboard."""
    db = get_database()
    user_id = current_user["user_id"]

    total = await db.predictions.count_documents({"user_id": user_id})
    high_risk = await db.predictions.count_documents({"user_id": user_id, "risk_level": "HIGH"})
    low_risk = total - high_risk

    # Class distribution
    pipeline = [
        {"$match": {"user_id": user_id}},
        {"$group": {"_id": "$predicted_class", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
    ]
    class_dist = {}
    most_common = None
    async for doc in db.predictions.aggregate(pipeline):
        class_dist[doc["_id"]] = doc["count"]
        if most_common is None:
            most_common = doc["_id"]

    # Monthly scans (last 12 months)
    monthly_pipeline = [
        {"$match": {"user_id": user_id}},
        {
            "$group": {
                "_id": {
                    "year": {"$year": "$created_at"},
                    "month": {"$month": "$created_at"},
                },
                "count": {"$sum": 1},
            }
        },
        {"$sort": {"_id.year": 1, "_id.month": 1}},
        {"$limit": 12},
    ]
    monthly_scans = []
    async for doc in db.predictions.aggregate(monthly_pipeline):
        monthly_scans.append({
            "month": f"{doc['_id']['year']}-{doc['_id']['month']:02d}",
            "count": doc["count"],
        })

    # Recent scans
    cursor = db.predictions.find({"user_id": user_id}).sort("created_at", -1).limit(5)
    recent = []
    async for doc in cursor:
        recent.append(prediction_doc_to_response(doc))

    return StatsResponse(
        total_scans=total,
        high_risk_count=high_risk,
        low_risk_count=low_risk,
        most_common_class=most_common,
        class_distribution=class_dist,
        monthly_scans=monthly_scans,
        recent_scans=recent,
    )


@router.get("/{prediction_id}", response_model=PredictionResult)
async def get_prediction(
    prediction_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Get a single prediction by ID."""
    db = get_database()
    try:
        doc = await db.predictions.find_one({
            "_id": ObjectId(prediction_id),
            "user_id": current_user["user_id"],
        })
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid prediction ID")

    if not doc:
        raise HTTPException(status_code=404, detail="Prediction not found")

    return prediction_doc_to_response(doc)


@router.delete("/{prediction_id}")
async def delete_prediction(
    prediction_id: str,
    current_user: dict = Depends(get_current_user),
):
    """Delete a prediction."""
    db = get_database()
    try:
        result = await db.predictions.delete_one({
            "_id": ObjectId(prediction_id),
            "user_id": current_user["user_id"],
        })
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid prediction ID")

    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Prediction not found")

    # Decrement scan count
    await db.users.update_one(
        {"_id": ObjectId(current_user["user_id"])},
        {"$inc": {"total_scans": -1}},
    )

    return {"message": "Prediction deleted successfully"}


@router.get("/info/diseases")
async def get_disease_info():
    """Get information about all detectable diseases."""
    return DISEASE_INFO
