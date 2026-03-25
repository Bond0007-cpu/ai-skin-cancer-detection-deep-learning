"""
Pydantic models for predictions.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime


class PredictionResult(BaseModel):
    id: str
    user_id: str
    image_filename: str
    image_base64: Optional[str] = None
    predicted_class: str
    confidence: float
    class_probabilities: Dict[str, float]
    disease_name: str
    description: str
    risk_level: str
    precautions: List[str]
    recommendation: str
    created_at: datetime


class PredictionListResponse(BaseModel):
    predictions: List[PredictionResult]
    total: int
    page: int
    per_page: int


class StatsResponse(BaseModel):
    total_scans: int
    high_risk_count: int
    low_risk_count: int
    most_common_class: Optional[str] = None
    class_distribution: Dict[str, int]
    monthly_scans: List[Dict]
    recent_scans: List[PredictionResult]
