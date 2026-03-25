"""
AI Skin Cancer Detection — FastAPI Backend
Main application entry point.
"""

import os
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import connect_to_mongo, close_mongo_connection
from app.ml.model_service import load_model
from app.auth.routes import router as auth_router
from app.predictions.routes import router as predictions_router
from app.admin.routes import router as admin_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    await connect_to_mongo()
    load_model()
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    print("🚀 AI Skin Cancer Detection API is ready!")
    yield
    # Shutdown
    await close_mongo_connection()


app = FastAPI(
    title="AI Skin Cancer Detection API",
    description="Production-grade dermoscopic image classification API with JWT auth, "
                "prediction history, and admin dashboard.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Static files for uploaded images
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Register routers
app.include_router(auth_router)
app.include_router(predictions_router)
app.include_router(admin_router)


@app.get("/")
async def root():
    return {
        "message": "AI Skin Cancer Detection API v2.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "2.0.0",
        "model": "EfficientNetB4",
        "classes": 7,
    }
