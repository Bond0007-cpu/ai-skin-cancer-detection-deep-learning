# Multi-stage Docker build for AI Skin Cancer Detection Backend
FROM python:3.10-slim AS builder

WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime image
FROM python:3.10-slim

WORKDIR /app

# Install OpenCV system dependencies (required for image processing)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local

# Copy application code
COPY app/ ./app/

# Environment Variables
ENV PATH=/root/.local/bin:$PATH
ENV MODEL_PATH=/app/models/exported/efficientnetb4_savedmodel
ENV UPLOAD_DIR=/app/uploads
ENV MAX_IMAGE_SIZE_MB=10

# Create uploads directory
RUN mkdir -p /app/uploads

EXPOSE 8000

# Start FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
