# Use Python 3.11 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV BUILD_ONLY_PACKAGES="pycld2 polyglot pyannote-audio"

# Install system dependencies required for audio processing and language detection
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libasound2-dev \
    portaudio19-dev \
    build-essential \
    gcc \
    git \
    cmake \
    libicu-dev \
    pkg-config \
    wget \
    unzip \
    # Additional dependencies for pycld2
    libpcre3-dev \
    g++ \
    # Required for other packages
    libsndfile-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install core packages without the problematic ones
RUN grep -v -E "pycld2|polyglot|pyannote-audio" requirements.txt > core_requirements.txt && \
    pip install --no-cache-dir -r core_requirements.txt

# Install problematic packages separately with specific flags
RUN pip install --no-cache-dir pycld2 || echo "pycld2 installation failed, continuing anyway" && \
    pip install --no-cache-dir polyglot || echo "polyglot installation failed, continuing anyway" && \
    pip install --no-cache-dir pyannote-audio || echo "pyannote-audio installation failed, continuing anyway"

# Copy application code
COPY . .

# Create a file to indicate whether optional packages were installed
RUN pip list | grep -E "pycld2|polyglot|pyannote-audio" > /app/installed_optional_packages.txt || echo "No optional packages installed" > /app/installed_optional_packages.txt

# Create a health check endpoint file
RUN echo 'from fastapi import FastAPI, APIRouter\n\nrouter = APIRouter()\n\n@router.get("/")\nasync def health_check():\n    return {"status": "healthy"}' > /app/routes/health.py

# Set environment variables for Railway
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose the port the app will run on
EXPOSE ${PORT}

# Command to run the application
CMD uvicorn main:app --host ${HOST} --port ${PORT}
