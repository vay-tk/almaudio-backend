# Use Python 3.10 as base image
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
    # Additional dependencies for pycld2 - use libpcre2-dev instead of libpcre3-dev
    libpcre2-dev \
    g++ \
    # Required for other packages
    libsndfile-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Create two sets of requirements: core and optional
RUN grep -v -E "pycld2|polyglot|pyannote-audio" requirements.txt > core_requirements.txt

# Install core packages first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r core_requirements.txt

# Install optional packages with fallbacks
RUN pip install --no-cache-dir pycld2 || echo "pycld2 installation failed, continuing anyway" && \
    pip install --no-cache-dir polyglot || echo "polyglot installation failed, continuing anyway" && \
    pip install --no-cache-dir pyannote-audio || echo "pyannote-audio installation failed, continuing anyway"

# Copy application code
COPY . .

# Create directories that might not exist
RUN mkdir -p /app/routes

# Create a health check endpoint file
RUN echo 'from fastapi import APIRouter\n\nrouter = APIRouter()\n\n@router.get("/")\nasync def health_check():\n    return {"status": "healthy"}' > /app/routes/health.py

# Create a fallback for language detection if pycld2 fails
RUN echo 'def detect(text): return True, None, [("en", "ENGLISH", 100, 0.0)]' > /app/cld2_fallback.py

# Set environment variables for Railway
ENV HOST=0.0.0.0
ENV PORT=8000

# Expose the port the app will run on
EXPOSE ${PORT}

# Create a startup script to handle initialization
RUN echo '#!/bin/bash\n\
echo "Starting AI Audio Analyzer..."\n\
python -m pip list\n\
echo "Checking for language detection support:"\n\
python -c "import langdetect; print(f\"langdetect is available\")" || echo "langdetect not available"\n\
python -c "try:\n\
    import pycld2\n\
    print(\"pycld2 is available\")\n\
except ImportError:\n\
    print(\"pycld2 not available\")\n"\n\
exec uvicorn main:app --host ${HOST} --port ${PORT}' > /app/start.sh && \
chmod +x /app/start.sh

# Command to run the application using the startup script
CMD ["/app/start.sh"]
