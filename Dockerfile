# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV BUILD_ONLY_PACKAGES="pycld2 polyglot pyannote-audio"

# 10 minute timeout for pip
ENV PIP_DEFAULT_TIMEOUT=600

# Install system dependencies required for audio processing and language detection
# Add timeout to prevent hanging during installation
RUN apt-get update && \
    timeout 600 apt-get install -y --no-install-recommends \
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

# Install core packages first with timeout
RUN timeout 600 pip install --no-cache-dir --upgrade pip && \
    timeout 600 pip install --no-cache-dir -r core_requirements.txt

# Install optional packages with fallbacks and timeouts
RUN timeout 300 pip install --no-cache-dir pycld2 || echo "pycld2 installation failed, continuing anyway" && \
    timeout 300 pip install --no-cache-dir polyglot || echo "polyglot installation failed, continuing anyway" && \
    timeout 300 pip install --no-cache-dir pyannote-audio || echo "pyannote-audio installation failed, continuing anyway"

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

# Create a startup script with timeout handling
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
echo "Starting web server..."\n\
uvicorn main:app --host ${HOST} --port ${PORT}' > /app/start.sh && \
chmod +x /app/start.sh

# Health check config
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Command to run the application using the startup script
CMD ["/app/start.sh"]
