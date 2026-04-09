FROM python:3.11-slim

# Install system dependencies for:
# - Tesseract OCR
# - OpenCV (libgl1, libglib2.0, etc.)
# - EasyOCR (libffi, libssl, etc.)
# - PyMuPDF (mupdf)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libffi-dev \
    libssl-dev \
    mupdf \
    mupdf-tools \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV TESSERACT_PATH=/usr/bin/tesseract
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
# Install PyTorch CPU-only (lightweight for EasyOCR)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Expose port (Render sets PORT env variable)
EXPOSE 8000

# Run the app
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}