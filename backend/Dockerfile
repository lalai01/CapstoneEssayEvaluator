FROM python:3.11-slim

# Install Tesseract OCR and required system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    && rm -rf /var/lib/apt/lists/*

# Set environment variable for Tesseract path
ENV TESSERACT_PATH=/usr/bin/tesseract

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the backend code
COPY . .

# Use the PORT environment variable (Render sets this)
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}