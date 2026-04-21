FROM python:3.11-slim

# Install system dependencies including Java and unzip
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    libtesseract-dev \
    libleptonica-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libx11-6 \
    libxcb1 \
    libxkbcommon-x11-0 \
    mupdf \
    wget \
    default-jre-headless \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Set Tesseract environment variables
ENV TESSERACT_PATH=/usr/bin/tesseract
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install PyTorch CPU (required for EasyOCR)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download LanguageTool model
RUN mkdir -p /root/.cache/language_tool_python && \
    wget -q -O /tmp/LanguageTool.zip https://languagetool.org/download/LanguageTool-stable.zip && \
    unzip -q /tmp/LanguageTool.zip -d /root/.cache/language_tool_python/ && \
    rm /tmp/LanguageTool.zip

# Copy the rest of the application
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]