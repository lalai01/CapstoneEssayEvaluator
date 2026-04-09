FROM python:3.11-slim

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
    mupdf \
    && rm -rf /var/lib/apt/lists/*

ENV TESSERACT_PATH=/usr/bin/tesseract
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}