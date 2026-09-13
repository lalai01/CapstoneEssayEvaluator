FROM python:3.11-slim

# System dependencies:
# - tesseract-ocr / -eng           → printed-text OCR
# - libtesseract / libleptonica    → build headers for pytesseract
# - libgl1, libglib2.0-0, libsm6,
#   libxext6, libxrender1          → OpenCV runtime
# - libgomp1                       → OpenMP (needed by torch)
# - libx11-6, libxcb1,
#   libxkbcommon-x11-0             → Qt/X libs OpenCV links against
# - mupdf                          → PyMuPDF native lib
# - wget, unzip                    → used during build only
# - default-jre-headless           → required by language-tool-python
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng libtesseract-dev libleptonica-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    libx11-6 libxcb1 libxkbcommon-x11-0 mupdf wget unzip \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV TESSERACT_PATH=/usr/bin/tesseract
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

WORKDIR /app

# Install CPU-only torch FIRST so requirements.txt doesn't pull the GPU wheel.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Download TrOCR weights and save a complete copy (including pooler
# weights) to /opt/trocr. Loading from this local directory avoids the
# "Some weights were newly initialized" warning and the resulting drop
# in per-token confidence.
RUN python -c "from transformers import TrOCRProcessor, VisionEncoderDecoderModel; \
    p = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten'); \
    m = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten'); \
    p.save_pretrained('/opt/trocr'); \
    m.save_pretrained('/opt/trocr'); \
    print('TrOCR saved to /opt/trocr')"

COPY . .
RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]