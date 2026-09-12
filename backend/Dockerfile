FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr tesseract-ocr-eng libtesseract-dev libleptonica-dev \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 libgomp1 \
    libx11-6 libxcb1 libxkbcommon-x11-0 mupdf wget \
    default-jre-headless unzip \
    && rm -rf /var/lib/apt/lists/*

ENV TESSERACT_PATH=/usr/bin/tesseract
ENV TESSDATA_PREFIX=/usr/share/tesseract-ocr/4.00/tessdata

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /root/.cache/language_tool_python && \
    wget -q -O /tmp/LanguageTool.zip https://languagetool.org/download/LanguageTool-stable.zip && \
    unzip -q /tmp/LanguageTool.zip -d /root/.cache/language_tool_python/ && \
    rm /tmp/LanguageTool.zip && \
    mv /root/.cache/language_tool_python/LanguageTool-*/* /root/.cache/language_tool_python/ && \
    rmdir /root/.cache/language_tool_python/LanguageTool-*

COPY . .
RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
