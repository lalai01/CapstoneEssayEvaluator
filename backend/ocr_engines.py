import re
import cv2
import numpy as np
import easyocr
from PIL import Image
import pytesseract

# Lazy initialization
_easyocr_reader = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)
    return _easyocr_reader

def preprocess_image_bytes(image_bytes):
    """Convert bytes to grayscale, denoise, threshold, upscale."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    # Adaptive threshold
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # Upscale
    upscaled = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return upscaled

def extract_with_tesseract(image_bytes):
    """Tesseract OCR."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # Use config for English and good segmentation
    config = '--psm 6 --oem 3 -l eng'
    text = pytesseract.image_to_string(pil_img, config=config)
    text = re.sub(r'\s+', ' ', text).strip()
    # Dummy confidence
    confidence = 80.0
    return text, confidence

def extract_with_easyocr(image_bytes):
    """EasyOCR."""
    reader = get_easyocr_reader()
    img = preprocess_image_bytes(image_bytes)
    result = reader.readtext(img, detail=0, paragraph=True)
    text = ' '.join(result)
    text = re.sub(r'\s+', ' ', text).strip()
    confidence = 85.0
    return text, confidence

# PaddleOCR is not included due to size; you can add it similarly.