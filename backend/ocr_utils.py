import os
import re
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import fitz
from google.cloud import vision
from google.oauth2 import service_account
import easyocr

# Tesseract path detection
possible_paths = [
    '/usr/bin/tesseract',
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
]
for path in possible_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        break

# Google Vision client
_vision_client = None
def get_vision_client():
    global _vision_client
    if _vision_client is None:
        creds_path = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        if creds_path and os.path.exists(creds_path):
            credentials = service_account.Credentials.from_service_account_file(creds_path)
            _vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        else:
            _vision_client = vision.ImageAnnotatorClient()
    return _vision_client

# EasyOCR reader
_easyocr_reader = None
def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)
    return _easyocr_reader

def clean_ocr_text(text: str) -> str:
    text = re.sub(r'--- Page \d+ ---\s*\n?', '', text)
    lines = text.split('\n')
    cleaned = [line for line in lines if len(re.findall(r'[A-Za-z]{3,}', line)) >= 3]
    return '\n'.join(cleaned)

# ---------- Google Vision (single image) ----------
def extract_with_google_vision(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    upscaled = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, encoded = cv2.imencode('.png', upscaled)
    processed_bytes = encoded.tobytes()
    
    client = get_vision_client()
    image = vision.Image(content=processed_bytes)
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise Exception(f'Vision API error: {response.error.message}')
    annotation = response.full_text_annotation
    text = clean_ocr_text(annotation.text if annotation else "")
    
    confidence = 0.0
    if annotation.pages:
        total_conf, total_symbols = 0, 0
        for page in annotation.pages:
            for block in page.blocks:
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        for symbol in word.symbols:
                            if symbol.confidence:
                                total_conf += symbol.confidence
                                total_symbols += 1
        if total_symbols > 0:
            confidence = (total_conf / total_symbols) * 100
    return text, confidence, 'google_vision'

# ---------- Tesseract (single image) ----------
def extract_with_tesseract(image_bytes, preprocessing=True):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if preprocessing:
        if pil_img.mode != 'L':
            pil_img = pil_img.convert('L')
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(2.0)
        pil_img = pil_img.filter(ImageFilter.SHARPEN)
        pil_img = pil_img.point(lambda x: 0 if x < 128 else 255, '1')
    config = '--psm 6 --oem 3 -l eng'
    text = pytesseract.image_to_string(pil_img, config=config)
    text = clean_ocr_text(text)
    try:
        data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    except:
        avg_confidence = 75
    return text, avg_confidence, 'tesseract'

# ---------- EasyOCR (single image) ----------
def extract_with_easyocr(image_bytes):
    reader = get_easyocr_reader()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb)
    if not results:
        return "", 0.0, 'easyocr'
    text_parts = []
    confidences = []
    for (bbox, text, conf) in results:
        text_parts.append(text)
        confidences.append(conf)
    full_text = " ".join(text_parts)
    avg_conf = (sum(confidences) / len(confidences)) * 100
    full_text = clean_ocr_text(full_text)
    return full_text, avg_conf, 'easyocr'

# ---------- Image dispatcher ----------
def extract_text_from_image(image_path, preprocessing=True):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    from image_quality import recommend_ocr_engine
    engine = recommend_ocr_engine(image_bytes)
    print(f"Selected OCR engine: {engine}")
    if engine == 'google_vision':
        return extract_with_google_vision(image_bytes)
    elif engine == 'easyocr':
        return extract_with_easyocr(image_bytes)
    else:
        return extract_with_tesseract(image_bytes, preprocessing)