import os
import re
import tempfile
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import fitz  # PyMuPDF
from google.cloud import vision
from google.oauth2 import service_account

# ---------- Tesseract path detection ----------
possible_paths = [
    '/usr/bin/tesseract',                     # Linux (Render, Ubuntu)
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
]

tesseract_found = False
for path in possible_paths:
    if os.path.exists(path):
        pytesseract.pytesseract.tesseract_cmd = path
        tesseract_found = True
        print(f"✅ Tesseract found at: {path}")
        break

if not tesseract_found:
    print("⚠️ WARNING: Tesseract OCR not found. OCR features will not work.")

# ---------- Google Vision client (lazy init) ----------
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

# ---------- Helper: clean OCR text ----------
def clean_ocr_text(text: str) -> str:
    """Remove page markers and lines with very few English words."""
    text = re.sub(r'--- Page \d+ ---\s*\n?', '', text)
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        words = re.findall(r'[A-Za-z]{3,}', line)
        if len(words) >= 3:
            cleaned.append(line)
    return '\n'.join(cleaned)

# ---------- Google Vision extraction (with OpenCV preprocessing) ----------
def extract_with_google_vision(image_bytes):
    """Preprocess image and call Google Vision API."""
    # Decode with OpenCV
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Denoise
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    # Otsu threshold
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Upscale
    upscaled = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, encoded = cv2.imencode('.png', upscaled)
    processed_bytes = encoded.tobytes()

    client = get_vision_client()
    image = vision.Image(content=processed_bytes)
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise Exception(f'Vision API error: {response.error.message}')
    annotation = response.full_text_annotation
    text = annotation.text if annotation else ""
    text = clean_ocr_text(text)

    # Compute average confidence
    confidence = 0.0
    if annotation.pages:
        total_conf = 0
        total_symbols = 0
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
    return text, confidence

# ---------- Tesseract extraction ----------
def extract_with_tesseract(image_bytes, preprocessing=True):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if preprocessing:
        pil_img = preprocess_image_for_ocr(pil_img)
    config = '--psm 6 --oem 3 -l eng'
    text = pytesseract.image_to_string(pil_img, config=config)
    text = clean_ocr_text(text)
    # Rough confidence
    try:
        data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    except:
        avg_confidence = 75
    return text, avg_confidence

# ---------- Image preprocessing for Tesseract ----------
def preprocess_image_for_ocr(image):
    if isinstance(image, Image.Image):
        if image.mode != 'L':
            image = image.convert('L')
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        image = image.filter(ImageFilter.SHARPEN)
        image = image.point(lambda x: 0 if x < 128 else 255, '1')
    return image

# ---------- PDF processing with first-page quality check ----------
def pdf_needs_google_vision(pdf_path):
    """Check first page quality and decide engine."""
    doc = fitz.open(pdf_path)
    if len(doc) == 0:
        return False
    pix = doc[0].get_pixmap()
    img_bytes = pix.tobytes("png")
    doc.close()
    from image_quality import recommend_ocr_engine
    engine = recommend_ocr_engine(img_bytes)
    return engine == 'google_vision'

def extract_images_from_pdf(pdf_path, dpi=300):
    """Extract images from PDF pages for Tesseract processing."""
    images = []
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("ppm")
        pil_image = Image.frombytes("RGB", [pix.width, pix.height], img_data)
        images.append(pil_image)
    pdf_document.close()
    return images

def extract_text_from_pdf_page(pil_image, page_num, preprocessing=True):
    if preprocessing:
        pil_image = preprocess_image_for_ocr(pil_image)
    config = '--psm 6 --oem 3 -l eng'
    text = pytesseract.image_to_string(pil_image, config=config)
    text = clean_ocr_text(text)
    return text

def process_pdf_document(pdf_path):
    """Synchronous Tesseract OCR for PDFs."""
    page_images = extract_images_from_pdf(pdf_path)
    all_text = []
    for i, pil_image in enumerate(page_images):
        page_text = extract_text_from_pdf_page(pil_image, i + 1)
        if page_text:
            all_text.append(page_text)
    return "\n\n".join(all_text), len(page_images)

def extract_text_from_pdf(pdf_path):
    """
    Decide engine based on first page quality.
    Returns:
      - If Google Vision needed: job_id (string)
      - If Tesseract: tuple (text, page_count)
    """
    if pdf_needs_google_vision(pdf_path):
        # Use async Google Vision batch
        from ocr_jobs import start_pdf_ocr_job
        with open(pdf_path, 'rb') as f:
            pdf_bytes = f.read()
        job_id = start_pdf_ocr_job(pdf_bytes)
        return job_id
    else:
        text, page_count = process_pdf_document(pdf_path)
        return text, page_count

# ---------- Main image dispatcher (auto engine) ----------
def extract_text_from_image(image_path, preprocessing=True):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    from image_quality import recommend_ocr_engine
    engine = recommend_ocr_engine(image_bytes)
    print(f"Selected OCR engine: {engine}")
    if engine == 'google_vision':
        return extract_with_google_vision(image_bytes)
    else:
        return extract_with_tesseract(image_bytes, preprocessing)