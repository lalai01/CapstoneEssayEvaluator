import os
import re
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import fitz  # PyMuPDF

# Configure Tesseract path (supports Windows and Linux)
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

# ----------------------------------------------------------------------
# Helper: clean OCR text (remove page markers, keep only lines with English words)
# ----------------------------------------------------------------------
def clean_ocr_text(text: str) -> str:
    """Remove page markers and lines that contain very few English words."""
    # Remove any remaining "--- Page X ---" markers
    text = re.sub(r'--- Page \d+ ---\s*\n?', '', text)
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        # Count alphabetic words (at least 3 characters)
        words = re.findall(r'[A-Za-z]{3,}', line)
        if len(words) >= 3:
            cleaned.append(line)
    return '\n'.join(cleaned)

# ----------------------------------------------------------------------
# PDF processing
# ----------------------------------------------------------------------
def extract_images_from_pdf(pdf_path, dpi=300):
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
    # Force English language, page segmentation mode 6 (uniform block)
    config = '--psm 6 --oem 3 -l eng'
    text = pytesseract.image_to_string(pil_image, config=config)
    text = clean_ocr_text(text)
    return text

def process_pdf_document(pdf_path):
    page_images = extract_images_from_pdf(pdf_path)
    all_text = []
    for i, pil_image in enumerate(page_images):
        page_text = extract_text_from_pdf_page(pil_image, i + 1)
        if page_text:
            all_text.append(page_text)   # no page markers
    return "\n\n".join(all_text), len(page_images)

# ----------------------------------------------------------------------
# Image preprocessing
# ----------------------------------------------------------------------
def preprocess_image_for_ocr(image):
    if isinstance(image, Image.Image):
        if image.mode != 'L':
            image = image.convert('L')
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        image = image.filter(ImageFilter.SHARPEN)
        # Adaptive-like threshold
        image = image.point(lambda x: 0 if x < 128 else 255, '1')
    return image

def extract_text_from_image(image_path, preprocessing=True):
    image = Image.open(image_path)
    if preprocessing:
        image = preprocess_image_for_ocr(image)
    config = '--psm 6 --oem 3 -l eng'
    text = pytesseract.image_to_string(image, config=config)
    text = clean_ocr_text(text)
    # Confidence (rough estimate)
    try:
        data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    except:
        avg_confidence = 75
    return text, avg_confidence