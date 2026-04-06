import os
import re
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import fitz  # PyMuPDF

# Configure Tesseract path for Render (Linux) and local development
# Render Dockerfile installs tesseract at /usr/bin/tesseract
possible_paths = [
    '/usr/bin/tesseract',  # Linux (Render, Ubuntu)
    r'C:\Program Files\Tesseract-OCR\tesseract.exe',  # Windows
    r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',  # Windows (32-bit)
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
    print("   On Render: Make sure Dockerfile installs tesseract-ocr")
    print("   On Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")

def extract_images_from_pdf(pdf_path, dpi=300):
    """Extract images from PDF pages"""
    images = []
    try:
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
    except Exception as e:
        raise Exception(f"PDF processing failed: {str(e)}")

def extract_text_from_pdf_page(pil_image, page_num, preprocessing=True):
    """Extract text from a single PDF page image"""
    try:
        if preprocessing:
            pil_image = preprocess_image_for_ocr(pil_image)
        
        config = '--psm 6 --oem 3'
        text = pytesseract.image_to_string(pil_image, config=config)
        
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        
        return text
    except Exception as e:
        raise Exception(f"OCR failed on page {page_num}: {str(e)}")

def process_pdf_document(pdf_path):
    """Process entire PDF document and extract text from all pages"""
    try:
        page_images = extract_images_from_pdf(pdf_path)
        
        all_text = []
        total_pages = len(page_images)
        
        for i, pil_image in enumerate(page_images):
            page_text = extract_text_from_pdf_page(pil_image, i + 1)
            if page_text:
                all_text.append(f"--- Page {i + 1} ---\n{page_text}")
        
        return "\n\n".join(all_text), total_pages
    except Exception as e:
        raise Exception(f"PDF processing failed: {str(e)}")

def preprocess_image_for_ocr(image):
    """Preprocess image to improve OCR accuracy"""
    if isinstance(image, Image.Image):
        if image.mode != 'L':
            image = image.convert('L')
        
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)
        
        image = image.filter(ImageFilter.SHARPEN)
        
        image = image.point(lambda x: 0 if x < 128 else 255, '1')
    
    return image

def extract_text_from_image(image_path, preprocessing=True):
    """Extract text from image using Tesseract OCR"""
    try:
        image = Image.open(image_path)
        
        if preprocessing:
            image = preprocess_image_for_ocr(image)
        
        config = '--psm 6 --oem 3'
        text = pytesseract.image_to_string(image, config=config)
        
        try:
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            confidences = [int(conf) for conf in data['conf'] if conf != '-1']
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        except:
            avg_confidence = 75
        
        text = re.sub(r'\n+', '\n', text)
        text = re.sub(r' +', ' ', text)
        text = text.strip()
        
        return text, avg_confidence
        
    except Exception as e:
        raise Exception(f"OCR failed: {str(e)}")