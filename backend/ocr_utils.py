import os
import re
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
import fitz
from google.cloud import vision
from google.oauth2 import service_account

possible_paths = ['/usr/bin/tesseract', r'C:\Program Files\Tesseract-OCR\tesseract.exe']
for p in possible_paths:
    if os.path.exists(p):
        pytesseract.pytesseract.tesseract_cmd = p
        break

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

_easyocr_reader = None
def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        try:
            import easyocr
            _easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        except:
            _easyocr_reader = None
    return _easyocr_reader

_paddle_ocr = None
def get_paddle_ocr():
    global _paddle_ocr
    if _paddle_ocr is None:
        try:
            from paddleocr import PaddleOCR
            _paddle_ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        except:
            _paddle_ocr = None
    return _paddle_ocr

def clean_ocr_text(text):
    text = re.sub(r'--- Page \d+ ---\s*\n?', '', text)
    lines = text.split('\n')
    cleaned = [l for l in lines if len(re.findall(r'[A-Za-z]{3,}', l)) >= 3]
    return '\n'.join(cleaned)

def extract_with_google_vision(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    upscaled = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    _, encoded = cv2.imencode('.png', upscaled)
    client = get_vision_client()
    image = vision.Image(content=encoded.tobytes())
    response = client.document_text_detection(image=image)
    if response.error.message:
        raise Exception(f'Vision API error: {response.error.message}')
    annotation = response.full_text_annotation
    text = clean_ocr_text(annotation.text if annotation else "")
    # Confidence calculation simplified
    conf = 90.0 if text else 0.0
    return text, conf, 'google_vision'

def extract_with_paddleocr(image_bytes):
    ocr = get_paddle_ocr()
    if ocr is None:
        raise Exception("PaddleOCR not available")
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    result = ocr.ocr(img, cls=True)
    if not result or not result[0]:
        return "", 0.0, 'paddleocr'
    texts = [line[1][0] for line in result[0]]
    confs = [line[1][1] for line in result[0]]
    full_text = " ".join(texts)
    avg_conf = (sum(confs) / len(confs)) * 100
    full_text = clean_ocr_text(full_text)
    return full_text, avg_conf, 'paddleocr'

def extract_with_easyocr(image_bytes):
    reader = get_easyocr_reader()
    if reader is None:
        raise Exception("EasyOCR not available")
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = reader.readtext(img_rgb)
    if not results:
        return "", 0.0, 'easyocr'
    texts = [text for (_, text, _) in results]
    confs = [conf for (_, _, conf) in results]
    full_text = " ".join(texts)
    avg_conf = (sum(confs) / len(confs)) * 100
    full_text = clean_ocr_text(full_text)
    return full_text, avg_conf, 'easyocr'

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
        confs = [int(c) for c in data['conf'] if c != '-1']
        avg_conf = sum(confs) / len(confs) if confs else 75
    except:
        avg_conf = 75
    return text, avg_conf, 'tesseract'

def extract_text_from_image(image_path, preprocessing=True):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
    from image_quality import get_priority_engines
    engines = get_priority_engines()
    for engine in engines:
        try:
            if engine == 'google_vision':
                text, conf, eng = extract_with_google_vision(image_bytes)
            elif engine == 'paddleocr':
                text, conf, eng = extract_with_paddleocr(image_bytes)
            elif engine == 'easyocr':
                text, conf, eng = extract_with_easyocr(image_bytes)
            else:
                text, conf, eng = extract_with_tesseract(image_bytes, preprocessing)
            if text and len(text.strip()) > 10:
                print(f"OCR succeeded with {engine}")
                return text, conf, eng
        except Exception as e:
            print(f"{engine} failed: {e}")
            continue
    return "", 0.0, "none"

def extract_text_from_pdf(pdf_path):
    from ocr_jobs import start_pdf_ocr_job
    with open(pdf_path, 'rb') as f:
        pdf_bytes = f.read()
    return start_pdf_ocr_job(pdf_bytes)