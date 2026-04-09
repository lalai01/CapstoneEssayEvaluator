import threading
import uuid
from io import BytesIO
import fitz
from PIL import Image
import cv2
import numpy as np
from image_quality import get_priority_engines
import pytesseract

jobs = {}

def extract_pdf_with_engine(pdf_bytes, engine, dpi=150):
    """Extract PDF using a specific engine on each page."""
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    all_text = []
    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        # Convert to numpy for OCR functions
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Use the same extraction functions as for images
        if engine == 'google_vision':
            from ocr_utils import extract_with_google_vision
            text, _, _ = extract_with_google_vision(img_bytes)
        elif engine == 'paddleocr':
            from ocr_utils import extract_with_paddleocr
            text, _, _ = extract_with_paddleocr(img_bytes)
        elif engine == 'easyocr':
            from ocr_utils import extract_with_easyocr
            text, _, _ = extract_with_easyocr(img_bytes)
        else:
            # Tesseract
            from ocr_utils import extract_with_tesseract
            text, _, _ = extract_with_tesseract(img_bytes)
        all_text.append(text)
    doc.close()
    return "\n\n".join(all_text)

def start_pdf_ocr_job(pdf_bytes):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'processing', 'result': None, 'error': None}
    
    def worker():
        try:
            # Check first page to decide which engine to try first (optional)
            engines = get_priority_engines()
            text = None
            for engine in engines:
                try:
                    text = extract_pdf_with_engine(pdf_bytes, engine)
                    if text and len(text.strip()) > 20:
                        print(f"PDF OCR succeeded with {engine}")
                        break
                except Exception as e:
                    print(f"PDF OCR with {engine} failed: {e}")
                    continue
            if not text or not text.strip():
                text = "[No text could be extracted from this PDF. Please check the document quality or type manually.]"
            jobs[job_id]['result'] = text
            jobs[job_id]['status'] = 'completed'
        except Exception as e:
            jobs[job_id]['error'] = str(e)
            jobs[job_id]['status'] = 'failed'
    
    threading.Thread(target=worker).start()
    return job_id

def get_job_status(job_id):
    return jobs.get(job_id)