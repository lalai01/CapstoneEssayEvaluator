import threading
import uuid
from io import BytesIO
import fitz
from PIL import Image
import cv2
import numpy as np
from image_quality import get_priority_engines

jobs = {}

def extract_pdf_with_engine(pdf_bytes, engine, dpi=150):
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    all_text = []
    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap(dpi=dpi)
        img_bytes = pix.tobytes("png")
        if engine == 'google_vision':
            from ocr_utils import extract_with_google_vision
            text, _, _ = extract_with_google_vision(img_bytes)
        elif engine == 'paddleocr':
            from ocr_utils import extract_with_paddleocr
            text, _, _ = extract_with_paddleocr(img_bytes)
        elif engine == 'easyocr':
            from ocr_utils import extract_with_easyocr
            text, _, _ = extract_with_easyocr(img_bytes)
        elif engine == 'tesseract':
            from ocr_utils import extract_with_tesseract
            text, _, _ = extract_with_tesseract(img_bytes)
        else:
            text = ""
        all_text.append(text)
    doc.close()
    return "\n\n".join(all_text)

def start_pdf_ocr_job(pdf_bytes):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'processing', 'result': None, 'error': None, 'engine': None, 'current_engine': None}
    def worker():
        try:
            engines = get_priority_engines()
            text = None
            used_engine = None
            for engine in engines:
                jobs[job_id]['current_engine'] = engine
                try:
                    text = extract_pdf_with_engine(pdf_bytes, engine)
                    if text and len(text.strip()) > 20:
                        used_engine = engine
                        print(f"PDF OCR succeeded with {engine}")
                        break
                except Exception as e:
                    print(f"PDF OCR with {engine} failed: {e}")
                    continue
            if not text or not text.strip():
                jobs[job_id]['error'] = "All OCR engines (including Tesseract) failed. Please check the PDF quality."
                jobs[job_id]['status'] = 'failed'
            else:
                jobs[job_id]['result'] = text
                jobs[job_id]['engine'] = used_engine
                jobs[job_id]['status'] = 'completed'
            jobs[job_id]['current_engine'] = None
        except Exception as e:
            jobs[job_id]['error'] = str(e)
            jobs[job_id]['status'] = 'failed'
    threading.Thread(target=worker).start()
    return job_id

def get_job_status(job_id):
    return jobs.get(job_id)