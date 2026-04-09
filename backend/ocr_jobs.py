import threading
import uuid
import time
from io import BytesIO
import fitz
from PIL import Image
import pytesseract

jobs = {}

def extract_pdf_with_tesseract(pdf_bytes, dpi=150):
    """Lightweight Tesseract PDF processing (fallback)"""
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    all_text = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        # Minimal preprocessing to save memory
        img = img.convert('L')
        text = pytesseract.image_to_string(img, config='--psm 6')
        all_text.append(text)
    doc.close()
    return "\n\n".join(all_text)

def extract_pdf_with_easyocr(pdf_bytes, dpi=150):
    """Try EasyOCR on PDF (convert each page to image)"""
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
        all_text = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            # Convert to numpy for EasyOCR
            import numpy as np
            img_np = np.array(img)
            results = reader.readtext(img_np)
            page_text = " ".join([text for (_, text, _) in results])
            all_text.append(page_text)
        doc.close()
        return "\n\n".join(all_text)
    except Exception as e:
        print(f"EasyOCR PDF failed: {e}, falling back to Tesseract")
        return extract_pdf_with_tesseract(pdf_bytes)

def start_pdf_ocr_job(pdf_bytes):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'processing', 'result': None, 'error': None}
    
    def worker():
        try:
            # Try EasyOCR first, fallback to Tesseract
            text = extract_pdf_with_easyocr(pdf_bytes)
            if not text.strip():
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