import threading
import uuid
from io import BytesIO
import fitz
from PIL import Image
import pytesseract
import cv2
import numpy as np
from image_quality import recommend_ocr_engine, EASYOCR_AVAILABLE

jobs = {}

def extract_pdf_with_tesseract(pdf_bytes, dpi=150):
    doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
    all_text = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = img.convert('L')
        text = pytesseract.image_to_string(img, config='--psm 6')
        all_text.append(text)
    doc.close()
    return "\n\n".join(all_text)

def extract_pdf_with_easyocr(pdf_bytes, dpi=150):
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
        all_text = []
        for page in doc:
            pix = page.get_pixmap(dpi=dpi)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
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
            # Check first page to decide engine (for messy handwriting)
            doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
            if len(doc) == 0:
                jobs[job_id]['error'] = "Empty PDF"
                jobs[job_id]['status'] = 'failed'
                return
            pix = doc[0].get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            doc.close()
            # Convert to bytes for recommend_ocr_engine
            engine = recommend_ocr_engine(img_bytes)
            print(f"PDF OCR engine selected: {engine}")
            
            if engine == 'easyocr' and EASYOCR_AVAILABLE:
                text = extract_pdf_with_easyocr(pdf_bytes)
            else:
                text = extract_pdf_with_tesseract(pdf_bytes)
            
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