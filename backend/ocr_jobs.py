import threading
import uuid
import time
from ocr_engines import submit_pdf_to_vision_async, poll_vision_operation, extract_pdf_with_tesseract
from image_quality import recommend_ocr_engine
import fitz
from io import BytesIO

jobs = {}

def start_pdf_ocr_job(pdf_bytes):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'processing', 'result': None, 'error': None}
    
    def worker():
        try:
            # Quick check: sample first page to decide engine
            doc = fitz.open(stream=BytesIO(pdf_bytes), filetype="pdf")
            if len(doc) == 0:
                jobs[job_id]['error'] = "Empty PDF"
                jobs[job_id]['status'] = 'failed'
                return
            pix = doc[0].get_pixmap()
            img_bytes = pix.tobytes("png")
            doc.close()
            engine = recommend_ocr_engine(img_bytes)
            print(f"PDF OCR engine selected: {engine}")
            
            if engine == 'google_vision':
                op_name = submit_pdf_to_vision_async(pdf_bytes)
                result = None
                while result is None:
                    time.sleep(2)
                    result = poll_vision_operation(op_name)
                jobs[job_id]['result'] = result
            else:
                # Use Tesseract in background
                text = extract_pdf_with_tesseract(pdf_bytes)
                jobs[job_id]['result'] = text
            jobs[job_id]['status'] = 'completed'
        except Exception as e:
            jobs[job_id]['error'] = str(e)
            jobs[job_id]['status'] = 'failed'
    
    threading.Thread(target=worker).start()
    return job_id

def get_job_status(job_id):
    return jobs.get(job_id)