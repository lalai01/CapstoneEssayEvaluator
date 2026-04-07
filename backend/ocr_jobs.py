import threading
import uuid
import time
from ocr_engines import submit_pdf_to_vision_async, poll_vision_operation

jobs = {}  # job_id -> {'status': 'pending'/'done', 'result': text, 'error': str}

def start_pdf_ocr_job(pdf_bytes):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {'status': 'processing', 'result': None, 'error': None}
    def worker():
        try:
            op_name = submit_pdf_to_vision_async(pdf_bytes)
            # Poll until done (could be long, we'll do simple loop)
            result = None
            while result is None:
                time.sleep(2)
                result = poll_vision_operation(op_name)
            jobs[job_id]['result'] = result
            jobs[job_id]['status'] = 'completed'
        except Exception as e:
            jobs[job_id]['error'] = str(e)
            jobs[job_id]['status'] = 'failed'
    thread = threading.Thread(target=worker)
    thread.start()
    return job_id

def get_job_status(job_id):
    return jobs.get(job_id)