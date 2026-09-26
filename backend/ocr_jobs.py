import threading
import uuid
from ocr_utils import extract_text_from_image_bytes, extract_pdf_text_or_route

jobs = {}


def start_pdf_ocr_job(pdf_bytes):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "result": None,
        "error": None,
        "engine": None,
        "confidence": None,
        "current_engine": None,
    }

    def worker():
        try:
            kind, payload = extract_pdf_text_or_route(pdf_bytes)

            if kind == "text":
                jobs[job_id]["result"] = payload
                jobs[job_id]["engine"] = "pymupdf (text layer)"
                jobs[job_id]["confidence"] = 100.0
                jobs[job_id]["status"] = "completed"
                return

            # Scanned PDF — OCR each page
            all_text = []
            used_engines = set()
            confidences = []
            for i, img_bytes in enumerate(payload):
                jobs[job_id]["current_engine"] = f"page {i+1}/{len(payload)}"
                text, conf, engine = extract_text_from_image_bytes(img_bytes)
                used_engines.add(engine)
                confidences.append(conf)
                all_text.append(text)

            jobs[job_id]["result"] = "\n\n".join(all_text)
            jobs[job_id]["engine"] = ", ".join(sorted(used_engines))
            jobs[job_id]["confidence"] = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )
            jobs[job_id]["status"] = "completed"
        except Exception as e:
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["status"] = "failed"
        finally:
            jobs[job_id]["current_engine"] = None

    threading.Thread(target=worker).start()
    return job_id


def get_job_status(job_id):
    return jobs.get(job_id)