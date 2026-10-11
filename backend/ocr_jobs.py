import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from ocr_utils import extract_text_from_image_bytes, split_pdf_pages

# Tesseract spawns OpenMP threads per process; with several pages OCR'd in
# parallel that oversubscribes the CPU and is slower than one thread each.
os.environ.setdefault("OMP_THREAD_LIMIT", "1")

# Scanned pages OCR'd concurrently. Tesseract pages run truly in parallel
# (separate processes); TrOCR pages share one model behind a lock in
# handwriting_ocr, so they queue but overlap with Tesseract work.
PDF_OCR_WORKERS = int(os.environ.get("PDF_OCR_WORKERS", min(4, os.cpu_count() or 2)))

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
            pages = split_pdf_pages(pdf_bytes)
            image_idx = [i for i, (kind, _) in enumerate(pages) if kind == "image"]
            total = len(image_idx)

            texts = [payload if kind == "text" else "" for kind, payload in pages]
            used_engines = {"pymupdf (text layer)"} if total < len(pages) else set()
            # Text-layer pages are exact, so they count as 100% confidence.
            confidences = [100.0] * (len(pages) - total)

            if total:
                done = 0
                lock = threading.Lock()
                jobs[job_id]["current_engine"] = f"OCR 0/{total} pages"

                def ocr_page(i):
                    nonlocal done
                    text, conf, engine = extract_text_from_image_bytes(pages[i][1])
                    with lock:
                        done += 1
                        jobs[job_id]["current_engine"] = f"OCR {done}/{total} pages"
                    return i, text, conf, engine

                with ThreadPoolExecutor(max_workers=max(1, min(PDF_OCR_WORKERS, total))) as pool:
                    for i, text, conf, engine in pool.map(ocr_page, image_idx):
                        texts[i] = text
                        used_engines.add(engine)
                        confidences.append(conf)

            jobs[job_id]["result"] = "\n\n".join(t for t in texts if t and t.strip())
            jobs[job_id]["engine"] = ", ".join(sorted(used_engines)) or "none"
            jobs[job_id]["confidence"] = (
                sum(confidences) / len(confidences) if confidences else 0.0
            )
            jobs[job_id]["status"] = "completed"
        except Exception as e:
            jobs[job_id]["error"] = str(e)
            jobs[job_id]["status"] = "failed"
        finally:
            jobs[job_id]["current_engine"] = None

    threading.Thread(target=worker, daemon=True).start()
    return job_id


def get_job_status(job_id):
    return jobs.get(job_id)
