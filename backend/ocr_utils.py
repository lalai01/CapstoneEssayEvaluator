import cv2
import numpy as np
from io import BytesIO


def get_priority_engines():
    """
    Informational helper. Actual routing happens inside
    extract_text_from_image_bytes.
    """
    engines = []
    try:
        import pytesseract  # noqa: F401
        engines.append("tesseract")
    except ImportError:
        pass
    try:
        from transformers import TrOCRProcessor  # noqa: F401
        engines.append("trocr")
    except ImportError:
        pass
    return engines


# ---------------------------------------------------------------------------
# Image quality helpers
# ---------------------------------------------------------------------------
def estimate_sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def estimate_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)


def detect_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        return 0.0
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = theta * 180 / np.pi - 90
        angles.append(angle)
    median_angle = np.median(angles)
    return median_angle if abs(median_angle) < 45 else 0.0


def is_handwritten(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
    return 0.08 < edge_density < 0.45


def assess_handwriting_messiness(image_bytes):
    """
    Return a 0..1 score estimating how 'messy' the writing is.
    Higher = more likely handwriting / harder for a printed-text engine.
    Returns 0.0 on any error.
    """
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return 0.0

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        h, w = edges.shape
        block_h, block_w = h // 4, w // 4

        densities = []
        for i in range(4):
            for j in range(4):
                block = edges[i * block_h:(i + 1) * block_h,
                              j * block_w:(j + 1) * block_w]
                if block.size > 0:
                    densities.append(np.sum(block > 0) / block.size)

        edge_variance = float(np.var(densities)) if densities else 0.0

        _, thresh = cv2.threshold(
            gray, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        small = sum(1 for c in contours if cv2.contourArea(c) < 50)
        total = len(contours)
        broken_ratio = small / max(1, total)

        messiness = (edge_variance * 2 + broken_ratio * 1.5) / 3.5
        return float(min(1.0, messiness))
    except Exception as e:
        print(f"assess_handwriting_messiness failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Unified OCR entry points
# ---------------------------------------------------------------------------
def extract_text_from_image(image_path):
    """Path-based wrapper for backward compatibility."""
    with open(image_path, "rb") as f:
        return extract_text_from_image_bytes(f.read())


# A page Tesseract reads this cleanly is printed text: skip the slow
# handwriting model.
PRINTED_MIN_RATIO = 0.9
PRINTED_MIN_CONFIDENCE = 75
PRINTED_MIN_WORDS = 20
# When both readers return text, prefer the handwriting model unless
# Tesseract reads clearly more real words.
HANDWRITING_PREFERENCE = 0.05


def read_essay_image(image_bytes, check_page=True):
    """
    Read an English essay from an image (handwritten or printed).

    1. Reject images that aren't a page of writing (a face, a car...).
    2. Tesseract first: if it reads clean English with high confidence, the
       page is printed and the handwriting model is skipped.
    3. Otherwise read it with Gemini vision when GEMINI_API_KEY is set: it
       reads whole handwritten pages far better than the local model.
    4. Without Gemini (no key, offline, out of quota), run the handwriting
       model (TrOCR) and keep whichever result has more real English words.
    5. Reject results that don't read as English.

    Returns {"text", "confidence", "engine", "quality", "is_text", "reason"}.
    """
    from PIL import Image
    from text_quality import english_quality, looks_like_english_text

    not_text = lambda reason: {"text": "", "confidence": 0.0, "engine": "none",  # noqa: E731
                               "quality": 0.0, "is_text": False, "reason": reason}

    if check_page:
        from essay_image_check import looks_like_page
        is_page, metrics, reason = looks_like_page(image_bytes)
        print(f"[OCR] page check: {is_page} {metrics}")
        if not is_page:
            return not_text(reason)

    candidates = []
    try:
        pil = Image.open(BytesIO(image_bytes))
        text, conf = tesseract_text_and_confidence(pil)
        q = english_quality(text)
        candidates.append({"text": text, "confidence": conf, "engine": "tesseract", "q": q})
        print(f"[OCR] tesseract: {q['words']} words, {q['ratio']:.0%} English, conf {conf:.0f}")
        if (q["ratio"] >= PRINTED_MIN_RATIO and conf >= PRINTED_MIN_CONFIDENCE
                and q["words"] >= PRINTED_MIN_WORDS):
            print("[OCR] printed page: skipping the handwriting model")
            return {"text": text, "confidence": conf, "engine": "tesseract",
                    "quality": q["ratio"], "is_text": True, "reason": ""}
    except Exception as e:
        print(f"[OCR] tesseract failed: {e}")

    import gemini_ocr
    if gemini_ocr.enabled():
        try:
            is_text, text, model = gemini_ocr.read_page(image_bytes)
            print(f"[OCR] gemini ({model}): {len(text.split())} words, is_text={is_text}")
            # A page that passed the page check but that Gemini calls "no
            # text" still gets the local readers: a wrong "no" would reject
            # a real essay.
            if is_text and looks_like_english_text(text):
                # Gemini gives no confidence score; lower it for each word it
                # marked as unreadable so the UI flags pages worth checking.
                conf = max(50.0, 95.0 - 5.0 * text.lower().count("[illegible]"))
                return {"text": text, "confidence": conf, "engine": "gemini-vision",
                        "quality": english_quality(text)["ratio"], "is_text": True, "reason": ""}
            print("[OCR] gemini text doesn't read as English; trying the local readers")
        except Exception as e:
            print(f"[OCR] gemini unavailable, using the local readers: {e}")

    try:
        from handwriting_ocr import ocr_handwriting_image
        text, conf = ocr_handwriting_image(image_bytes)
        q = english_quality(text)
        candidates.append({"text": text, "confidence": conf, "engine": "trocr-handwriting", "q": q})
        print(f"[OCR] handwriting: {q['words']} words, {q['ratio']:.0%} English, conf {conf:.0f}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[OCR] handwriting model failed: {e}")

    usable = [c for c in candidates if c["q"]["words"] > 0]
    if not usable:
        return not_text("No readable text was found in the image.")

    def score(c):
        bonus = HANDWRITING_PREFERENCE if c["engine"] != "tesseract" else 0.0
        return c["q"]["ratio"] + bonus
    best = max(usable, key=score)

    if not looks_like_english_text(best["text"]):
        return not_text("The image doesn't contain readable English paragraphs.")
    return {"text": best["text"], "confidence": best["confidence"], "engine": best["engine"],
            "quality": best["q"]["ratio"], "is_text": True, "reason": ""}


def extract_text_from_image_bytes(image_bytes):
    """(text, confidence, engine) for one page; empty text if it isn't an essay page."""
    try:
        r = read_essay_image(image_bytes)
        return r["text"], r["confidence"], r["engine"]
    except Exception as e:
        print(f"extract_text_from_image_bytes failed: {e}")
        return "", 0.0, "none"


# English only: one language model keeps Tesseract fast and avoids
# misreads from other scripts.
TESSERACT_CONFIG = "-l eng --oem 1 --psm 6"


def tesseract_text_and_confidence(pil_image):
    """
    One Tesseract pass that yields both the text and the mean word
    confidence (image_to_string + image_to_data would OCR the page twice).
    Lines are rebuilt from Tesseract's block/paragraph/line numbering.
    """
    import pytesseract

    data = pytesseract.image_to_data(
        pil_image, config=TESSERACT_CONFIG, output_type=pytesseract.Output.DICT
    )

    lines = []
    current_key = None
    current_words = []
    last_par = None
    conf_values = []
    for i, word in enumerate(data["text"]):
        if not word or not word.strip():
            continue
        try:
            c = float(data["conf"][i])
        except (TypeError, ValueError):
            c = -1
        if c >= 0:
            conf_values.append(c)

        par = (data["block_num"][i], data["par_num"][i])
        key = par + (data["line_num"][i],)
        if key != current_key:
            if current_words:
                lines.append(" ".join(current_words))
            if last_par is not None and par != last_par:
                lines.append("")  # paragraph break
            current_key, current_words, last_par = key, [], par
        current_words.append(word.strip())
    if current_words:
        lines.append(" ".join(current_words))

    conf = sum(conf_values) / len(conf_values) if conf_values else 0.0
    return "\n".join(lines), conf


def extract_text_from_image(image_path):
    """Path-based wrapper for backward compatibility."""
    with open(image_path, "rb") as f:
        return extract_text_from_image_bytes(f.read())


# ---------------------------------------------------------------------------
# Other document formats
# ---------------------------------------------------------------------------
def extract_docx_text(docx_bytes):
    """Read text directly from a .docx file. No OCR involved."""
    from docx import Document
    doc = Document(BytesIO(docx_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


# Pages with at least this many characters in their text layer are read
# directly; anything less is treated as a scanned page and OCR'd.
MIN_TEXT_LAYER_CHARS = 50
PDF_RENDER_DPI = 200


def split_pdf_pages(pdf_bytes):
    """
    Decide per page whether OCR is needed, so a PDF that mixes typed and
    scanned pages only OCRs the scanned ones.

    Returns a list in page order of:
      ("text",  page_text)   -> page has a usable text layer (instant)
      ("image", png_bytes)   -> scanned/handwritten page to OCR
    Blank pages are skipped.
    """
    import fitz  # PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    try:
        for page in doc:
            text = page.get_text()
            if len(text.strip()) >= MIN_TEXT_LAYER_CHARS:
                pages.append(("text", text.strip()))
                continue
            # Grayscale render: a third of the data of RGB and all the OCR
            # engines convert to grayscale anyway.
            pix = page.get_pixmap(dpi=PDF_RENDER_DPI, colorspace=fitz.csGRAY)
            if np.frombuffer(pix.samples, np.uint8).min(initial=255) > 245:
                continue  # blank page
            pages.append(("image", pix.tobytes("png")))
    finally:
        doc.close()
    return pages