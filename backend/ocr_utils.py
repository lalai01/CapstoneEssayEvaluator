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


def extract_text_from_image_bytes(image_bytes):
    import pytesseract
    from PIL import Image

    try:
        messiness = assess_handwriting_messiness(image_bytes)
        print(f"[OCR] messiness = {messiness:.3f}")

        if messiness > 0.2:
            try:
                from handwriting_ocr import ocr_handwriting_image
                text, conf = ocr_handwriting_image(image_bytes)
                print(f"[OCR] TrOCR returned {len(text)} chars, conf={conf:.1f}")
                if text and len(text.strip()) >= 30:
                    return text, conf, "trocr-handwriting"
                print("[OCR] TrOCR output too short; falling back to Tesseract")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[OCR] TrOCR failed, falling back to Tesseract: {e}")

        pil = Image.open(BytesIO(image_bytes))
        text, conf = tesseract_text_and_confidence(pil)
        if not text.strip():
            return "", 0.0, "tesseract"
        return text, conf, "tesseract"
    except Exception as e:
        print(f"extract_text_from_image_bytes failed: {e}")
        return "", 0.0, "tesseract"


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