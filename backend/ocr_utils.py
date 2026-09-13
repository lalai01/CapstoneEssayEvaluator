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
    """
    Unified OCR entry point. Takes raw image bytes and returns
    (text, confidence, engine).

    Routes:
      - handwriting (messiness > 0.2) -> TrOCR
      - printed                       -> Tesseract (--oem 1 --psm 6)
    """
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
        text = pytesseract.image_to_string(pil, config="--oem 1 --psm 6")

        if not text or not text.strip():
            return "", 0.0, "tesseract"

        data = pytesseract.image_to_data(
            pil, output_type=pytesseract.Output.DICT
        )
        conf_values = []
        for c, t in zip(data["conf"], data.get("text", [])):
            try:
                v = float(c)
            except (TypeError, ValueError):
                continue
            if v >= 0 and t and t.strip():
                conf_values.append(v)

        conf = sum(conf_values) / len(conf_values) if conf_values else 0.0
        return text, conf, "tesseract"
    except Exception as e:
        print(f"extract_text_from_image_bytes failed: {e}")
        return "", 0.0, "tesseract"


# ---------------------------------------------------------------------------
# Other document formats
# ---------------------------------------------------------------------------
def extract_docx_text(docx_bytes):
    """Read text directly from a .docx file. No OCR involved."""
    from docx import Document
    doc = Document(BytesIO(docx_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_pdf_text_or_route(pdf_bytes):
    """
    Returns one of:
      ("text",   full_text)                 -> PDF had a text layer
      ("images", [page_png_bytes, ...])     -> scanned PDF, pages to OCR
    """
    import fitz  # PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    total_text = ""
    for page in doc:
        total_text += page.get_text()

    if len(total_text.strip()) > 100:
        doc.close()
        return "text", total_text

    page_images = []
    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap(dpi=200)
        page_images.append(pix.tobytes("png"))
    doc.close()
    return "images", page_images