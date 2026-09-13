import cv2
import numpy as np
from io import BytesIO


def get_priority_engines():
    """
    Legacy helper kept for compatibility. The routing now happens inside
    extract_text_from_image_bytes, so this is informational only.
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


# ... estimate_sharpness, estimate_contrast, detect_skew, is_handwritten,
#     assess_handwriting_messiness — unchanged, keep as-is ...


def extract_text_from_image(image_path):
    """Path-based wrapper for backward compatibility."""
    with open(image_path, "rb") as f:
        return extract_text_from_image_bytes(f.read())


def extract_text_from_image_bytes(image_bytes):
    """
    Unified OCR entry point. Takes raw image bytes and returns
    (text, confidence, engine).

    Routes:
      - handwriting (messiness > 0.5) → TrOCR
      - printed                       → Tesseract (--oem 1 --psm 6)
    """
    import pytesseract
    from PIL import Image

    try:
        # --- Decide route ---
        messiness = assess_handwriting_messiness(image_bytes)

        if messiness > 0.5:
            try:
                from handwriting_ocr import ocr_handwriting_image
                text, conf = ocr_handwriting_image(image_bytes)
                if text and text.strip():
                    return text, conf, "trocr-handwriting"
                print(f"TrOCR returned empty; falling back to Tesseract")
            except Exception as e:
                print(f"TrOCR failed, falling back to Tesseract: {e}")

        # --- Printed-text path ---
        pil = Image.open(BytesIO(image_bytes))
        text = pytesseract.image_to_string(pil, config="--oem 1 --psm 6")

        if not text or not text.strip():
            return "", 0.0, "tesseract"

        data = pytesseract.image_to_data(pil, output_type=pytesseract.Output.DICT)
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


def extract_docx_text(docx_bytes):
    """Read text directly from a .docx file. No OCR involved."""
    from docx import Document
    doc = Document(BytesIO(docx_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return "\n\n".join(paragraphs)


def extract_pdf_text_or_route(pdf_bytes):
    """
    Returns one of:
      ("text",   full_text)                 → PDF had a text layer
      ("images", [page_png_bytes, ...])     → scanned PDF, pages to OCR
    """
    import fitz  # PyMuPDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    # 1. Check for an embedded text layer
    total_text = ""
    for page in doc:
        total_text += page.get_text()

    if len(total_text.strip()) > 100:
        doc.close()
        return "text", total_text

    # 2. Scanned — render pages to PNG for OCR
    page_images = []
    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap(dpi=200)
        page_images.append(pix.tobytes("png"))
    doc.close()
    return "images", page_images