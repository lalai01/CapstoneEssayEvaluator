import cv2
import numpy as np

PADDLEOCR_AVAILABLE = False
EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR  # noqa: F401
    PADDLEOCR_AVAILABLE = True
except ImportError:
    pass

try:
    import easyocr  # noqa: F401
    EASYOCR_AVAILABLE = True
except ImportError:
    pass


def get_priority_engines():
    """
    Return OCR engines in the order they should be tried.
    Only includes engines whose Python packages are importable
    in this container. Tesseract is always included as a fallback.
    """
    engines = []
    if PADDLEOCR_AVAILABLE:
        engines.append("paddleocr")
    if EASYOCR_AVAILABLE:
        engines.append("easyocr")
    engines.append("tesseract")   # Always fallback
    return engines


# ---- image quality helpers (kept for future engine-selection logic) ----
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
        edge_variance = np.var(densities) if densities else 0
        _, thresh = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        small = sum(1 for c in contours if cv2.contourArea(c) < 50)
        total = len(contours)
        broken_ratio = small / max(1, total)
        messiness = (edge_variance * 2 + broken_ratio * 1.5) / 3.5
        return min(1.0, messiness)
    except Exception as e:
        print(f"assess_handwriting_messiness failed: {e}")
        return 0.0


def extract_text_from_image(image_path):
    """
    Extract text using Tesseract with per-word confidence scoring.
    Returns a tuple: (text, confidence, engine_name).
    If no text is detected, returns ("", 0.0, "tesseract") — never a
    fake confidence value for empty output.
    """
    import pytesseract
    from PIL import Image

    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)

        # No text → don't compute a bogus confidence from empty tokens.
        if not text or not text.strip():
            return "", 0.0, "tesseract"

        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        # Only count confidence for tokens that actually contain text.
        # Tesseract reports high conf for whitespace/box regions; those
        # inflate the average and lie about accuracy.
        conf_values = []
        tokens = data.get("text", [])
        for c, t in zip(data["conf"], tokens):
            try:
                v = float(c)
            except (TypeError, ValueError):
                continue
            if v >= 0 and t and t.strip():
                conf_values.append(v)

        conf = sum(conf_values) / len(conf_values) if conf_values else 0.0
        return text, conf, "tesseract"
    except Exception as e:
        print(f"Tesseract failed on {image_path}: {e}")
        return "", 0.0, "tesseract"