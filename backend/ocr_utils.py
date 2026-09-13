import cv2
import numpy as np
import os

PADDLEOCR_AVAILABLE = False
EASYOCR_AVAILABLE = False

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    pass

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass

def get_priority_engines():
    engines = []
    if PADDLEOCR_AVAILABLE:
        engines.append('paddleocr')
    if EASYOCR_AVAILABLE:
        engines.append('easyocr')
    engines.append('tesseract')   # Always fallback
    return engines

# ---- the rest of your image quality functions remain unchanged ----
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
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
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
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 30, 100)
        h, w = edges.shape
        block_h, block_w = h//4, w//4
        densities = []
        for i in range(4):
            for j in range(4):
                block = edges[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                if block.size > 0:
                    densities.append(np.sum(block > 0) / block.size)
        edge_variance = np.var(densities) if densities else 0
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        small = sum(1 for c in contours if cv2.contourArea(c) < 50)
        total = len(contours)
        broken_ratio = small / max(1, total)
        messiness = (edge_variance * 2 + broken_ratio * 1.5) / 3.5
        return min(1.0, messiness)
    except:
        return 0.0
    
def extract_text_from_image(image_path):
    """
    Extract text with automatic engine selection:
    1. Preprocess the image
    2. If handwriting is detected as messy → use TrOCR
    3. Else → Tesseract with both raw and preprocessed paths
    """
    # ---- Try handwriting recognizer for messy images ----
    try:
        from image_quality import assess_handwriting_messiness
        with open(image_path, "rb") as f:
            contents = f.read()
        messiness = assess_handwriting_messiness(contents)
        if messiness > 0.6:
            from handwriting_ocr import ocr_handwriting_image
            text = ocr_handwriting_image(contents)
            if text.strip():
                return text, 85.0, "trocr_handwriting"
    except Exception as e:
        print(f"Handwriting path failed: {e}")

    # ---- Standard Tesseract path with preprocessing ----
    import pytesseract
    from PIL import Image

    best_text = ""
    best_conf = 0.0

    for path in [image_path]:
        try:
            img = Image.open(path)
            text = pytesseract.image_to_string(img)
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            conf_values = [float(c) for c in data["conf"] if c != "-1"]
            conf = sum(conf_values) / len(conf_values) if conf_values else 0

            if len(text.strip()) > len(best_text.strip()) and conf >= best_conf:
                best_text = text
                best_conf = conf
        except Exception as e:
            print(f"Tesseract failed on {path}: {e}")

    return best_text, best_conf, "tesseract"