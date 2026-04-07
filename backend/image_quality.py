import cv2
import numpy as np
import easyocr

# Global EasyOCR reader (lazy init)
_easyocr_reader = None

def get_easyocr_reader():
    global _easyocr_reader
    if _easyocr_reader is None:
        _easyocr_reader = easyocr.Reader(['en'], gpu=False)
    return _easyocr_reader

def estimate_sharpness(image):
    """Laplacian variance – higher means sharper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def estimate_contrast(image):
    """RMS contrast."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.std(gray)

def detect_skew(image):
    """Detect skew angle in degrees."""
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
    """
    Heuristic: handwriting tends to have edge density between 0.08 and 0.45.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
    return 0.08 < edge_density < 0.45

def assess_handwriting_messiness(image_bytes):
    """
    Returns a messiness score between 0 (clean) and 1 (very messy).
    Criteria:
      - Edge density uniformity (messy handwriting has irregular edge density)
      - Number of small blobs (broken characters)
      - (If EasyOCR is used, its per-character confidence variance)
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 1. Edge density variance
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
    
    # 2. Broken characters – small connected components
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_components = sum(1 for c in contours if cv2.contourArea(c) < 50)
    total_components = len(contours)
    broken_ratio = small_components / max(1, total_components)
    
    # 3. Use EasyOCR on a small sample to get confidence variance
    try:
        reader = get_easyocr_reader()
        # Sample only central region for speed
        h, w, _ = img.shape
        crop = img[h//4:3*h//4, w//4:3*w//4]
        _, encoded = cv2.imencode('.jpg', crop)
        results = reader.readtext(encoded.tobytes())
        if results:
            confs = [conf for (_, _, conf) in results]
            conf_variance = np.var(confs) if len(confs) > 1 else 0
            # Normalise (max variance ~0.25)
            conf_score = min(1.0, conf_variance / 0.25)
        else:
            conf_score = 0
    except Exception:
        conf_score = 0
    
    # Combine scores (weights can be tuned)
    messiness = (edge_variance * 2 + broken_ratio * 1.5 + conf_score) / 4.5
    return min(1.0, messiness)

def recommend_ocr_engine(image_bytes):
    """
    Returns 'easyocr', 'google_vision', or 'tesseract'.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return 'google_vision'
    
    sharpness = estimate_sharpness(img)
    contrast = estimate_contrast(img)
    skew = abs(detect_skew(img))
    handwritten = is_handwritten(img)
    
    # If handwriting is detected, evaluate messiness
    if handwritten:
        messiness = assess_handwriting_messiness(image_bytes)
        if messiness > 0.6:          # threshold for messy handwriting
            return 'easyocr'
        # Otherwise fall through to normal logic
    
    # Original logic
    if sharpness < 300 or contrast < 60 or skew > 2:
        return 'google_vision'
    else:
        return 'tesseract'