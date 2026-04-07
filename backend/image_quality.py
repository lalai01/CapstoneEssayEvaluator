import cv2
import numpy as np

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
    # Return only small angles (skew correction beyond 45° is unusual)
    return median_angle if abs(median_angle) < 45 else 0.0

def is_handwritten(image):
    """
    Heuristic: handwriting tends to have edge density between 0.08 and 0.45.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
    return 0.08 < edge_density < 0.45

def recommend_ocr_engine(image_bytes):
    """
    Analyze image quality and decide OCR engine.
    Returns 'google_vision' for poor quality or handwriting, otherwise 'tesseract'.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return 'google_vision'   # fallback

    sharpness = estimate_sharpness(img)
    contrast = estimate_contrast(img)
    skew = abs(detect_skew(img))
    handwritten = is_handwritten(img)

    # Aggressive thresholds: use Google Vision for any sign of trouble
    if handwritten or sharpness < 300 or contrast < 60 or skew > 2:
        return 'google_vision'
    else:
        return 'tesseract'