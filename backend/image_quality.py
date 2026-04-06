import cv2
import numpy as np
import math

def estimate_sharpness(image):
    """Laplacian variance – higher = sharper."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    variance = laplacian.var()
    return variance

def estimate_contrast(image):
    """RMS contrast."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    contrast = np.sqrt(np.mean((gray - mean) ** 2))
    return contrast

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
    Heuristic: use edge density and texture.
    Handwriting tends to have more short, wavy edges.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    # Count number of edge pixels
    edge_pixels = np.sum(edges > 0)
    total_pixels = gray.shape[0] * gray.shape[1]
    edge_density = edge_pixels / total_pixels
    # Handwriting often has edge density between 0.05 and 0.2
    # Also check for regularity (low)
    return edge_density > 0.05 and edge_density < 0.25

def recommend_ocr_engine(image_bytes):
    """
    Analyze image and return recommended engine: 'tesseract', 'easyocr', or 'paddleocr'.
    """
    # Decode image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return 'tesseract'  # fallback

    sharpness = estimate_sharpness(img)
    contrast = estimate_contrast(img)
    skew = abs(detect_skew(img))
    handwritten = is_handwritten(img)

    # Decision logic
    if handwritten:
        # Handwriting is better with EasyOCR or PaddleOCR
        return 'easyocr'
    elif sharpness < 100:   # low sharpness
        return 'easyocr'
    elif contrast < 30:
        return 'easyocr'
    elif skew > 5:
        return 'easyocr'
    else:
        return 'tesseract'