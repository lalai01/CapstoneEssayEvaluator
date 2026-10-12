"""
Quick check, before any OCR, that an uploaded image looks like a page of
writing (handwritten or printed English paragraphs) and not a photo of a
face, a car, scenery, etc.

A page of writing has many small marks (letters) of similar height, sitting
in rows, on a mostly plain, light, low-colour background. Photos have few
letter-like marks, they don't line up in rows, and the background varies.
The OCR result is checked again afterwards (text_quality), so this check
can stay lenient: it only has to catch obvious non-documents cheaply.
"""
import cv2
import numpy as np


def ink_mask(gray):
    """
    Binary ink mask (255 = ink) that copes with uneven lighting in phone
    photos, with ruled notebook lines, margins and specks removed.
    Returns (mask, component labels, component stats).
    """
    block = max(15, (min(gray.shape) // 25) | 1)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, block, 15)
    long_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, gray.shape[1] // 8), 1))
    long_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, gray.shape[0] // 6)))
    rules = cv2.bitwise_or(cv2.morphologyEx(bw, cv2.MORPH_OPEN, long_h),
                           cv2.morphologyEx(bw, cv2.MORPH_OPEN, long_v))
    bw = cv2.subtract(bw, rules)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(bw, 8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= 12
    labels = np.where(keep[labels], labels, 0)
    return (labels > 0).astype(np.uint8) * 255, labels, stats


def page_metrics(image_bytes):
    """Measurements used to decide whether an image is a page of writing."""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    if img.shape[1] > 1000:
        f = 1000 / img.shape[1]
        img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    _, _, stats = ink_mask(gray)
    s = stats[1:]
    if len(s) == 0:
        return {"glyphs": 0, "aligned": 0.0, "light_bg": 0.0, "saturation": 0.0, "ink": 0.0}

    h = s[:, cv2.CC_STAT_HEIGHT].astype(float)
    w = s[:, cv2.CC_STAT_WIDTH].astype(float)
    area = s[:, cv2.CC_STAT_AREA].astype(float)
    fill = area / np.maximum(1.0, h * w)
    glyph = (h >= max(5, 0.008 * H)) & (h <= 0.12 * H) & (w <= 0.25 * W) & (fill > 0.05) & (fill < 0.95)
    g = s[glyph]
    n = len(g)
    aligned = 0.0
    if n >= 10:
        cy = g[:, cv2.CC_STAT_TOP] + g[:, cv2.CC_STAT_HEIGHT] / 2.0
        cx = g[:, cv2.CC_STAT_LEFT] + g[:, cv2.CC_STAT_WIDTH] / 2.0
        mh = float(np.median(g[:, cv2.CC_STAT_HEIGHT]))
        dy = np.abs(cy[:, None] - cy[None, :])
        dx = np.abs(cx[:, None] - cx[None, :])
        neighbours = ((dy < 0.6 * mh) & (dx < 5 * mh) & (dx > 0)).sum(axis=1)
        aligned = float((neighbours >= 2).mean())

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # The paper: its typical brightness, and how much of the image is close to it.
    paper = float(np.percentile(gray, 75))
    light_bg = float(((np.abs(gray.astype(float) - paper) < 35) & (gray > 110)).mean())
    return {
        "glyphs": int(n),
        "aligned": round(aligned, 3),
        "light_bg": round(light_bg, 3),
        "saturation": round(float(hsv[:, :, 1].mean()), 1),
        "ink": round(float(area.sum()) / (H * W), 3),
    }


# Thresholds chosen on 13 handwritten pages (scanned and phone-photo style,
# with shadow, tilt, blur and lamp tint) and printed pages, versus everyday
# photos in colour and grayscale (face, animals, objects). Letters lining up
# in rows is the strongest signal: pages scored >= 0.85, photos <= 0.61.
# A shadow across a page lowers the light-background share, so that test
# only rejects images with almost no paper at all.
MIN_GLYPHS = 40
MIN_ALIGNED = 0.7
MIN_LIGHT_BACKGROUND = 0.15
MAX_SATURATION = 90


def looks_like_page(image_bytes):
    """(is_page, metrics, reason). reason explains a rejection."""
    m = page_metrics(image_bytes)
    if m is None:
        return False, None, "The file could not be opened as an image."
    if m["glyphs"] < MIN_GLYPHS:
        return False, m, "No lines of writing were found in the image."
    if m["aligned"] < MIN_ALIGNED:
        return False, m, "The marks in the image don't form lines of text."
    if m["light_bg"] < MIN_LIGHT_BACKGROUND or m["saturation"] > MAX_SATURATION:
        return False, m, "The image looks like a photo rather than a page of writing."
    return True, m, ""
