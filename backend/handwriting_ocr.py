import math
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

_processor = None
_model = None


def _load():
    global _processor, _model
    if _processor is None:
        _processor = TrOCRProcessor.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        _model = VisionEncoderDecoderModel.from_pretrained(
            "microsoft/trocr-base-handwritten"
        )
        _model.eval()


def deskew(gray):
    """Rotate the image to correct small skew angles."""
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) < 100:
        return gray
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5 or abs(angle) > 15:
        return gray
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def segment_lines(gray, min_height=20, gap_factor=0.5):
    """
    Robust line segmentation for both clean and messy handwriting.
    Returns a list of (top, bottom) row bands.
    """
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = thresh.sum(axis=1).astype(np.float32)

    # Smooth the projection to survive gaps in messy writing
    kernel = np.ones(5) / 5
    proj_smooth = np.convolve(proj, kernel, mode="same")

    nz = proj_smooth[proj_smooth > 0]
    if len(nz) == 0:
        return []
    thresh_val = nz.mean() * 0.15

    above = proj_smooth > thresh_val
    bands = []
    start = None
    for i, a in enumerate(above):
        if a and start is None:
            start = i
        elif not a and start is not None:
            bands.append((start, i - 1))
            start = None
    if start is not None:
        bands.append((start, len(above) - 1))

    # Merge bands separated by less than gap_factor * median height
    if len(bands) > 1:
        heights = [b - a for a, b in bands]
        median_h = float(np.median(heights))
        min_gap = gap_factor * median_h
        merged = [bands[0]]
        for a, b in bands[1:]:
            prev_a, prev_b = merged[-1]
            if a - prev_b < min_gap:
                merged[-1] = (prev_a, b)
            else:
                merged.append((a, b))
        bands = merged

    bands = [(a, b) for a, b in bands if (b - a) >= min_height]
    return bands


def _transcribe_line(pil_image):
    """Return (text, mean_logprob_confidence_0_to_100)."""
    pixel_values = _processor(images=pil_image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = _model.generate(
            pixel_values,
            max_new_tokens=128,
            output_scores=True,
            return_dict_in_generate=True,
        )

    ids = outputs.sequences[0]
    text = _processor.batch_decode([ids], skip_special_tokens=True)[0]

    # Confidence: geometric mean of per-token probabilities
    scores = outputs.scores
    if scores:
        probs = []
        for i, s in enumerate(scores):
            if i + 1 >= len(ids):
                break
            p = torch.softmax(s[0], dim=-1)[ids[i + 1]].item()
            probs.append(p)
        if probs:
            gm = math.exp(sum(math.log(p + 1e-9) for p in probs) / len(probs))
            return text, gm * 100

    return text, 0.0


def ocr_handwriting_image(image_bytes):
    """Full-page handwriting OCR: deskew → segment → per-line TrOCR."""
    _load()

    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return "", 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = deskew(gray)

    bands = segment_lines(gray)
    if not bands:
        return "", 0.0

    texts = []
    confidences = []
    for (top, bottom) in bands:
        pad = 4
        crop = img[max(0, top - pad):bottom + pad, :]
        pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        line_text, line_conf = _transcribe_line(pil)
        if line_text.strip():
            texts.append(line_text)
            confidences.append(line_conf)

    if not texts:
        return "", 0.0

    full = "\n".join(texts)
    avg_conf = sum(confidences) / len(confidences)
    return full, avg_conf