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
        _processor = TrOCRProcessor.from_pretrained("/opt/trocr")
        _model = VisionEncoderDecoderModel.from_pretrained("/opt/trocr")
        _model.eval()


def _crop_to_content(gray):
    """Remove white borders/background so projection sees only text."""
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8))
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return gray
    x, y, w, h = cv2.boundingRect(coords)
    pad = 20
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(gray.shape[1] - x, w + 2 * pad)
    h = min(gray.shape[0] - y, h + 2 * pad)
    return gray[y:y + h, x:x + w]


def _upscale_if_small(gray, min_height=1200):
    """Upscale images that are too small for TrOCR's line crops."""
    h, w = gray.shape
    if h >= min_height:
        return gray
    scale = min_height / h
    new_w = int(w * scale)
    return cv2.resize(gray, (new_w, min_height),
                      interpolation=cv2.INTER_CUBIC)


def deskew(gray):
    """Correct small rotations. Safe no-op if no clear skew."""
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
    return cv2.warpAffine(gray, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def segment_lines(gray):
    """
    Robust horizontal line segmentation.
    Returns a list of (top, bottom, is_paragraph_break) tuples.
    `is_paragraph_break` is True when the gap above this band is much
    larger than the normal line gap — i.e. a blank line between paragraphs.
    """
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,
                              np.ones((3, 3), np.uint8))

    thresh = cv2.dilate(thresh, np.ones((1, 25), np.uint8), iterations=1)

    proj = thresh.sum(axis=1).astype(np.float32)

    kernel = np.ones(5) / 5
    proj_smooth = np.convolve(proj, kernel, mode="same")

    if proj_smooth.max() == 0:
        return []

    thresh_val = proj_smooth.max() * 0.4

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

    if not bands:
        return []

    raw_count = len(bands)

    if len(bands) > 1:
        heights = [b - a for a, b in bands]
        median_h = float(np.median(heights))
        min_gap = min(12, int(0.3 * median_h))
        merged = [bands[0]]
        for a, b in bands[1:]:
            prev_a, prev_b = merged[-1]
            if a - prev_b < min_gap:
                merged[-1] = (prev_a, b)
            else:
                merged.append((a, b))
        bands = merged

    min_h = max(15, gray.shape[0] // 100)
    bands = [(a, b) for a, b in bands if (b - a) >= min_h]

    # Detect paragraph breaks: a gap above a band that is larger than
    # ~1.6× the median line gap counts as a paragraph boundary.
    if len(bands) > 1:
        gaps = [bands[i][0] - bands[i - 1][1] for i in range(1, len(bands))]
        median_gap = float(np.median(gaps))
        paragraph_threshold = max(median_gap * 1.6, median_gap + 12)
    else:
        paragraph_threshold = float("inf")

    tagged = []
    for i, (top, bottom) in enumerate(bands):
        if i == 0:
            tagged.append((top, bottom, False))
        else:
            gap = top - bands[i - 1][1]
            is_break = gap >= paragraph_threshold
            tagged.append((top, bottom, is_break))

    breaks = sum(1 for _, _, b in tagged if b)
    print(f"[TrOCR] segment_lines: {raw_count} raw bands, "
          f"{len(tagged)} after merge+filter, {breaks} paragraph breaks")
    return tagged


def _transcribe_batch(pil_images):
    """
    Transcribe a batch of line crops in one forward pass.
    Returns (list_of_texts, list_of_confidences).
    """
    pixel_values = _processor(
        images=pil_images, return_tensors="pt"
    ).pixel_values

    with torch.no_grad():
        outputs = _model.generate(
            pixel_values,
            max_new_tokens=128,
            output_scores=True,
            return_dict_in_generate=True,
        )

    texts = _processor.batch_decode(
        outputs.sequences, skip_special_tokens=True
    )

    confidences = []
    scores = outputs.scores
    for seq_idx, ids in enumerate(outputs.sequences):
        probs = []
        for i, s in enumerate(scores):
            if i + 1 >= len(ids):
                break
            p = torch.softmax(s[seq_idx], dim=-1)[ids[i + 1]].item()
            probs.append(p)
        if probs:
            gm = math.exp(
                sum(math.log(p + 1e-9) for p in probs) / len(probs)
            )
            confidences.append(gm * 100)
        else:
            confidences.append(0.0)

    return texts, confidences


def ocr_handwriting_image(image_bytes):
    """
    Full-page handwriting OCR:
      decode -> crop to content -> deskew -> upscale ->
      segment lines -> batch-transcribe with TrOCR.
    Preserves paragraph breaks as blank lines.
    """
    _load()

    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print("[TrOCR] image decode failed")
        return "", 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = _crop_to_content(gray)
    gray = deskew(gray)
    gray = _upscale_if_small(gray)

    bands = segment_lines(gray)
    print(f"[TrOCR] segmentation found {len(bands)} line bands")

    if not bands:
        return "", 0.0

    # Build crops and remember each band's paragraph-break flag
    crops = []
    flags = []
    for (top, bottom, is_break) in bands:
        pad = 8
        crop = gray[max(0, top - pad):bottom + pad, :]
        crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10,
                                  cv2.BORDER_CONSTANT, value=255)
        crops.append(Image.fromarray(crop).convert("RGB"))
        flags.append(is_break)

    if not crops:
        print("[TrOCR] no crops to process")
        return "", 0.0

    BATCH_SIZE = 4
    texts = []
    confidences = []
    for i in range(0, len(crops), BATCH_SIZE):
        batch = crops[i:i + BATCH_SIZE]
        batch_texts, batch_confs = _transcribe_batch(batch)
        for t, c in zip(batch_texts, batch_confs):
            if t.strip():
                texts.append(t)
                confidences.append(c)

    if not texts:
        print("[TrOCR] no lines produced text")
        return "", 0.0

    # Join lines with a single newline, but insert a blank line at
    # every paragraph break so the extracted text mirrors the page layout.
    parts = []
    for idx, t in enumerate(texts):
        is_break = flags[idx] if idx < len(flags) else False
        if idx > 0 and is_break:
            parts.append("")          # blank line → paragraph break
        parts.append(t)

    full = "\n".join(parts)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    print(f"[TrOCR] produced {len(full)} chars across {len(texts)} lines, "
          f"{sum(1 for f in flags if f)} paragraph breaks, avg conf={avg_conf:.1f}")
    return full, avg_conf