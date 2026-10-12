import math
import os
import threading
import cv2
import numpy as np
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import torch

from essay_image_check import ink_mask as _ink_mask

_processor = None
_model = None
_load_lock = threading.Lock()
# One model instance is shared by all requests and PDF page workers.
# PyTorch already uses every core per forward pass, so running two passes
# at once only makes both slower; serialize them.
_infer_lock = threading.Lock()

# Where the TrOCR weights live (the Dockerfile saves them to /opt/trocr).
TROCR_PATH = os.environ.get("TROCR_PATH", "/opt/trocr")
# A handwritten line is rarely more than ~30 tokens; 64 leaves headroom
# while stopping runaway generations early.
MAX_LINE_TOKENS = 64
BATCH_SIZE = 8
# Beam search reads handwriting noticeably better than greedy decoding, at
# roughly 1.5-2x the time per line. Set HANDWRITING_BEAMS=1 for speed.
NUM_BEAMS = max(1, int(os.environ.get("HANDWRITING_BEAMS", "3")))


def _load():
    global _processor, _model
    with _load_lock:
        if _processor is not None:
            return
        processor = TrOCRProcessor.from_pretrained(TROCR_PATH)
        model = VisionEncoderDecoderModel.from_pretrained(TROCR_PATH)
        model.eval()
        # Optional int8 dynamic quantization: roughly 2x faster on CPU with
        # a small accuracy cost. Enable with TROCR_QUANTIZE=1 after checking
        # results on your own samples.
        if os.environ.get("TROCR_QUANTIZE") == "1":
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
            print("[TrOCR] using int8 dynamic quantization")
        _model = model
        _processor = processor


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


def deskew(gray):
    """
    Straighten a slightly rotated page: try small angles and keep the one
    whose row-ink profile is sharpest (text lines become crisp peaks).
    Safe no-op when no rotation helps.
    """
    small = gray
    scale = 1.0
    if gray.shape[1] > 800:
        scale = 800 / gray.shape[1]
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    mask, _, _ = _ink_mask(small)
    if cv2.countNonZero(mask) < 200:
        return gray
    h, w = mask.shape
    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-6.0, 6.01, 0.5):
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rot = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)
        score = float(np.var(rot.sum(axis=1, dtype=np.float64)))
        if score > best_score:
            best_angle, best_score = float(angle), score
    if abs(best_angle) < 0.5:
        return gray
    H, W = gray.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), best_angle, 1.0)
    return cv2.warpAffine(gray, M, (W, H), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _letter_height(stats, page_height):
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    heights = heights[(heights > 4) & (heights < page_height / 4)]
    return float(np.median(heights)) if len(heights) else 20.0


def segment_lines(gray):
    """
    Find text lines. Returns a list of (top, bottom, is_paragraph_break).

    Handwriting has ascenders and descenders reaching into the next line, so
    a fixed ink threshold merges neighbouring lines. Instead each line is
    found as a peak of the row-ink profile (peaks at least about one letter
    height apart) and lines are split at the quietest row between peaks.
    A gap between line centres well above the usual spacing marks a new
    paragraph.
    """
    mask, _, stats = _ink_mask(gray)
    letter_h = _letter_height(stats, gray.shape[0])
    profile = mask.sum(axis=1).astype(np.float32) / 255.0
    sigma = max(1.5, letter_h * 0.35)
    k = int(sigma * 3) * 2 + 1
    smooth = cv2.GaussianBlur(profile.reshape(-1, 1), (1, k), sigma).ravel()
    if smooth.max() <= 0:
        return []

    min_gap = max(6, int(letter_h * 0.9))
    floor = smooth.max() * 0.12
    peaks = []
    for y in range(1, len(smooth) - 1):
        if smooth[y] >= floor and smooth[y] >= smooth[y - 1] and smooth[y] > smooth[y + 1]:
            if peaks and y - peaks[-1] < min_gap:
                if smooth[y] > smooth[peaks[-1]]:
                    peaks[-1] = y
            else:
                peaks.append(y)
    if not peaks:
        return []

    bands = []
    for i, p in enumerate(peaks):
        if i > 0:
            top = peaks[i - 1] + int(np.argmin(smooth[peaks[i - 1]:p]))
        else:
            top = p
            while top > 0 and smooth[top - 1] > floor * 0.25:
                top -= 1
        if i + 1 < len(peaks):
            bottom = p + int(np.argmin(smooth[p:peaks[i + 1]]))
        else:
            bottom = p
            while bottom < len(smooth) - 1 and smooth[bottom + 1] > floor * 0.25:
                bottom += 1
        bands.append((top, bottom))

    spacing = np.diff(peaks)
    usual = float(np.median(spacing)) if len(spacing) else float("inf")
    tagged = [(t, b, i > 0 and (peaks[i] - peaks[i - 1]) > usual * 1.6)
              for i, (t, b) in enumerate(bands)]
    print(f"[TrOCR] segment_lines: {len(tagged)} lines, letter height {letter_h:.0f}px, "
          f"{sum(1 for _, _, br in tagged if br)} paragraph breaks")
    return tagged


def _line_crops(gray, bands):
    """
    One clean image per line: keep only the strokes whose centre lies in the
    line (so tails of letters from the lines above and below don't confuse
    the model), trimmed to the line's own width, on a white background.
    """
    mask, labels, stats = _ink_mask(gray)
    centers = stats[:, cv2.CC_STAT_TOP] + stats[:, cv2.CC_STAT_HEIGHT] / 2.0
    crops = []
    for top, bottom, is_break in bands:
        ids = [i for i in range(1, len(stats))
               if top <= centers[i] < bottom + 1 and stats[i, cv2.CC_STAT_AREA] >= 12]
        if not ids:
            continue
        x0 = int(min(stats[i, cv2.CC_STAT_LEFT] for i in ids))
        x1 = int(max(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] for i in ids))
        y0 = int(min(stats[i, cv2.CC_STAT_TOP] for i in ids))
        y1 = int(max(stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] for i in ids))
        ink = int(sum(stats[i, cv2.CC_STAT_AREA] for i in ids))
        if ink < 60 or (x1 - x0) < 15:
            continue  # a stray mark, not a line of writing
        region = labels[y0:y1, x0:x1]
        keep = np.isin(region, ids)
        # Grow the kept strokes slightly so anti-aliased edges survive.
        keep = cv2.dilate(keep.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
        line = np.where(keep, gray[y0:y1, x0:x1], 255).astype(np.uint8)
        line = cv2.copyMakeBorder(line, 12, 12, 16, 16, cv2.BORDER_CONSTANT, value=255)
        crops.append((Image.fromarray(line).convert("RGB"), is_break))
    return crops


def _transcribe_batch(pil_images):
    """
    Transcribe a batch of line crops in one forward pass.
    Returns (list_of_texts, list_of_confidences).
    """
    pixel_values = _processor(
        images=pil_images, return_tensors="pt"
    ).pixel_values

    with _infer_lock, torch.inference_mode():
        outputs = _model.generate(
            pixel_values,
            max_new_tokens=MAX_LINE_TOKENS,
            num_beams=NUM_BEAMS,
            early_stopping=NUM_BEAMS > 1,
            no_repeat_ngram_size=3,   # stops "the the the" loops
            output_scores=True,
            return_dict_in_generate=True,
        )

    texts = _processor.batch_decode(
        outputs.sequences, skip_special_tokens=True
    )

    if NUM_BEAMS > 1 and getattr(outputs, "sequences_scores", None) is not None:
        # Length-normalised log-probability of each chosen sequence.
        confidences = [math.exp(float(s)) * 100 for s in outputs.sequences_scores]
        return texts, confidences

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


def page_line_crops(gray):
    """
    Grayscale page -> [(line image, starts_paragraph)], top to bottom.
    Used both for reading and by export_training_data.py, so training crops
    look exactly like the ones the model reads in production.
    """
    gray = _crop_to_content(gray)
    gray = deskew(gray)
    # Tiny handwriting (e.g. a distant phone photo): enlarge so strokes
    # survive binarisation; otherwise keep the size (TrOCR resizes lines).
    _, _, stats = _ink_mask(gray)
    if _letter_height(stats, gray.shape[0]) < 14:
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    bands = segment_lines(gray)
    return _line_crops(gray, bands) if bands else []


def ocr_handwriting_image(image_bytes):
    """
    Full-page handwriting OCR:
      decode -> crop to content -> deskew -> find lines -> clean line crops ->
      batch-transcribe with TrOCR.
    Preserves paragraph breaks as blank lines.
    """
    _load()

    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        print("[TrOCR] image decode failed")
        return "", 0.0

    crops = page_line_crops(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    if not crops:
        print("[TrOCR] no line crops to process")
        return "", 0.0

    texts, confidences, flags = [], [], []
    for i in range(0, len(crops), BATCH_SIZE):
        batch = crops[i:i + BATCH_SIZE]
        batch_texts, batch_confs = _transcribe_batch([c[0] for c in batch])
        for (_, is_break), t, c in zip(batch, batch_texts, batch_confs):
            if t.strip():
                texts.append(t.strip())
                confidences.append(c)
                flags.append(is_break)

    if not texts:
        print("[TrOCR] no lines produced text")
        return "", 0.0

    # One line per row; a blank line at each paragraph break.
    parts = []
    for idx, t in enumerate(texts):
        if idx > 0 and flags[idx]:
            parts.append("")
        parts.append(t)

    full = "\n".join(parts)
    avg_conf = sum(confidences) / len(confidences)
    print(f"[TrOCR] produced {len(full)} chars across {len(texts)} lines, "
          f"{sum(flags)} paragraph breaks, avg conf={avg_conf:.1f}")
    return full, avg_conf
