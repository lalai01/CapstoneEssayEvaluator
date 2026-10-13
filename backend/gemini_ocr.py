"""
Read an essay page with a Gemini vision model.

Gemini reads whole handwritten pages far better than TrOCR on the server's
CPU, and in a second or two. It is told to copy the page exactly, keeping
the student's spelling, grammar and punctuation mistakes, because the
evaluator grades those.

Configuration (backend/.env):
  GEMINI_API_KEY      key from https://aistudio.google.com/apikey
                      (unset = this reader is skipped; TrOCR is used)
  GEMINI_OCR_MODELS   comma-separated models, tried in order when one is
                      unavailable or out of quota
                      (default: gemini-3.8-flash,gemini-3.5-flash-lite)
"""
import base64
import json
import os

import cv2
import numpy as np
import requests

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
DEFAULT_MODELS = "gemini-3.8-flash,gemini-3.5-flash-lite"
TIMEOUT = 90
MAX_SIDE = 2400   # larger photos are scaled down before sending

PROMPT = """You are an exact transcription tool for student essays. Transcribe the
handwritten or printed English text in this image.

Rules:
- Copy exactly what is written. Keep every spelling, grammar, capitalization
  and punctuation mistake. Never correct, improve, complete or reword anything.
- Join the lines of a paragraph with single spaces. Separate paragraphs with
  one blank line. Keep the title or heading as its own line.
- A word split across two lines with a hyphen: write it as one word.
- Leave out words that are crossed out. Put words inserted with a caret or
  written above the line where the writer marked them.
- Write [illegible] for a word you cannot read. Do not guess whole phrases.
- Ignore ruled lines, margins, page numbers and stray marks.

Also return "lines": every written line of the page, top to bottom, exactly as
it appears on the paper (one entry per written line, a hyphen at a line end
kept as written, crossed-out words left out). This is used to build a
handwriting dataset, so line breaks must match the page.

Set "is_text" to false only if the image contains no readable written English."""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_text": {"type": "BOOLEAN"},
        "text": {"type": "STRING"},
        "lines": {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["is_text", "text", "lines"],
}


def enabled():
    return bool(os.environ.get("GEMINI_API_KEY", "").strip())


def _models():
    raw = os.environ.get("GEMINI_OCR_MODELS", DEFAULT_MODELS)
    return [m.strip() for m in raw.split(",") if m.strip()]


def _prepare(image_bytes):
    """JPEG bytes, scaled so the longest side is at most MAX_SIDE."""
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("image decode failed")
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("image encode failed")
    return buf.tobytes()


def read_page(image_bytes):
    """
    Returns (is_text, text, lines, model): the essay as paragraphs, and the
    page's written lines one per entry. Raises RuntimeError when no model could
    read the page (no key, network error, quota), so the caller can fall
    back to the local readers.
    """
    key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    body = {
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg",
                                 "data": base64.b64encode(_prepare(image_bytes)).decode()}},
                {"text": PROMPT},
            ],
        }],
        "generationConfig": {
            "temperature": 0,
            "responseMimeType": "application/json",
            "responseSchema": SCHEMA,
        },
    }

    errors = []
    for model in _models():
        try:
            resp = requests.post(API_URL.format(model=model), json=body, timeout=TIMEOUT,
                                 headers={"x-goog-api-key": key})
        except requests.RequestException as e:
            errors.append(f"{model}: {e}")
            continue
        if resp.status_code != 200:
            # 404 retired model, 429 out of quota, 5xx overloaded: try the next one.
            errors.append(f"{model}: HTTP {resp.status_code} {resp.text[:200]}")
            continue
        try:
            candidate = resp.json()["candidates"][0]
            raw = "".join(p.get("text", "") for p in candidate["content"]["parts"])
            data = json.loads(raw)
        except (KeyError, IndexError, ValueError) as e:
            errors.append(f"{model}: unreadable response ({e})")
            continue
        lines = [str(l).strip() for l in (data.get("lines") or []) if str(l).strip()]
        return bool(data.get("is_text")), (data.get("text") or "").strip(), lines, model

    raise RuntimeError("; ".join(errors) or "no Gemini model configured")
