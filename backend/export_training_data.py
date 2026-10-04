"""
Export accepted OCR Labeler samples from Supabase into a line-level
TrOCR training set.

For each accepted row in ocr_training_data, the page image is segmented
into line bands with the same preprocessing ocr_handwriting_image uses at
inference time, and each band is paired with the matching line of
corrected_text. Rows whose band count does not match the number of text
lines are skipped, since the pairing would be wrong.

Output:
  training-data/crops/{row_id}_{i}.png
  training-data/labels.csv   (file_name,text)

Usage (from backend/):
  python export_training_data.py
"""
import csv
import os

import cv2
import numpy as np
import requests

from supabase_client import supabase
from handwriting_ocr import (
    _crop_to_content,
    _upscale_if_small,
    deskew,
    segment_lines,
)

OUT_DIR = "training-data"
CROPS_DIR = os.path.join(OUT_DIR, "crops")
LABELS_PATH = os.path.join(OUT_DIR, "labels.csv")

PAGE_SIZE = 1000   # PostgREST returns at most 1000 rows per request
CROP_PAD = 8       # vertical padding around each band, as in ocr_handwriting_image
BORDER = 10        # white border around each crop, as in ocr_handwriting_image


def fetch_rows():
    """Fetch all accepted, labeled rows, paging past the 1000-row limit."""
    rows = []
    start = 0
    while True:
        resp = (
            supabase.table("ocr_training_data")
            .select("id, image_url, corrected_text, engine")
            .eq("accepted", True)
            .not_.is_("corrected_text", "null")
            .not_.is_("image_url", "null")
            .order("id")
            .range(start, start + PAGE_SIZE - 1)
            .execute()
        )
        batch = resp.data or []
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            return rows
        start += PAGE_SIZE


def prepare_page(image_bytes):
    """
    Decode and preprocess a page exactly as ocr_handwriting_image does, so
    band coordinates and crop appearance match what TrOCR sees at inference.
    """
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("image decode failed")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = _crop_to_content(gray)
    gray = deskew(gray)
    gray = _upscale_if_small(gray)
    return gray


def export_row(row, labels):
    """
    Write crops for one row and append their labels.
    Returns the number of crops written, or None if the row was skipped.
    """
    row_id = row["id"]

    resp = requests.get(row["image_url"], timeout=30)
    resp.raise_for_status()

    gray = prepare_page(resp.content)
    bands = segment_lines(gray)

    label_lines = [
        line for line in (row["corrected_text"] or "").splitlines()
        if line.strip()
    ]

    if len(bands) != len(label_lines):
        print(f"[export] row {row_id}: {len(bands)} bands vs "
              f"{len(label_lines)} text lines — skipped")
        return None

    row_labels = []
    for i, (top, bottom, _is_break) in enumerate(bands):
        crop = gray[max(0, top - CROP_PAD):bottom + CROP_PAD, :]
        crop = cv2.copyMakeBorder(crop, BORDER, BORDER, BORDER, BORDER,
                                  cv2.BORDER_CONSTANT, value=255)
        file_name = f"{row_id}_{i}.png"
        if not cv2.imwrite(os.path.join(CROPS_DIR, file_name), crop):
            raise IOError(f"failed to write {file_name}")
        row_labels.append((file_name, label_lines[i].strip()))

    labels.extend(row_labels)
    return len(row_labels)


def main():
    if supabase is None:
        raise SystemExit("Supabase client not configured "
                         "(set SUPABASE_URL and SUPABASE_KEY).")

    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CROPS_DIR, exist_ok=True)

    rows = fetch_rows()
    print(f"[export] {len(rows)} accepted rows to process")

    labels = []
    rows_processed = 0
    rows_skipped = 0
    crops_written = 0

    for row in rows:
        try:
            written = export_row(row, labels)
        except Exception as e:
            print(f"[export] row {row.get('id')}: error — {e}")
            rows_skipped += 1
            continue

        if written is None:
            rows_skipped += 1
        else:
            rows_processed += 1
            crops_written += written

    with open(LABELS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["file_name", "text"])
        writer.writerows(labels)

    print(f"Exported {crops_written} crops from {rows_processed} rows. "
          f"Skipped {rows_skipped} rows (band/line mismatch).")
    print(f"Wrote {LABELS_PATH}")
    print(f"Wrote {crops_written} PNG files to {CROPS_DIR}/")


if __name__ == "__main__":
    main()
