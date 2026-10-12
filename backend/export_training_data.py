"""
Build a line-level TrOCR training set from full handwritten pages.

Each page is cut into lines with the same code the app uses to read
handwriting (handwriting_ocr.page_line_crops), so the training crops look
exactly like what the model sees in production. The page's transcript must
have one text line per written line; pages where the counts differ are
skipped, since the pairing would be wrong.

Sources:
  (default)         accepted OCR Labeler samples in Supabase (ocr_training_data)
  --pages <folder>  page images with a .txt transcript of the same name,
                    e.g. my_handwriting/my_line_1.png + my_line_1.txt

Output (new samples are added to what is already there):
  training-data/crops/{prefix}_{i}.png
  training-data/labels.csv   (file_name,text)

Usage (from backend/):
  python export_training_data.py
  python export_training_data.py --pages my_handwriting
"""
import argparse
import csv
import glob
import os

import cv2
import numpy as np

from handwriting_ocr import page_line_crops

OUT_DIR = "training-data"
CROPS_DIR = os.path.join(OUT_DIR, "crops")
LABELS_PATH = os.path.join(OUT_DIR, "labels.csv")

PAGE_SIZE = 1000   # PostgREST returns at most 1000 rows per request
IMAGE_TYPES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def fetch_rows():
    """Fetch all accepted, labeled rows, paging past the 1000-row limit."""
    from supabase_client import supabase
    if supabase is None:
        raise SystemExit("Supabase client not configured "
                         "(set SUPABASE_URL and SUPABASE_KEY).")
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


def export_page(prefix, image_bytes, transcript, labels):
    """
    Write one crop per line and append (file_name, text) to labels.
    Returns the number of crops written, or None if the page was skipped.
    """
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("image decode failed")
    crops = page_line_crops(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))

    text_lines = [line.strip() for line in (transcript or "").splitlines() if line.strip()]
    if len(crops) != len(text_lines):
        print(f"[export] {prefix}: {len(crops)} written lines found vs "
              f"{len(text_lines)} text lines — skipped (put each written line "
              f"on its own line in the transcript)")
        return None

    for i, ((image, _starts_paragraph), text) in enumerate(zip(crops, text_lines)):
        file_name = f"{prefix}_{i}.png"
        image.save(os.path.join(CROPS_DIR, file_name))
        labels.append((file_name, text))
    return len(crops)


def supabase_pages():
    import requests
    for row in fetch_rows():
        def load(url=row["image_url"]):
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.content
        yield f"db{row['id']}", load, row["corrected_text"]


def folder_pages(folder):
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        stem, ext = os.path.splitext(path)
        if ext.lower() not in IMAGE_TYPES or not os.path.exists(stem + ".txt"):
            continue
        with open(stem + ".txt", encoding="utf-8", errors="replace") as f:
            transcript = f.read()

        def load(p=path):
            with open(p, "rb") as f:
                return f.read()
        yield f"page_{os.path.basename(stem)}", load, transcript


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--pages", help="folder of page images + .txt transcripts "
                                        "(default: OCR Labeler samples from Supabase)")
    args = parser.parse_args()

    os.makedirs(CROPS_DIR, exist_ok=True)

    # Keep earlier exports, but replace any previous rows from the same pages.
    pages = list(folder_pages(args.pages) if args.pages else supabase_pages())
    prefixes = {prefix + "_" for prefix, _, _ in pages}
    labels = []
    if os.path.exists(LABELS_PATH):
        with open(LABELS_PATH, newline="", encoding="utf-8") as f:
            labels = [(r["file_name"], r["text"]) for r in csv.DictReader(f)
                      if not any(r["file_name"].startswith(p) for p in prefixes)]
    kept = len(labels)
    print(f"[export] {len(pages)} pages to process")

    exported = skipped = crops_written = 0
    for prefix, load, transcript in pages:
        try:
            written = export_page(prefix, load(), transcript, labels)
        except Exception as e:
            print(f"[export] {prefix}: error — {e}")
            skipped += 1
            continue
        if written is None:
            skipped += 1
        else:
            exported += 1
            crops_written += written

    with open(LABELS_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["file_name", "text"])
        writer.writerows(labels)

    print(f"Exported {crops_written} line crops from {exported} pages. "
          f"Skipped {skipped} pages.")
    print(f"{LABELS_PATH} now has {len(labels)} lines ({kept} from earlier exports).")


if __name__ == "__main__":
    main()
