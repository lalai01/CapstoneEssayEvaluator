"""
How accurately each reader reads handwriting, measured on the OCR Labeler's
verified pages: what the reader produced (raw_ocr_text, before any AI
correction) against what an admin confirmed is on the page (corrected_text).

Character error rate (CER) and word error rate (WER) are edit distances
divided by the length of the verified text, summed over all pages of a
reader (so long pages count more than short ones). Line breaks and spacing
are ignored; case and punctuation count.
"""
import re

import numpy as np


def _distance(a, b):
    """Levenshtein distance between two sequences, one numpy row at a time."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    codes = {}
    av = np.array([codes.setdefault(x, len(codes)) for x in a])
    bv = np.array([codes.setdefault(x, len(codes)) for x in b])
    idx = np.arange(len(bv) + 1)
    prev = idx.copy()
    for i, ch in enumerate(av, 1):
        base = np.empty_like(prev)
        base[0] = i
        # substitution or match, and deletion
        base[1:] = np.minimum(prev[:-1] + (bv != ch), prev[1:] + 1)
        # insertion: cur[j] = min(base[j], cur[j-1] + 1), done for the whole row
        prev = np.minimum.accumulate(base - idx) + idx
    return int(prev[-1])


def _normalise(text):
    return re.sub(r"\s+", " ", text or "").strip()


def page_errors(read_text, verified_text):
    """(char_edits, chars, word_edits, words) for one page."""
    hyp, ref = _normalise(read_text), _normalise(verified_text)
    hw, rw = hyp.split(), ref.split()
    return _distance(hyp, ref), len(ref), _distance(hw, rw), len(rw)


def summarise(rows):
    """
    rows: [{"engine", "raw_ocr_text", "corrected_text"}] for verified samples.
    Returns per-reader totals plus dataset size.
    """
    by_engine = {}
    lines = 0
    for r in rows:
        verified = r.get("corrected_text") or ""
        if not verified.strip():
            continue
        lines += sum(1 for line in verified.splitlines() if line.strip())
        ce, c, we, w = page_errors(r.get("raw_ocr_text") or "", verified)
        e = by_engine.setdefault(r.get("engine") or "unknown",
                                 {"pages": 0, "char_edits": 0, "chars": 0, "word_edits": 0, "words": 0})
        e["pages"] += 1
        e["char_edits"] += ce
        e["chars"] += c
        e["word_edits"] += we
        e["words"] += w

    readers = []
    for engine, e in sorted(by_engine.items(), key=lambda kv: -kv[1]["pages"]):
        readers.append({
            "engine": engine,
            "pages": e["pages"],
            "words": e["words"],
            "cer": round(e["char_edits"] / max(1, e["chars"]), 4),
            "wer": round(e["word_edits"] / max(1, e["words"]), 4),
        })
    return {"verified_pages": sum(r["pages"] for r in readers), "verified_lines": lines, "readers": readers}
