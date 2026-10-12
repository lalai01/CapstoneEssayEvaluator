"""
How much of an OCR result is real English.

Every essay this app handles is an English paragraph, so the share of
recognised dictionary words is a reliable way to
  - choose between the printed-text reader (Tesseract) and the handwriting
    reader (TrOCR): whichever reads more real words wins, and
  - reject uploads that are not essays (a face, a car, a blank page), which
    produce few or no real words.
"""
import re
from functools import lru_cache

WORD_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")

# Below these, an upload is treated as "not an essay".
MIN_WORDS = 8
MIN_ENGLISH_RATIO = 0.5
# Models shown a non-text image tend to loop ("the the the ..."); real
# paragraphs use many different words.
MIN_DISTINCT_RATIO = 0.35


@lru_cache(maxsize=1)
def _speller():
    from spellchecker import SpellChecker
    return SpellChecker(language="en")


def english_quality(text):
    """
    {"words": word tokens, "known": dictionary words, "ratio": known/words,
     "distinct": distinct words / words}.
    Single letters only count as words when they are "a" or "i".
    """
    tokens = [t.lower() for t in WORD_RE.findall(text or "")]
    tokens = [t for t in tokens if len(t) > 1 or t in ("a", "i")]
    if not tokens:
        return {"words": 0, "known": 0, "ratio": 0.0, "distinct": 0.0}
    plain = [t.replace("'", "") for t in tokens]
    known_set = _speller().known(plain)
    known = sum(1 for t in plain if t in known_set)
    return {"words": len(tokens), "known": known, "ratio": known / len(tokens),
            "distinct": len(set(tokens)) / len(tokens)}


def looks_like_english_text(text):
    q = english_quality(text)
    return (q["words"] >= MIN_WORDS and q["ratio"] >= MIN_ENGLISH_RATIO
            and q["distinct"] >= MIN_DISTINCT_RATIO)
