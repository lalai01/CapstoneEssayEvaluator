"""
Essays written partly or wholly in another language (Tagalog/Filipino,
Cebuano/Bisaya, Ilocano, ...) are translated into English before they are
evaluated, since the rubric, the models' prompts and the grammar checker all
work in English.

Only the non-English parts are translated; English parts are copied as the
student wrote them, mistakes included, because grammar and mechanics are graded.
A plain English essay never reaches an AI: the local dictionary check
(text_quality) decides first.
"""
import re

from text_quality import english_quality

# English essays score 0.93-1.00 on the dictionary check (see
# evaluator.is_valid_essay); below this, ask an AI whether anything needs translating.
ENGLISH_RATIO = 0.9
PROVIDERS = ("gemini", "deepseek", "groq")

SYSTEM = """You translate student essays into English for a teacher who grades them.
The essay may mix English with Tagalog/Filipino, Cebuano/Bisaya, Ilocano,
Hiligaynon or any other language (for example "Taglish").

Rules:
- Translate every non-English word, phrase and sentence into natural English,
  faithfully and completely. Do not summarize, add, omit or improve ideas.
- Copy the parts already in English EXACTLY as written, keeping their spelling,
  grammar and punctuation mistakes. Never correct the student's English.
- Keep names of people, places, schools, foods and events as written.
- Keep the title and the paragraph breaks.

Answer in exactly this format:
LANGUAGES: <languages found, comma-separated, e.g. Tagalog, English>
TRANSLATED: <yes if you translated anything, otherwise no>
---
<the full essay in English>"""


def needs_translation(text):
    q = english_quality(text)
    return q["words"] > 0 and q["ratio"] < ENGLISH_RATIO


def _parse(reply):
    head, sep, body = reply.partition("\n---")
    if not sep:
        raise ValueError("the reply is not in the expected format")
    langs = re.search(r"LANGUAGES:\s*(.+)", head, re.I)
    done = re.search(r"TRANSLATED:\s*(\w+)", head, re.I)
    languages = [s.strip() for s in (langs.group(1) if langs else "").split(",") if s.strip()]
    return languages, bool(done and done.group(1).lower().startswith("y")), body.strip()


def translate_to_english(text):
    """
    {"translated": bool, "text": English text to evaluate, "languages": [...],
     "provider": AI used or None, "error": message when every AI failed}.
    When nothing needed translating, "text" is the original, unchanged.
    """
    original = (text or "").strip()
    if not needs_translation(original):
        return {"translated": False, "text": original, "languages": ["English"], "provider": None}

    from ai_models import call_cloud
    errors = []
    for provider in PROVIDERS:
        try:
            reply = call_cloud(provider, SYSTEM, f"Essay:\n\"\"\"\n{original}\n\"\"\"",
                               temperature=0.1, max_tokens=8192)
            languages, translated, english = _parse(reply["text"])
            other = [l for l in languages if l.lower() != "english"]
            if not translated or not other or not english:
                # Nothing to translate after all (e.g. many local names): keep the student's text.
                return {"translated": False, "text": original, "languages": languages or ["English"],
                        "provider": reply.get("provider")}
            # A cut-off reply would drop part of the essay from grading.
            if len(english.split()) < 0.6 * len(original.split()):
                raise ValueError("the translation is much shorter than the essay")
            return {"translated": True, "text": english, "languages": languages,
                    "provider": reply.get("provider"), "model": reply.get("model")}
        except Exception as e:
            errors.append(f"{provider}: {str(e)[:120]}")
    return {"translated": False, "text": original, "languages": [], "provider": None,
            "error": "Translation is unavailable right now (" + "; ".join(errors) + ")."}
