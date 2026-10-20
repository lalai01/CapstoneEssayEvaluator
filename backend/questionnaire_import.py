"""
Read a questionnaire from a Word (.docx) or PDF file and turn it into survey
questions for the survey editor. Nothing is saved here: the admin reviews the
result (and can change any question's type) in the editor.

Like the rubric import, two readers, in order:
  1. a layout reader that follows the document itself: "Part I." headings
     become tabs, "A." headings sub-headings, "Label: ☐ A ☐ B" a multiple-
     choice question, "Label: ____" a short answer, a lone "☐ I agree…" a
     consent box, numbered questions followed by blank lines an essay, and
     rating tables (Statement | 5 4 3 2 1, or Statement | one 5-1 group per
     AI model) agreement scales / per-model grids;
  2. an AI reader (Gemini, then DeepSeek) when the layout gives too little,
     e.g. a scanned or unusual PDF.

Questions use the stored shapes of the survey tables (question_type text,
essay, radio, checkbox or rating, with a marker in options[0] for headings,
scales and grids; see frontend/src/lib/surveyKit.js).
"""
import base64
import io
import json
import os
import re

import requests

BOX = "☐□❑▢◻⬜○"            # checkbox glyphs used for options
BOX_RE = re.compile(f"[{BOX}]")
BLANK_RE = re.compile(r"_{3,}")
PART_RE = re.compile(r"^part\s+[ivx]+\b", re.I)
SUB_RE = re.compile(r"^[A-Z]\.\s+\S")
NUMBERED_RE = re.compile(r"^\d+[.)]\s+(.+)$")
SCALE_RE = re.compile(r"^[1-5]$")
MIN_QUESTIONS = 5           # fewer answerable items than this: try the AI reader


# ---------- stored shapes ----------
def section(title, text=""):
    return {"question": title, "question_type": "text", "options": ["::section", text], "is_required": False}


def subsection(title, text=""):
    return {"question": title, "question_type": "text", "options": ["::subsection", text], "is_required": False}


def short_answer(label):
    numeric = re.search(r"\b(age|years?|number|no\. of)\b", label, re.I)
    return {"question": label, "question_type": "text", "options": ["::number"] if numeric else None, "is_required": True}


def choice(label, options, multiple=False):
    return {"question": label, "question_type": "checkbox" if multiple else "radio", "options": options, "is_required": True}


def consent(text):
    return {"question": "Consent", "question_type": "checkbox", "options": [text], "is_required": True}


def essay(text):
    return {"question": text, "question_type": "essay", "options": None, "is_required": True}


def likert(text):
    return {"question": text, "question_type": "rating", "options": ["::likert"], "is_required": True}


def matrix(text, models):
    return {"question": text, "question_type": "rating", "options": ["::matrix", *models], "is_required": True}


def _clean(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _is_heading_kind(q):
    return isinstance(q.get("options"), list) and q["options"][:1] in (["::section"], ["::subsection"])


def answerable(questions):
    return [q for q in questions if not _is_heading_kind(q)]


# ---------- document -> ordered blocks ----------
def docx_blocks(data):
    """[('p', text) | ('table', [[cell text]])] in document order."""
    import docx
    from docx.table import Table
    from docx.text.paragraph import Paragraph
    document = docx.Document(io.BytesIO(data))
    blocks = []
    for child in document.element.body.iterchildren():
        tag = child.tag.rsplit("}", 1)[-1]
        if tag == "p":
            text = Paragraph(child, document).text
            if _clean(text):
                blocks.append(("p", text))
        elif tag == "tbl":
            table = Table(child, document)
            rows = [[_clean(cell.text) for cell in row.cells] for row in table.rows]
            blocks.append(("table", rows))
    return blocks


def pdf_blocks(data):
    """Paragraph blocks and tables per page, in reading order (text inside a
    table's area is taken from the table, not repeated as paragraphs)."""
    import fitz
    blocks = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            tables = page.find_tables().tables
            areas = [fitz.Rect(t.bbox) for t in tables]
            items = [(t.bbox[1], ("table", [[_clean(c) for c in row] for row in t.extract()])) for t in tables]
            for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
                if any(fitz.Rect(x0, y0, x1, y1).intersects(a) for a in areas):
                    continue
                for line in text.split("\n\n"):
                    if _clean(line):
                        items.append((y0, ("p", line)))
            blocks.extend(b for _, b in sorted(items, key=lambda it: it[0]))
    return blocks


# ---------- layout reader ----------
def _table_questions(rows):
    """A rating table: header 'No. | Statement | …'. Scale columns 5..1 give
    agreement-scale items; named column groups (e.g. AI models, each over a
    5..1 sub-header) give per-model grids."""
    if not rows:
        return []
    head_at = next((i for i, r in enumerate(rows) if any(c.lower().startswith("statement") for c in r)), None)
    if head_at is None:
        return []
    header = rows[head_at]
    stmt_col = next(i for i, c in enumerate(header) if c.lower().startswith("statement"))
    groups = []                     # named column groups after the statement
    for c in header[stmt_col + 1:]:
        if c and not SCALE_RE.match(c) and (not groups or groups[-1] != c):
            groups.append(c)
    out = []
    for row in rows[head_at + 1:]:
        cells = row + [""] * (len(header) - len(row))
        text = cells[stmt_col] if stmt_col < len(cells) else ""
        rest = [c for i, c in enumerate(cells) if i != stmt_col and c not in (header[i] if i < len(header) else "",)]
        if not text or text == header[stmt_col] or all(SCALE_RE.match(c) or not c for c in rest):
            continue                # the 5 4 3 2 1 sub-header row (merged header cells repeat)
        out.append(matrix(text, groups) if len(groups) >= 2 else likert(text))
    return out


def layout_questions(blocks):
    """Questions from the document's own layout. Returns (questions, title)."""
    questions = []
    title = None
    pending_essay = None            # a numbered question waiting for its blank lines

    def last_heading():
        return questions[-1] if questions and _is_heading_kind(questions[-1]) else None

    def add_text(text):
        head = last_heading()
        if head:
            head["options"][1] = (head["options"][1] + "\n\n" + text).strip()
        else:
            questions.append(subsection("", text))

    for kind, value in blocks:
        if kind == "table":
            if pending_essay:
                questions.append(essay(pending_essay))
                pending_essay = None
            questions.extend(_table_questions(value))
            continue
        raw = value
        text = _clean(raw)
        if not text:
            continue
        if BLANK_RE.fullmatch(text.replace(" ", "")):          # answer lines only
            if pending_essay:
                questions.append(essay(pending_essay))
                pending_essay = None
            continue
        if pending_essay:                                     # numbered line without blanks: still a question
            questions.append(essay(pending_essay))
            pending_essay = None

        if BOX_RE.search(text):
            label, *opts = BOX_RE.split(text)
            label = _clean(label).rstrip(":").strip()
            opts = [_clean(o) for o in opts if _clean(o)]
            if not label and len(opts) == 1:
                questions.append(consent(opts[0]))
            elif opts:
                multiple = bool(re.search(r"check all|select all|all that apply", label, re.I))
                questions.append(choice(label or "Choose one", opts, multiple))
            continue
        blank = re.match(r"^(.{1,80}?):\s*_{3,}", text)
        if blank:
            questions.append(short_answer(_clean(blank.group(1))))
            continue
        if PART_RE.match(text):
            questions.append(section(text))
            continue
        if SUB_RE.match(text) and len(text) <= 90 and not text.endswith("."):
            questions.append(subsection(text))
            continue
        numbered = NUMBERED_RE.match(text)
        if numbered and text.endswith("?"):
            pending_essay = numbered.group(1)
            continue
        # A short title-like line opens a tab (e.g. "Confidentiality and
        # Voluntary Participation"); "General Instructions:" a sub-heading;
        # anything else is explanatory text under the latest heading.
        if len(text) <= 70 and not text.endswith((".", ":", "?")) and text[:1].isupper() and len(text.split()) <= 9:
            if title is None and not questions:
                title = text
            questions.append(section(text))
            continue
        if text.endswith(":") and len(text) <= 60:
            questions.append(subsection(text.rstrip(":")))
            continue
        add_text(text)

    if pending_essay:
        questions.append(essay(pending_essay))
    # Drop empty sub-headings that only carried text right after a tab title.
    return [q for q in questions if not (_is_heading_kind(q) and not q["question"] and not q["options"][1])], title


# ---------- AI reader ----------
AI_INSTRUCTIONS = """Extract the questionnaire from this document, keeping its order and wording.
- "sections": each major part (e.g. "Part I. Respondent Profile") with its title and any
  instructions under it ("description").
- Inside a section, "items" in order. kind is one of:
  subheading (a lettered group such as "A. Performance Expectancy", text in "description"),
  consent (a single "I agree" checkbox), text (short answer), number (e.g. age),
  radio (pick one: give "options"), checkbox (pick any: give "options"), essay (long answer),
  likert (a statement rated 5..1 Strongly Agree..Strongly Disagree),
  matrix (a statement rated 5..1 separately for several items such as AI models: put the
  item names in "options").
- Copy statements and options exactly. Ignore page numbers, blank answer lines and
  the "5 4 3 2 1" column headings."""

AI_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "description": {"type": "STRING"},
        "sections": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
            "title": {"type": "STRING"},
            "description": {"type": "STRING"},
            "items": {"type": "ARRAY", "items": {"type": "OBJECT", "properties": {
                "kind": {"type": "STRING"},
                "text": {"type": "STRING"},
                "description": {"type": "STRING"},
                "options": {"type": "ARRAY", "items": {"type": "STRING"}},
            }, "required": ["kind", "text"]}},
        }, "required": ["title", "items"]}},
    },
    "required": ["title", "sections"],
}


def _from_ai(result):
    questions = []
    for sec in result.get("sections") or []:
        questions.append(section(_clean(sec.get("title")) or "Section", (sec.get("description") or "").strip()))
        for it in sec.get("items") or []:
            kind = (it.get("kind") or "").lower()
            text = _clean(it.get("text"))
            opts = [_clean(o) for o in (it.get("options") or []) if _clean(o)]
            if not text:
                continue
            if kind == "subheading":
                questions.append(subsection(text, (it.get("description") or "").strip()))
            elif kind == "consent":
                questions.append(consent(text))
            elif kind == "number":
                questions.append({**short_answer(text), "options": ["::number"]})
            elif kind in ("radio", "checkbox") and opts:
                questions.append(choice(text, opts, kind == "checkbox"))
            elif kind == "essay":
                questions.append(essay(text))
            elif kind == "likert":
                questions.append(likert(text))
            elif kind == "matrix" and len(opts) >= 2:
                questions.append(matrix(text, opts))
            else:
                questions.append(short_answer(text))
    return questions


def _gemini(parts):
    import gemini_ocr
    if not gemini_ocr.enabled():
        raise RuntimeError("Gemini isn't configured")
    body = {"contents": [{"parts": parts}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json", "responseSchema": AI_SCHEMA}}
    errors = []
    for model in gemini_ocr._models():
        resp = requests.post(gemini_ocr.API_URL.format(model=model), json=body, timeout=180,
                             headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"].strip()})
        if resp.status_code != 200:
            errors.append(f"{model}: HTTP {resp.status_code}")
            continue
        candidate = resp.json()["candidates"][0]
        return json.loads("".join(p.get("text", "") for p in candidate["content"]["parts"]))
    raise RuntimeError("; ".join(errors))


def _deepseek(text):
    from ai_models import call_deepseek
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DeepSeek isn't configured")
    shape = ('{"title": "...", "description": "...", "sections": [{"title": "...", "description": "...", '
             '"items": [{"kind": "radio", "text": "...", "options": ["..."], "description": ""}]}]}')
    reply = call_deepseek("You convert questionnaires into JSON. Reply with a single JSON object only.",
                          f"{AI_INSTRUCTIONS}\n\nReply with JSON shaped like:\n{shape}\n\nQuestionnaire:\n\n{text[:30000]}",
                          json_mode=True, temperature=0.0)["text"]
    return json.loads(reply)


def _blocks_text(blocks):
    return "\n".join(v if k == "p" else "\n".join(" | ".join(r) for r in v) for k, v in blocks)


# ---------- entry point ----------
def import_questionnaire(data, filename):
    suffix = os.path.splitext(filename or "")[1].lower()
    if suffix not in (".docx", ".pdf"):
        raise ValueError("Upload the questionnaire as a Word file (.docx) or a PDF.")
    blocks = docx_blocks(data) if suffix == ".docx" else pdf_blocks(data)
    name = os.path.splitext(os.path.basename(filename))[0]
    questions, title = layout_questions(blocks)
    warnings, source = [], "layout"

    if len(answerable(questions)) < MIN_QUESTIONS:
        attempts = []
        readers = []
        if suffix == ".pdf":
            readers.append(("gemini", lambda: _gemini([
                {"inline_data": {"mime_type": "application/pdf", "data": base64.b64encode(data).decode()}},
                {"text": AI_INSTRUCTIONS}])))
        else:
            readers.append(("gemini", lambda: _gemini([{"text": AI_INSTRUCTIONS + "\n\nQuestionnaire:\n\n" + _blocks_text(blocks)[:30000]}])))
        readers.append(("deepseek", lambda: _deepseek(_blocks_text(blocks))))
        for reader, read in readers:
            try:
                result = read()
                ai_questions = _from_ai(result)
                if len(answerable(ai_questions)) > len(answerable(questions)):
                    questions, source = ai_questions, reader
                    title = _clean(result.get("title")) or title
                break
            except Exception as e:
                attempts.append(f"{reader}: {str(e)[:120]}")
        if attempts:
            print(f"[questionnaire import] AI readers: {attempts}")

    if not answerable(questions):
        raise ValueError("No questions could be found in this file. Check that it is the questionnaire, "
                         "or build the survey by hand.")
    kinds = {}
    for q in answerable(questions):
        key = q["options"][0][2:] if isinstance(q.get("options"), list) and str(q["options"][0]).startswith("::") else q["question_type"]
        kinds[key] = kinds.get(key, 0) + 1
    if any(q["question"] == "Choose one" for q in questions):
        warnings.append("Some multiple-choice questions had no label; they are named “Choose one”. Please rename them.")
    # Grids that rate AI models the app doesn't run (e.g. an older draft).
    rated = {m for q in questions if (q.get("options") or [None])[0] == "::matrix" for m in q["options"][1:]}
    unknown = sorted(m for m in rated if not re.search(r"deepseek|gpt-oss|qwen|llama|groq|gemini", m, re.I))
    if unknown:
        warnings.append(f"The grids rate {', '.join(unknown)}, which the app doesn't use now. "
                        "Use “🔁 Rename AI models in this tab” to match the AIs respondents will see.")
    # The file name is the clearest title (a document's first line is often a notice).
    return {"title": name if source == "layout" else (title or name), "questions": questions,
            "source": source, "counts": kinds, "warnings": warnings}
