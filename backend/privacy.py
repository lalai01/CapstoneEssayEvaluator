"""
Personal data in essays (Data Privacy Act of 2012, RA 10173).

An essay with a student's name, grade and section is information about the
student's education: sensitive personal information (Sec. 3(l)). Grading it
doesn't need who wrote it, so for a minor's essay the identifying details are
removed before the text reaches any AI or the database (proportionality,
Sec. 11). The browser does the same first (frontend/src/lib/redact.js); this
is the server-side safety net, with the same rules.
"""
import re

REDACTED = "[REDACTED]"
HEADER_LINES = 8   # the heading of a school paper is in its first lines

# Header lines on school papers: "Name: Juan Dela Cruz", "Grade & Section: 7-Rizal",
# "LRN: 123456789012", "Pangalan: ...", "Teacher: ...".
_HEADER_LABELS = (
    r"name|full\s*name|student(?:'s)?\s*name|learner(?:'s)?\s*name|pangalan|ngalan|"
    r"grade\s*(?:level)?\s*(?:&|and|/)?\s*section|gr\.?\s*(?:&|/)?\s*sec\.?|year\s*(?:&|and)\s*section|"
    r"section|seksyon|grade|grado|lrn|learner\s*reference\s*(?:number|no\.?)|student\s*(?:id|no\.?|number)|"
    r"id\s*(?:no\.?|number)|school|paaralan|eskwelahan|teacher|guro|adviser|tagapayo|"
    r"address|tirahan|birthday|birth\s*date|age|edad|contact\s*(?:no\.?|number)|parent|guardian|magulang"
).replace(r"\s", r"[ \t]")   # a label never spans two lines
HEADER_RE = re.compile(rf"^([ \t]*(?:{_HEADER_LABELS})[ \t]*[:\-–][ \t]*)(\S[^\n]*?)[ \t]*$",
                       re.IGNORECASE | re.MULTILINE)
LRN_RE = re.compile(r"(?<!\d)\d{12}(?!\d)")                       # DepEd Learner Reference Number
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?63|0)9\d{2}[\s-]?\d{3}[\s-]?\d{4}(?!\d)")


def redact_personal_info(text, names=None):
    """(redacted_text, findings). findings: [{"type", "text"}] of what was
    removed. `names` are people's names to remove wherever they appear (the
    whole name and each part of 3+ letters)."""
    if not text:
        return text or "", []
    findings = []

    def header(m):
        findings.append({"type": "header", "text": m.group(0).strip()})
        return f"{m.group(1)}{REDACTED}"
    # Only the paper's heading (its first lines), so an essay sentence that
    # happens to start with "School - ..." is left alone.
    lines = text.split("\n")
    head_end, seen = 0, 0
    for i, line in enumerate(lines):
        head_end = i + 1
        if line.strip():
            seen += 1
            if seen >= HEADER_LINES:
                break
    head = HEADER_RE.sub(header, "\n".join(lines[:head_end]))
    text = "\n".join([head] + lines[head_end:])

    for kind, rx, label in (("lrn", LRN_RE, "[LRN REDACTED]"), ("email", EMAIL_RE, "[EMAIL REDACTED]"),
                            ("phone", PHONE_RE, "[PHONE REDACTED]")):
        def sub(m, kind=kind, label=label):
            findings.append({"type": kind, "text": m.group(0)})
            return label
        text = rx.sub(sub, text)

    parts = set()
    for name in names or []:
        name = (name or "").strip()
        if not name:
            continue
        parts.add(name)
        parts.update(p for p in re.split(r"[\s,.]+", name) if len(p) >= 3)
    for part in sorted(parts, key=len, reverse=True):
        rx = re.compile(rf"(?<![\w\[]){re.escape(part)}(?![\w\]])", re.IGNORECASE)
        def sub(m):
            findings.append({"type": "name", "text": m.group(0)})
            return REDACTED
        text = rx.sub(sub, text)
    return text, findings
