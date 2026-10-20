"""
Evaluation activity for admins: every essay a signed-in user evaluated, and
what happened to it afterwards.

Each AI run is logged in the evaluation_runs table (see README). Runs of the
same essay by the same person are grouped by a fingerprint of the essay text,
which also matches them to that person's Knowledge Base saves and overrides,
so each essay gets a status:

    draft      evaluated, never saved
    saved      saved to the Knowledge Base (not approved)
    approved   saved by a teacher (approved; used by the AI's learning)
    overridden scores corrected by a non-teacher
    rejected   a teacher rejected the AI's result and overrode it
"""
import hashlib
import re

STATUS_ORDER = ["rejected", "overridden", "approved", "saved", "draft"]
STATUS_LABELS = {
    "draft": "Draft (not saved)",
    "saved": "Saved",
    "approved": "Approved (teacher)",
    "overridden": "Overridden",
    "rejected": "Rejected & overridden (teacher)",
}
PREVIEW = 600


def essay_hash(text):
    """Fingerprint of an essay: the same text (ignoring case and spacing)
    always gives the same value."""
    normal = re.sub(r"\s+", " ", (text or "").lower()).strip()
    return hashlib.sha1(normal.encode("utf-8")).hexdigest()[:20] if normal else ""


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def overall(scores):
    """(text, percent) for a scores object: '16/20' for holistic points,
    '78%' for analytic scores. (None, None) when it can't be computed."""
    if not isinstance(scores, dict):
        return None, None
    if _num(scores.get("holistic_points")) is not None and _num(scores.get("holistic_max")):
        p, m = scores["holistic_points"], scores["holistic_max"]
        return f"{p}/{m}", round(100 * (p - 1) / (m - 1), 1) if m > 1 else None
    if _num(scores.get("holistic_score")) is not None:
        h = scores["holistic_score"]
        return f"{h}/5", round(100 * (h - 1) / 4, 1)
    meta = scores.get("criteria_meta") or []
    total = weight = 0.0
    for c in meta:
        v, top = _num(scores.get(c.get("key"))), c.get("max_score") or 0
        if v is None or top <= 1:
            continue
        w = c.get("weight", 1) or 0
        total += w * (v - 1) / (top - 1)
        weight += w
    if not weight:
        return None, None
    pct = round(100 * total / weight, 1)
    return f"{round(pct)}%", pct


def entry_scores(row):
    """A Knowledge Base row's full scores (newer column, or nested)."""
    return row.get("scores") or (row.get("criterion_feedback") or {}).get("scores") or {}


def override_scores(row):
    """(AI's scores, the user's corrected scores) of an override row."""
    original = row.get("original_scores") or {}
    teacher = row.get("teacher_scores") or original.get("teacher_scores") or {}
    corrected = {**original, **{k: v for k, v in teacher.items() if _num(v) is not None}}
    return original, corrected if teacher else None


def _title(row, *keys):
    for k in keys:
        v = row.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def build_activity(runs, kb, overrides, people):
    """
    runs: evaluation_runs rows; kb: knowledge_base rows; overrides:
    learning_feedback rows; people: {user_id: {"email", "name", "role"}}.
    Returns {"essays", "overrides", "saved", "counts"} for the admin page.
    """
    def person(uid, fallback_email=None, fallback_role=None):
        p = people.get(uid) or {}
        return {"user_id": uid,
                "email": p.get("email") or fallback_email,
                "name": p.get("name"),
                "role": p.get("role") or fallback_role or "unknown"}

    groups = {}

    def group(uid, h, essay):
        key = f"{uid}:{h}"
        if key not in groups:
            groups[key] = {"key": key, "user_id": uid, "essay_hash": h, "title": None,
                           "essay": (essay or "")[:PREVIEW], "essay_length": len((essay or "").split()),
                           "runs": [], "kb": [], "overrides": [], "first_at": None, "last_at": None}
        return groups[key]

    def touch(g, at):
        if at:
            g["first_at"] = min(filter(None, [g["first_at"], at]))
            g["last_at"] = max(filter(None, [g["last_at"], at]))

    for r in runs:
        h = r.get("essay_hash") or essay_hash(r.get("essay"))
        g = group(r.get("user_id"), h, r.get("essay"))
        g["title"] = g["title"] or _title(r, "title")
        g["runs"].append({"id": r.get("id"), "at": r.get("created_at"), "ai": r.get("ai_label") or r.get("ai_choice"),
                          "eval_type": r.get("eval_type"), "overall": r.get("overall"),
                          "rubric": r.get("rubric_title"), "rag": r.get("used_rag")})
        touch(g, r.get("created_at"))
        people.setdefault(r.get("user_id"), {"email": r.get("user_email"), "role": r.get("user_role")})

    saved_list = []
    for row in kb:
        scores = entry_scores(row)
        text, pct = overall(scores)
        approved = bool(row.get("approved_by") or (row.get("criterion_feedback") or {}).get("approved_by"))
        item = {"id": row.get("id"), "at": row.get("created_at"),
                "title": _title(row, "title") or _title(scores, "title") or "Untitled essay",
                "ai": row.get("ai_model") or (row.get("criterion_feedback") or {}).get("ai_model") or (scores.get("evaluated_by") or {}).get("label"),
                "eval_type": row.get("eval_type"), "overall": text, "overall_pct": pct,
                "approved": approved, "approved_by": row.get("approved_by"),
                "satisfaction": row.get("satisfaction"), **person(row.get("user_id"))}
        saved_list.append(item)
        g = group(row.get("user_id"), essay_hash(row.get("essay")), row.get("essay"))
        g["title"] = g["title"] or item["title"]
        g["kb"].append({k: item[k] for k in ("id", "at", "ai", "overall", "approved", "approved_by", "satisfaction")})
        touch(g, row.get("created_at"))

    override_list = []
    for row in overrides:
        ai_scores, corrected = override_scores(row)
        ai_text, ai_pct = overall(ai_scores)
        new_text, new_pct = overall(corrected) if corrected else (None, None)
        teacher = bool(row.get("approved_by") or ai_scores.get("approved_by"))
        meta = ai_scores.get("criteria_meta") or []
        changes = []
        for c in meta:
            a, b = ai_scores.get(c.get("key")), (corrected or {}).get(c.get("key"))
            if _num(a) is not None and _num(b) is not None and a != b:
                changes.append({"label": c.get("label"), "ai": a, "new": b, "max": c.get("max_score")})
        if corrected and _num(ai_scores.get("holistic_points")) is not None and corrected.get("holistic_points") != ai_scores.get("holistic_points"):
            changes.append({"label": "Holistic score", "ai": ai_scores.get("holistic_points"),
                            "new": corrected.get("holistic_points"), "max": ai_scores.get("holistic_max")})
        item = {"id": row.get("id"), "at": row.get("created_at"),
                "title": _title(row, "title") or _title(ai_scores, "title") or "Untitled essay",
                "ai": row.get("ai_model") or ai_scores.get("ai_model") or (ai_scores.get("evaluated_by") or {}).get("label"),
                "ai_overall": ai_text, "ai_pct": ai_pct, "new_overall": new_text, "new_pct": new_pct,
                "changes": changes, "remarks": row.get("teacher_feedback"), "suggestions": row.get("suggested_changes"),
                "kind": "rejected" if teacher else "overridden", "approved_by": row.get("approved_by"),
                "satisfaction": row.get("satisfaction"), **person(row.get("user_id"))}
        override_list.append(item)
        g = group(row.get("user_id"), essay_hash(row.get("original_essay")), row.get("original_essay"))
        g["title"] = g["title"] or item["title"]
        g["overrides"].append({k: item[k] for k in ("id", "at", "kind", "ai", "ai_overall", "new_overall")})
        touch(g, row.get("created_at"))

    essays = []
    for g in groups.values():
        flags = {"rejected": any(o["kind"] == "rejected" for o in g["overrides"]),
                 "overridden": any(o["kind"] == "overridden" for o in g["overrides"]),
                 "approved": any(k["approved"] for k in g["kb"]),
                 "saved": bool(g["kb"])}
        status = next((s for s in STATUS_ORDER[:-1] if flags[s]), "draft")
        g["runs"].sort(key=lambda r: r["at"] or "")
        essays.append({**g, **person(g["user_id"]), "title": g["title"] or "Untitled essay",
                       "status": status, "status_label": STATUS_LABELS[status], "flags": flags,
                       "ais": sorted({r["ai"] for r in g["runs"] if r["ai"]})})
    essays.sort(key=lambda e: e["last_at"] or "", reverse=True)
    saved_list.sort(key=lambda e: e["at"] or "", reverse=True)
    override_list.sort(key=lambda e: e["at"] or "", reverse=True)
    counts = {s: sum(1 for e in essays if e["status"] == s) for s in STATUS_ORDER}
    return {"essays": essays, "overrides": override_list, "saved": saved_list, "counts": counts,
            "status_labels": STATUS_LABELS}
