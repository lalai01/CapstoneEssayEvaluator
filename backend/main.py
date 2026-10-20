import os
import re
import uuid
import tempfile
import threading
import time
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, Any, List, Literal
import ocr_utils
import evaluator
from supabase_client import supabase, supabase_admin
from ai_models import test_prompt
from ocr_jobs import get_job_status
from auth import get_current_user, get_optional_user, security
from supabase import create_client, Client
from evaluation_graph import run_evaluation   # ✅ LangGraph
from evaluator import resolve_rubric

# ---------- FastAPI App ----------
app = FastAPI(title="AI Essay Evaluator API")


@app.on_event("startup")
def _preload_local_ai():
    # Load the local AI into memory in the background, so the first essay
    # after a restart doesn't wait for it (startup isn't delayed).
    from ai_models import warm_up_ollama
    threading.Thread(target=warm_up_ollama, args=(evaluator.LOCAL_SCORING_MODEL,), daemon=True).start()

# ---------- CORS Configuration ----------
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://capstoneessayevaluator.web.app",
    "https://essay-evaluator.duckdns.org",
    "https://api.essay-evaluator.duckdns.org",
    "https://essay-evaluator.duckdns.org",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Global exception handler – always adds CORS headers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    origin = request.headers.get("origin")
    headers = {}
    if origin in ALLOWED_ORIGINS:
        headers["Access-Control-Allow-Origin"] = origin
        headers["Access-Control-Allow-Credentials"] = "true"
    import traceback
    traceback.print_exc()
    return JSONResponse(
        status_code=500,
        content={"detail": str(exc)},
        headers=headers,
    )

# ---------- Supabase helper ----------
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

def get_user_client(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Client:
    """Return a Supabase client that carries the user's JWT."""
    token = credentials.credentials
    c = create_client(SUPABASE_URL, SUPABASE_KEY)
    c.postgrest.auth(token)
    return c

# ---------- Models ----------
class EvaluationRequest(BaseModel):
    text: str
    evaluation_type: str = "analytic"
    rubric_id: Optional[int] = None   # omit to use the teacher's default rubric
    # "auto" picks a model per essay; or force "deepseek" / "local" (= "llama")
    # / "llama" / "qwen" ("gemma" = older name) / "groq". The Evaluate page runs all three AIs side by side
    # with strict_model=True: each result is that AI's alone, with no fallback
    # to another model.
    ai_model: Literal["auto", "deepseek", "local", "llama", "qwen", "gemma", "groq"] = "auto"
    strict_model: bool = False

class EvaluationResponse(BaseModel):
    scores: dict
    feedback: str

class OCRResponse(BaseModel):
    text: str
    confidence: float
    method: str
    page_count: Optional[int] = None
    engine: Optional[str] = None
    low_confidence: bool = False
    word_count: Optional[int] = None

NOT_ESSAY_MESSAGE = ("This doesn't look like an essay. Please attach a clear photo, scan, PDF or "
                     "Word file of a written or printed English essay.")

def _not_essay(reason: str):
    """422 the frontend recognises (code "not_essay") and shows as a prompt."""
    raise HTTPException(422, {"code": "not_essay", "message": NOT_ESSAY_MESSAGE, "reason": reason})

class KnowledgeEntry(BaseModel):
    title: Optional[str] = None
    essay: str

    main_statement: Optional[int] = None
    organization: Optional[int] = None
    evidence: Optional[int] = None
    analysis: Optional[int] = None
    grammar: Optional[int] = None

    holistic_score: Optional[int] = None
    level_description: Optional[str] = None

    coherence: Optional[int] = None
    content: Optional[int] = None

    criterion_feedback: Optional[dict] = None
    grammar_issues: Optional[list] = None
    recommendations: Optional[list] = None

    feedback: str
    eval_type: str
    accepted: bool = False
    satisfaction: int = 5
    teacher_feedback: Optional[str] = None
    # Full /evaluate scores (any rubric's criteria, rubric used, evaluated_by)
    # and a readable model label, so entries can be filtered by AI.
    scores: Optional[Dict[str, Any]] = None
    ai_model: Optional[str] = None

# criteria accepts both the legacy shape ({key: {"4": "..."}}) and the
# configurable shape ({key: {"label", "max_score", "weight", "levels"}}).
class RubricCreate(BaseModel):
    title: str
    description: Optional[str] = None
    criteria: Dict[str, Dict[str, Any]]
    is_active: bool = True
    is_system: bool = False   # school-wide (admins only) vs the teacher's own

class RubricUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    criteria: Optional[Dict[str, Dict[str, Any]]] = None
    is_active: Optional[bool] = None

# original_scores is the full /evaluate `scores` object, which also carries
# criterion_feedback (dict), grammar_issues / recommendations (lists),
# level_description (str) and criteria_meta (list) — not just ints.
class OverrideRequest(BaseModel):
    title: Optional[str] = None
    ai_model: Optional[str] = None
    original_essay: str
    original_scores: Dict[str, Any]
    teacher_scores: Optional[Dict[str, Any]] = None
    teacher_feedback: str
    suggested_changes: Optional[str] = None
    accepted: bool = False
    satisfaction: Optional[int] = 5

class RevisionRequest(BaseModel):
    text: str
    grammar_issues: Optional[List[Dict[str, Any]]] = None
    recommendations: Optional[List[Dict[str, Any]]] = None
    ai_model: Literal["auto", "deepseek", "local"] = "auto"

class PromptTestRequest(BaseModel):
    ai_provider: str
    system_prompt: str
    user_prompt: str
    model: Optional[str] = None
    json_mode: bool = False   # ask the model for strict JSON output
    json_schema: Optional[Dict[str, Any]] = None   # exact structure (Ollama 0.5+)

class PromptTestResponse(BaseModel):
    result: Dict[str, Any]

class SavedEssayEntry(BaseModel):
    title: str
    essay: str

class RatingEntry(BaseModel):
    rating: int
    comment: Optional[str] = None

class CommentEntry(BaseModel):
    rating_id: int
    parent_id: Optional[int] = None
    body: str

class ReactionEntry(BaseModel):
    comment_id: int
    reaction_type: str

class SurveyCreate(BaseModel):
    title: str
    description: Optional[str] = None
    is_active: bool = True

class SurveyUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None

class QuestionCreate(BaseModel):
    survey_id: int
    question: str
    question_type: str
    options: Optional[List[str]] = None
    is_required: bool = True
    order_number: int = 0

class QuestionUpdate(BaseModel):
    question: Optional[str] = None
    question_type: Optional[str] = None
    options: Optional[List[str]] = None
    is_required: Optional[bool] = None
    order_number: Optional[int] = None

class SurveyResponseSubmit(BaseModel):
    survey_id: int
    answers: Dict[str, str]

class OCRCorrection(BaseModel):
    sample_id: int
    corrected_text: str

# ---------- Database client for private, per-user tables ----------
# Tables like knowledge_base, learning_feedback and saved_essays have
# row-level security on. Requests from the backend's anon key carry no user,
# so RLS rejects inserts and silently returns no rows. The backend verifies
# every user itself (auth.py) and always sets/filters user_id from that
# verified identity, so it uses the service-role key for these tables.
def _db():
    return supabase_admin or supabase

# ---------- Health ----------
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ---------- Admins ----------
# No admin emails live in the code. A user is an admin when any of these holds:
#   - their confirmed email is in the admin_users table in Supabase (seeded
#     with SQL, then managed in the app's Admins tab). Emails can be added
#     before the person ever signs in, and any sign-in method works.
#   - app_metadata.role == "admin" (set server-side in Supabase; users
#     cannot change app_metadata themselves)
#   - their confirmed email is in the optional ADMIN_EMAILS server setting
#     (environment variable on the VPS; empty by default)

def _builtin_admin_emails() -> List[str]:
    raw = os.environ.get("ADMIN_EMAILS", "")
    return [e.strip().lower() for e in raw.split(",") if e.strip()]

_ROLE_CACHE_SECONDS = 60
_role_cache = {}  # table -> {"at": float, "emails": set}

def _table_role_emails(table: str, refresh: bool = False) -> set:
    """Emails listed in a role table (admin_users or teachers), cached briefly."""
    import time
    now = time.time()
    cached = _role_cache.get(table)
    if not refresh and cached and now - cached["at"] < _ROLE_CACHE_SECONDS:
        return cached["emails"]
    if not supabase_admin:
        return set()
    try:
        if table == "teachers":
            emails = {r["email"] for r in _teacher_rows() if _teacher_status(r) == "active"}
        else:
            rows = supabase_admin.table(table).select("email").execute().data
            emails = {r["email"].lower() for r in rows if r.get("email")}
    except Exception as e:
        # A brief database hiccup must not turn admins into students: keep the
        # last good list and don't cache the failure, so the next request
        # tries again.
        print(f"⚠️ {table} lookup failed; using last known list: {e}")
        return cached["emails"] if cached else set()
    _role_cache[table] = {"at": now, "emails": emails}
    return emails


# ---------- Teacher records and the role activity log ----------
# A teacher row is never hard-deleted: "Remove" sets deleted_at/deleted_by
# (soft delete, restorable), and "Inactive" keeps the row but removes the
# teacher rights until an admin reactivates it. Every change is written to
# role_audit_log with the admin who made it.

def _teacher_rows() -> List[Dict[str, Any]]:
    try:
        rows = supabase_admin.table("teachers").select("*").order("created_at").execute().data
    except Exception as e:
        if "teachers" in str(e):
            raise HTTPException(500, "Database migration needed: create the teachers table (see README).")
        raise
    for r in rows:
        r["email"] = (r.get("email") or "").lower()
    return rows

def _teacher_status(row) -> str:
    """active | inactive | removed. Rows from before the migration count as active."""
    if row.get("deleted_at"):
        return "removed"
    return "inactive" if row.get("status") == "inactive" else "active"

def _log_role_change(role: str, email: str, action: str, actor: Optional[str], details: Optional[dict] = None):
    """Record who changed a role. Logging never blocks the change itself."""
    try:
        supabase_admin.table("role_audit_log").insert({
            "role": role, "email": email, "action": action,
            "actor": actor, "details": details or {},
        }).execute()
    except Exception as e:
        print(f"⚠️ role_audit_log write failed (run the README migration): {e}")

def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def _update_teacher(email: str, changes: dict):
    """Update a teacher row, with a clear message if the status columns are missing."""
    try:
        supabase_admin.table("teachers").update(changes).eq("email", email).execute()
    except Exception as e:
        if any(f"'{c}'" in str(e) for c in changes):
            raise HTTPException(500, "Database migration needed: add the teacher status columns (see README).")
        raise

def _table_admin_emails(refresh: bool = False) -> set:
    return _table_role_emails("admin_users", refresh)

def _confirmed_email(user) -> str:
    """The user's email if confirmed; unconfirmed emails could belong to anyone."""
    email = (user.get("email") or "").lower()
    return email if email and user.get("email_confirmed", True) else ""

def is_admin(user):
    if not user:
        return False
    if user.get("role") == "admin":
        return True
    email = _confirmed_email(user)
    return bool(email) and (email in _builtin_admin_emails() or email in _table_admin_emails())

# Teachers ("approvers") may override AI evaluations, approve evaluations
# into the shared Knowledge Base, and create their own rubrics. Admins run
# the system (roles, surveys, OCR labels, the school-wide rubric) and are NOT
# teachers unless they are also listed as one; one person can hold both
# roles. Everyone else signed in is a student: they can evaluate essays and
# save them privately, but their saves are not approved.
def is_teacher(user):
    if not user:
        return False
    if user.get("role") == "teacher":
        return True
    email = _confirmed_email(user)
    return bool(email) and email in _table_role_emails("teachers")

def role_of(user) -> str:
    return "admin" if is_admin(user) else "teacher" if is_teacher(user) else "student"

def _require_teacher(user, action="do this"):
    if not is_teacher(user):
        raise HTTPException(403, f"Only teachers can {action}. Ask an admin to add you as a teacher.")

class AdminEmail(BaseModel):
    email: str

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

def _require_admin(user):
    if not is_admin(user):
        raise HTTPException(403, "Admin only")

def _require_service_key():
    if not supabase_admin:
        raise HTTPException(500, "SUPABASE_SERVICE_KEY is not set on the server; admin management needs it.")

@app.get("/me")
def who_am_i(user: dict = Depends(get_current_user)):
    """Who the caller is and their role (drives the tabs and role tag)."""
    return {
        "id": user["id"],
        "email": user.get("email"),
        "role": role_of(user),
        "is_admin": is_admin(user),
        "is_teacher": is_teacher(user),
    }

@app.get("/admin/admins")
def list_admins(user: dict = Depends(get_current_user)):
    _require_admin(user)
    _require_service_key()
    try:
        rows = supabase_admin.table("admin_users").select("*").order("created_at").execute().data
    except Exception as e:
        if "admin_users" in str(e):
            raise HTTPException(500, "Database migration needed: create the admin_users table (see README).")
        raise
    builtin = [{"email": e, "builtin": True, "added_by": "server setting (ADMIN_EMAILS)"} for e in _builtin_admin_emails()]
    added = [{**r, "builtin": False} for r in rows if r["email"] not in _builtin_admin_emails()]
    return builtin + added

@app.post("/admin/admins")
def add_admin(entry: AdminEmail, user: dict = Depends(get_current_user)):
    _require_admin(user)
    _require_service_key()
    email = entry.email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise HTTPException(422, "Enter a valid email address.")
    supabase_admin.table("admin_users").upsert(
        {"email": email, "added_by": user.get("email")}, on_conflict="email"
    ).execute()
    _table_admin_emails(refresh=True)
    _log_role_change("admin", email, "added", user.get("email"))
    return {"status": "added", "email": email}

@app.delete("/admin/admins/{email}")
def remove_admin(email: str, user: dict = Depends(get_current_user)):
    _require_admin(user)
    _require_service_key()
    email = email.strip().lower()
    if email in _builtin_admin_emails():
        raise HTTPException(400, "This admin is set in the server's ADMIN_EMAILS setting and can't be removed here.")
    if email == (user.get("email") or "").lower():
        raise HTTPException(400, "You can't remove your own admin access.")
    supabase_admin.table("admin_users").delete().eq("email", email).execute()
    _table_admin_emails(refresh=True)
    _log_role_change("admin", email, "removed", user.get("email"))
    return {"status": "removed", "email": email}

class TeacherStatusUpdate(BaseModel):
    status: Literal["active", "inactive"]

@app.get("/admin/teachers")
def list_teachers(include_removed: bool = True, user: dict = Depends(get_current_user)):
    """All teacher records with their status: active, inactive or removed."""
    _require_admin(user)
    _require_service_key()
    rows = [{**r, "status": _teacher_status(r)} for r in _teacher_rows()]
    return rows if include_removed else [r for r in rows if r["status"] != "removed"]

@app.post("/admin/teachers")
def add_teacher(entry: AdminEmail, user: dict = Depends(get_current_user)):
    """Add a teacher, or restore one that was removed."""
    _require_admin(user)
    _require_service_key()
    email = entry.email.strip().lower()
    if not _EMAIL_RE.match(email):
        raise HTTPException(422, "Enter a valid email address.")
    existing = next((r for r in _teacher_rows() if r["email"] == email), None)
    actor = user.get("email")
    if existing is None:
        supabase_admin.table("teachers").insert({"email": email, "added_by": actor}).execute()
        action = "added"
    elif _teacher_status(existing) == "removed":
        _update_teacher(email, {"deleted_at": None, "deleted_by": None, "status": "active",
                                "updated_by": actor, "updated_at": _now_iso()})
        action = "restored"
    else:
        raise HTTPException(409, f"{email} is already a teacher ({_teacher_status(existing)}).")
    _table_role_emails("teachers", refresh=True)
    _log_role_change("teacher", email, action, actor)
    return {"status": action, "email": email}

@app.patch("/admin/teachers/{email}")
def set_teacher_status(email: str, update: TeacherStatusUpdate, user: dict = Depends(get_current_user)):
    """Activate or deactivate a teacher. Inactive teachers keep their record
    but lose teacher rights until reactivated."""
    _require_admin(user)
    _require_service_key()
    email = email.strip().lower()
    existing = next((r for r in _teacher_rows() if r["email"] == email), None)
    if existing is None or _teacher_status(existing) == "removed":
        raise HTTPException(404, f"{email} is not a current teacher. Restore them first.")
    if _teacher_status(existing) == update.status:
        return {"status": update.status, "email": email}
    actor = user.get("email")
    _update_teacher(email, {"status": update.status, "updated_by": actor, "updated_at": _now_iso()})
    _table_role_emails("teachers", refresh=True)
    _log_role_change("teacher", email, "activated" if update.status == "active" else "deactivated", actor)
    return {"status": update.status, "email": email}

@app.delete("/admin/teachers/{email}")
def remove_teacher(email: str, user: dict = Depends(get_current_user)):
    """Soft delete: the record stays (with who removed it and when) and can be restored."""
    _require_admin(user)
    _require_service_key()
    email = email.strip().lower()
    existing = next((r for r in _teacher_rows() if r["email"] == email), None)
    if existing is None or _teacher_status(existing) == "removed":
        raise HTTPException(404, f"{email} is not a current teacher.")
    actor = user.get("email")
    _update_teacher(email, {"deleted_at": _now_iso(), "deleted_by": actor, "status": "inactive",
                            "updated_by": actor, "updated_at": _now_iso()})
    _table_role_emails("teachers", refresh=True)
    _log_role_change("teacher", email, "removed", actor)
    return {"status": "removed", "email": email}

@app.get("/admin/role-log")
def role_log(limit: int = 100, user: dict = Depends(get_current_user)):
    """Who added, removed, activated or deactivated teachers and admins."""
    _require_admin(user)
    _require_service_key()
    limit = max(1, min(limit, 500))
    try:
        return supabase_admin.table("role_audit_log").select("*") \
            .order("created_at", desc=True).limit(limit).execute().data
    except Exception as e:
        if "role_audit_log" in str(e):
            raise HTTPException(500, "Database migration needed: create the role_audit_log table (see README).")
        raise

# ---------- Comments & Reactions ----------
@app.post("/comments")
def create_comment(
    entry: CommentEntry,
    user: dict = Depends(get_current_user),
    user_client: Client = Depends(get_user_client)
):
    data = {
        "rating_id": entry.rating_id,
        "user_id": user["id"],
        "parent_id": entry.parent_id,
        "body": entry.body
    }
    result = user_client.table("comments").insert(data).execute()
    return {"id": result.data[0]["id"]}

@app.get("/comments/{rating_id}")
def list_comments(rating_id: int, user=Depends(get_current_user)):
    comments_result = supabase.table("comments").select("*").eq("rating_id", rating_id).order("created_at", desc=False).execute()
    comments = comments_result.data
    if not comments:
        return []
    user_ids = list(set(c["user_id"] for c in comments if c.get("user_id")))
    comment_ids = [c["id"] for c in comments]
    user_profiles = {}
    if user_ids:
        try:
            profiles = supabase.rpc("get_user_profiles", {"user_ids": user_ids}).execute()
            for p in profiles.data:
                user_profiles[p["id"]] = {
                    "full_name": p.get("full_name") or "Anonymous",
                    "avatar_url": p.get("avatar_url")
                }
        except Exception as e:
            print(f"Failed to fetch user profiles: {e}")
    reactions_result = supabase.table("comment_reactions").select("*").in_("comment_id", comment_ids).execute()
    reactions = reactions_result.data
    comment_reactions_map = {}
    user_reactions_map = {}
    for r in reactions:
        cid = r["comment_id"]
        if cid not in comment_reactions_map:
            comment_reactions_map[cid] = {}
        rtype = r["reaction_type"]
        comment_reactions_map[cid][rtype] = comment_reactions_map[cid].get(rtype, 0) + 1
        if r["user_id"] == user["id"]:
            if cid not in user_reactions_map:
                user_reactions_map[cid] = []
            user_reactions_map[cid].append(rtype)
    enriched = []
    for c in comments:
        profile = user_profiles.get(c["user_id"], {})
        c["user_name"] = profile.get("full_name") or "Anonymous"
        c["user_avatar"] = profile.get("avatar_url")
        c["reactions"] = comment_reactions_map.get(c["id"], {})
        c["user_reactions"] = user_reactions_map.get(c["id"], [])
        enriched.append(c)
    return enriched

@app.post("/reactions")
def toggle_reaction(
    entry: ReactionEntry,
    user: dict = Depends(get_current_user),
    user_client: Client = Depends(get_user_client)
):
    existing = user_client.table("comment_reactions").select("*").eq("comment_id", entry.comment_id).eq("user_id", user["id"]).eq("reaction_type", entry.reaction_type).execute()
    if existing.data:
        user_client.table("comment_reactions").delete().eq("id", existing.data[0]["id"]).execute()
        return {"status": "removed"}
    else:
        user_client.table("comment_reactions").insert({
            "comment_id": entry.comment_id,
            "user_id": user["id"],
            "reaction_type": entry.reaction_type
        }).execute()
        return {"status": "added"}

# ---------- Ratings ----------
@app.post("/ratings")
def submit_rating(
    entry: RatingEntry,
    user: dict = Depends(get_current_user),
    user_client: Client = Depends(get_user_client)
):
    existing = user_client.table("ratings").select("*").eq("user_id", user["id"]).execute()
    if existing.data:
        result = user_client.table("ratings").update({
            "rating": entry.rating,
            "comment": entry.comment,
            "updated_at": "now()"
        }).eq("user_id", user["id"]).execute()
    else:
        result = user_client.table("ratings").insert({
            "user_id": user["id"],
            "rating": entry.rating,
            "comment": entry.comment
        }).execute()
    return {"id": result.data[0]["id"]}

@app.get("/ratings")
def list_ratings():
    result = supabase.table("ratings").select("*").order("created_at", desc=True).execute()
    ratings = result.data
    if not ratings:
        return []
    user_ids = list(set(r["user_id"] for r in ratings if r.get("user_id")))
    user_profiles = {}
    if user_ids:
        try:
            profiles = supabase.rpc("get_user_profiles", {"user_ids": user_ids}).execute()
            for p in profiles.data:
                user_profiles[p["id"]] = {
                    "full_name": p.get("full_name") or "Anonymous",
                    "avatar_url": p.get("avatar_url")
                }
        except Exception as e:
            print(f"Failed to fetch user profiles: {e}")
    for r in ratings:
        profile = user_profiles.get(r["user_id"], {})
        r["user_name"] = profile.get("full_name") or "Anonymous"
        r["user_avatar"] = profile.get("avatar_url")
    return ratings

@app.get("/ratings/summary")
def get_rating_summary():
    result = supabase.table("ratings").select("rating").execute()
    ratings = [r["rating"] for r in result.data]
    avg = sum(ratings) / len(ratings) if ratings else 0
    return {
        "average": round(avg, 1),
        "count": len(ratings),
        "distribution": {
            "1": ratings.count(1),
            "2": ratings.count(2),
            "3": ratings.count(3),
            "4": ratings.count(4),
            "5": ratings.count(5)
        }
    }

@app.get("/admin/user-profiles")
def admin_get_user_profiles(ids: str, user: dict = Depends(get_current_user)):
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    user_ids = [uid.strip() for uid in ids.split(",") if uid.strip()]
    if not user_ids:
        return {"profiles": []}
    try:
        result = supabase.rpc("get_user_profiles", {"user_ids": user_ids}).execute()
        return {"profiles": result.data}
    except Exception as e:
        raise HTTPException(500, str(e))

# ---------- Admin Surveys ----------
@app.post("/surveys")
def create_survey(
    survey: SurveyCreate,
    user: dict = Depends(get_current_user),
    user_client: Client = Depends(get_user_client)
):
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    data = survey.dict()
    try:
        if supabase_admin:
            result = supabase_admin.table("surveys").insert(data).execute()
        else:
            result = user_client.table("surveys").insert(data).execute()
    except Exception:
        result = user_client.table("surveys").insert(data).execute()
    return {"id": result.data[0]["id"]}

@app.get("/surveys")
def list_surveys(
    active_only: bool = False,
    user_client: Client = Depends(get_user_client)
):
    query = user_client.table("surveys").select("*")
    if active_only:
        query = query.eq("is_active", True)
    result = query.order("created_at", desc=True).execute()
    return result.data

@app.put("/surveys/{id}")
def update_survey(
    id: int,
    survey: SurveyUpdate,
    user: dict = Depends(get_current_user),
    user_client: Client = Depends(get_user_client)
):
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    data = {k: v for k, v in survey.dict().items() if v is not None}
    try:
        if supabase_admin:
            supabase_admin.table("surveys").update(data).eq("id", id).execute()
        else:
            user_client.table("surveys").update(data).eq("id", id).execute()
    except Exception:
        user_client.table("surveys").update(data).eq("id", id).execute()
    return {"status": "updated"}

@app.delete("/surveys/{id}")
def delete_survey(
    id: int,
    user: dict = Depends(get_current_user),
    user_client: Client = Depends(get_user_client)
):
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    try:
        if supabase_admin:
            supabase_admin.table("surveys").delete().eq("id", id).execute()
        else:
            user_client.table("surveys").delete().eq("id", id).execute()
    except Exception:
        user_client.table("surveys").delete().eq("id", id).execute()
    return {"status": "deleted"}

#region Admin Survey
@app.post("/surveys/{survey_id}/questions")
def add_question(
    survey_id: int,
    question: QuestionCreate,
    user: dict = Depends(get_current_user),
    user_client: Client = Depends(get_user_client)
):
    if not is_admin(user):
        raise HTTPException(403, "Admin only")

    data = question.dict()
    data["survey_id"] = survey_id
    db = _survey_admin_db()
    try:
        existing = db.table("survey_questions") \
            .select("order_number") \
            .eq("survey_id", survey_id) \
            .order("order_number", desc=True) \
            .limit(1) \
            .execute()
        max_order = existing.data[0]["order_number"] if existing.data else 0
        data["order_number"] = max_order + 1
        result = db.table("survey_questions").insert(data).execute()
    except Exception as e:
        _survey_db_error(e)
    return {"id": result.data[0]["id"]}

# Survey questions and files are written with the service key: the tables'
# row-level security doesn't let a user's own token change them. Errors are
# explained instead of surfacing as a bare "violates row-level security".
SURVEY_KEY_HELP = ("Set SUPABASE_SERVICE_KEY on the server to the service_role key "
                   "(Supabase → Project Settings → API) and restart the backend.")


def _survey_admin_db():
    if not supabase_admin:
        raise HTTPException(500, "Survey changes can't be saved: SUPABASE_SERVICE_KEY isn't set. " + SURVEY_KEY_HELP)
    return supabase_admin


def _survey_db_error(e):
    message = str(getattr(e, "message", None) or e)
    print(f"❌ survey write failed: {message}")
    if "42501" in message or "row-level security" in message:
        raise HTTPException(500, "The database refused the change (row-level security), so the server's "
                                 "SUPABASE_SERVICE_KEY is not the service_role key. " + SURVEY_KEY_HELP)
    raise HTTPException(500, message[:300])


class QuestionBulkItem(BaseModel):
    question: str
    question_type: str
    options: Optional[List[str]] = None
    is_required: bool = True


class QuestionBulk(BaseModel):
    questions: List[QuestionBulkItem]


@app.post("/surveys/{survey_id}/questions/bulk")
def add_questions_bulk(
    survey_id: int,
    payload: QuestionBulk,
    user: dict = Depends(get_current_user),
    user_client: Client = Depends(get_user_client)
):
    """Add many questions at once, in order (e.g. a questionnaire template)."""
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    if not payload.questions or len(payload.questions) > 300:
        raise HTTPException(400, "Send 1 to 300 questions")
    db = _survey_admin_db()
    try:
        existing = db.table("survey_questions") \
            .select("order_number") \
            .eq("survey_id", survey_id) \
            .order("order_number", desc=True) \
            .limit(1) \
            .execute()
        start = existing.data[0]["order_number"] if existing.data else 0
        rows = [{**q.dict(), "survey_id": survey_id, "order_number": start + i + 1}
                for i, q in enumerate(payload.questions)]
        result = db.table("survey_questions").insert(rows).execute()
    except HTTPException:
        raise
    except Exception as e:
        _survey_db_error(e)
    return {"count": len(rows), "ids": [r["id"] for r in (result.data or [])]}


@app.post("/surveys/import")
def import_survey_file(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """Read a questionnaire (.docx or PDF) into survey questions. Nothing is
    saved: the admin creates the survey from the result and edits it."""
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    data = file.file.read()
    if len(data) > 15 * 1024 * 1024:
        raise HTTPException(413, "The file is too large (max 15 MB).")
    from questionnaire_import import import_questionnaire
    try:
        return import_questionnaire(data, file.filename or "questionnaire.docx")
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        print(f"❌ questionnaire import failed: {e}")
        raise HTTPException(500, "The questionnaire couldn't be read. Check that it's a valid .docx or PDF.")


class QuestionOrder(BaseModel):
    ids: List[int]


@app.put("/surveys/{survey_id}/order")
def reorder_questions(survey_id: int, payload: QuestionOrder, user: dict = Depends(get_current_user)):
    """Set the questions' order (and so their tab) to the given id sequence."""
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    db = _survey_admin_db()
    try:
        for position, qid in enumerate(payload.ids, start=1):
            db.table("survey_questions").update({"order_number": position}) \
                .eq("id", qid).eq("survey_id", survey_id).execute()
    except Exception as e:
        _survey_db_error(e)
    return {"status": "reordered", "count": len(payload.ids)}


# ---------- Survey attachments (Supabase Storage, private bucket) ----------
SURVEY_BUCKET = "survey-files"
SURVEY_FILE_TYPES = {".pdf", ".doc", ".docx", ".xls", ".xlsx", ".csv", ".txt",
                     ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ppt", ".pptx"}
SURVEY_FILE_LIMIT = 20 * 1024 * 1024
_survey_bucket_ready = False


def _survey_storage():
    global _survey_bucket_ready
    db = _survey_admin_db()
    if not _survey_bucket_ready:
        try:
            db.storage.create_bucket(SURVEY_BUCKET, options={"public": False})
        except Exception as e:
            if "exist" not in str(e).lower() and "duplicate" not in str(e).lower():
                print(f"⚠️ could not create bucket {SURVEY_BUCKET}: {e}")
        _survey_bucket_ready = True
    return db.storage.from_(SURVEY_BUCKET)


@app.post("/surveys/{survey_id}/files")
def upload_survey_file(survey_id: int, file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """Admin attaches a file (e.g. an essay to review) to a survey."""
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    name = os.path.basename(file.filename or "file")
    ext = os.path.splitext(name)[1].lower()
    if ext not in SURVEY_FILE_TYPES:
        raise HTTPException(400, f"Files of type {ext or '(none)'} can't be attached. Use PDF, Word, Excel, PowerPoint, CSV, text or an image.")
    data = file.file.read()
    if len(data) > SURVEY_FILE_LIMIT:
        raise HTTPException(413, "Files can be at most 20 MB.")
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name)[-80:]
    path = f"survey-{survey_id}/{uuid.uuid4().hex[:12]}-{safe}"
    try:
        _survey_storage().upload(path, data, {"content-type": file.content_type or "application/octet-stream"})
    except HTTPException:
        raise
    except Exception as e:
        _survey_db_error(e)
    return {"path": path, "name": name, "type": file.content_type or "", "size": len(data)}


@app.get("/survey-files/url")
def survey_file_url(path: str, user: dict = Depends(get_current_user)):
    """A one-hour link to view or download a survey attachment (signed-in users)."""
    if not re.fullmatch(r"survey-\d+/[A-Za-z0-9._-]+", path or ""):
        raise HTTPException(400, "Invalid file path")
    try:
        signed = _survey_storage().create_signed_url(path, 3600)
    except HTTPException:
        raise
    except Exception as e:
        _survey_db_error(e)
    return {"url": signed.get("signedURL") or signed.get("signedUrl")}


@app.delete("/survey-files")
def delete_survey_file(path: str, user: dict = Depends(get_current_user)):
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    if not re.fullmatch(r"survey-\d+/[A-Za-z0-9._-]+", path or ""):
        raise HTTPException(400, "Invalid file path")
    try:
        _survey_storage().remove([path])
    except HTTPException:
        raise
    except Exception as e:
        _survey_db_error(e)
    return {"status": "deleted"}


@app.get("/surveys/{survey_id}/questions")
def list_questions(
    survey_id: int,
    user_client: Client = Depends(get_user_client)
):
    result = user_client.table("survey_questions") \
        .select("*") \
        .eq("survey_id", survey_id) \
        .order("order_number") \
        .execute()
    return result.data

@app.get("/surveys/{survey_id}/my-response")
def get_my_response(
    survey_id: int,
    user: dict = Depends(get_current_user),
    user_client: Client = Depends(get_user_client)
):
    resp = user_client.table("survey_responses") \
        .select("id") \
        .eq("survey_id", survey_id) \
        .eq("user_id", user["id"]) \
        .limit(1) \
        .execute()
    return {"already_submitted": len(resp.data) > 0}

@app.get("/surveys/{survey_id}/my-responses")
def get_my_responses(
    survey_id: int,
    user: dict = Depends(get_current_user),
    user_client: Client = Depends(get_user_client)
):
    answers = user_client.table("survey_responses") \
        .select("*") \
        .eq("survey_id", survey_id) \
        .eq("user_id", user["id"]) \
        .execute()

    if not answers.data:
        return {"responses": [], "questions": []}

    question_ids = [a["question_id"] for a in answers.data]
    questions = user_client.table("survey_questions") \
        .select("*") \
        .in_("id", question_ids) \
        .execute()

    q_map = {q["id"]: q for q in questions.data}

    detailed = []
    for a in answers.data:
        q = q_map.get(a["question_id"], {})
        detailed.append({
            "question": q.get("question", ""),
            "question_type": q.get("question_type", "text"),
            "options": q.get("options"),
            "order_number": q.get("order_number", 0),
            "answer": a["answer"],
            "created_at": a["created_at"]
        })
    detailed.sort(key=lambda d: d["order_number"] or 0)   # questionnaire order

    return {"responses": detailed}

@app.put("/questions/{id}")
def update_question(
    id: int,
    question: QuestionUpdate,
    user: dict = Depends(get_current_user),
    user_client: Client = Depends(get_user_client)
):
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    data = {k: v for k, v in question.dict().items() if v is not None}
    try:
        _survey_admin_db().table("survey_questions").update(data).eq("id", id).execute()
    except HTTPException:
        raise
    except Exception as e:
        _survey_db_error(e)
    return {"status": "updated"}

@app.delete("/questions/{id}")
def delete_question(
    id: int,
    user: dict = Depends(get_current_user),
    user_client: Client = Depends(get_user_client)
):
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    try:
        _survey_admin_db().table("survey_questions").delete().eq("id", id).execute()
    except HTTPException:
        raise
    except Exception as e:
        _survey_db_error(e)
    return {"status": "deleted"}

#region User Survey Responses
@app.post("/surveys/{survey_id}/respond")
def submit_survey_response(
    survey_id: int,
    payload: SurveyResponseSubmit,
    user: dict = Depends(get_current_user),
    user_client: Client = Depends(get_user_client)
):
    survey = user_client.table("surveys").select("id,is_active").eq("id", survey_id).eq("is_active", True).execute()
    if not survey.data:
        raise HTTPException(404, "Survey not found or inactive")
    for question_id_str, answer in payload.answers.items():
        user_client.table("survey_responses").upsert({
            "survey_id": survey_id,
            "question_id": int(question_id_str),
            "user_id": user["id"],
            "answer": answer
        }, on_conflict="survey_id,question_id,user_id").execute()
    return {"status": "submitted"}

@app.get("/surveys/{survey_id}/responses")
def get_survey_responses(
    survey_id: int,
    user: dict = Depends(get_current_user),
    user_client: Client = Depends(get_user_client)
):
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    if supabase_admin:
        try:
            responses = supabase_admin.table("survey_responses") \
                .select("*") \
                .eq("survey_id", survey_id) \
                .execute()
            return responses.data
        except Exception as e:
            print(f"supabase_admin failed: {e}")
    responses = user_client.table("survey_responses") \
        .select("*") \
        .eq("survey_id", survey_id) \
        .execute()
    return responses.data

#end region

#region Rubrics
# Two kinds of rubric share the `rubrics` table:
#   - school-wide (is_system = true): created and edited by admins only; the
#     active one is the default for everyone.
#   - personal (is_system = false): any signed-in teacher manages their own;
#     their active one is their default and overrides the school-wide rubric.
# Requires: alter table rubrics add column if not exists is_system boolean not null default false;

def _validate_criteria(criteria: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Normalize submitted criteria into the stored configurable shape."""
    if not criteria:
        raise HTTPException(422, "A rubric needs at least one criterion.")
    normalized = evaluator.normalize_rubric(criteria)
    if len(normalized) != len(criteria):
        raise HTTPException(422, "Every criterion must be an object.")
    for key in normalized:
        if not key.replace("_", "").isalnum():
            raise HTTPException(422, f"Invalid criterion key '{key}': use letters, numbers and underscores.")
    # JSON object keys must be strings.
    return {
        key: {**c, "levels": {str(lvl): text for lvl, text in c["levels"].items()}}
        for key, c in normalized.items()
    }

def _rubric_client(user_client: Client) -> Client:
    return supabase_admin or user_client

def _get_rubric_or_404(client: Client, rubric_id: int) -> dict:
    result = client.table("rubrics").select("*").eq("id", rubric_id).limit(1).execute()
    if not result.data:
        raise HTTPException(404, "Rubric not found")
    return result.data[0]

def _check_can_edit(user: dict, row: dict):
    if row.get("is_system"):
        if not is_admin(user):
            raise HTTPException(403, "Only admins can change school-wide rubrics.")
    else:
        if row.get("user_id") != user["id"]:
            raise HTTPException(403, "You can only change your own rubrics.")

def _deactivate_others(client: Client, row: dict):
    """Keep one active rubric per scope: school-wide, or per teacher."""
    query = (
        client.table("rubrics").update({"is_active": False})
        .neq("id", row["id"]).eq("is_active", True)
    )
    if row.get("is_system"):
        query = query.eq("is_system", True)
    else:
        query = query.eq("is_system", False).eq("user_id", row["user_id"])
    query.execute()

def _with_flags(row: dict, user: dict) -> dict:
    return {
        **row,
        "is_system": bool(row.get("is_system")),
        "can_edit": is_admin(user) if row.get("is_system") else row.get("user_id") == user["id"],
    }

@app.post("/rubrics/import")
def import_rubric_file(file: UploadFile = File(...), user: dict = Depends(get_current_user)):
    """
    Read criteria from an uploaded rubric (PDF, .xlsx or .csv). Nothing is
    saved: the editor shows them for the user to review, change and save.
    """
    data = file.file.read()
    if len(data) > 10 * 1024 * 1024:
        raise HTTPException(413, "The file is too large (10 MB maximum).")
    import rubric_import
    try:
        return rubric_import.import_rubric(data, file.filename or "")
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        print(f"❌ Rubric import failed: {e}")
        raise HTTPException(500, "The file couldn't be read. Is it a valid PDF or Excel file?")

@app.post("/rubrics")
def create_rubric(rubric: RubricCreate,
                  user: dict = Depends(get_current_user),
                  user_client: Client = Depends(get_user_client)):
    if rubric.is_system and not is_admin(user):
        raise HTTPException(403, "Only admins can create school-wide rubrics.")
    data = rubric.dict()
    data["criteria"] = _validate_criteria(rubric.criteria)
    data["user_id"] = user["id"]
    client = _rubric_client(user_client)
    result = client.table("rubrics").insert(data).execute()
    # Merge what was saved with what the database returned (id, timestamps).
    row = {**data, **result.data[0]}
    if rubric.is_active:
        _deactivate_others(client, row)
    return {"id": row["id"]}

@app.get("/rubrics")
def list_rubrics(user: dict = Depends(get_current_user),
                 user_client: Client = Depends(get_user_client)):
    """The caller's own rubrics plus school-wide ones (all of them for admins,
    only the active one for teachers)."""
    client = _rubric_client(user_client)
    try:
        own = (
            client.table("rubrics").select("*")
            .eq("user_id", user["id"]).eq("is_system", False)
            .order("created_at", desc=True).execute().data
        )
        system = client.table("rubrics").select("*").eq("is_system", True)
        if not is_admin(user):
            system = system.eq("is_active", True)
        system = system.order("created_at", desc=True).execute().data
    except Exception as e:
        if "is_system" in str(e):
            raise HTTPException(500, "Database migration needed: add the rubrics.is_system column (see README).")
        raise
    return [_with_flags(r, user) for r in system + own]

@app.put("/rubrics/{rubric_id}")
def update_rubric(rubric_id: int,
                  rubric: RubricUpdate,
                  user: dict = Depends(get_current_user),
                  user_client: Client = Depends(get_user_client)):
    client = _rubric_client(user_client)
    row = _get_rubric_or_404(client, rubric_id)
    _check_can_edit(user, row)
    data = {k: v for k, v in rubric.dict().items() if v is not None}
    data.pop("is_system", None)  # a rubric's scope is fixed at creation
    if "criteria" in data:
        data["criteria"] = _validate_criteria(rubric.criteria)
    if data:
        client.table("rubrics").update(data).eq("id", rubric_id).execute()
    if rubric.is_active:
        _deactivate_others(client, row)
    return {"status": "updated"}

@app.delete("/rubrics/{rubric_id}")
def delete_rubric(rubric_id: int,
                  user: dict = Depends(get_current_user),
                  user_client: Client = Depends(get_user_client)):
    client = _rubric_client(user_client)
    row = _get_rubric_or_404(client, rubric_id)
    _check_can_edit(user, row)
    client.table("rubrics").delete().eq("id", rubric_id).execute()
    return {"status": "deleted"}

@app.get("/rubric/active")
def get_active_rubric_meta(rubric_id: Optional[int] = None,
                           user: Optional[dict] = Depends(get_optional_user)):
    """The rubric an evaluation would use (the caller's default, or
    `rubric_id`), normalized for the frontend."""
    raw, info = resolve_rubric(user["id"] if user else None, rubric_id)
    criteria = evaluator.normalize_rubric(raw)
    return {
        "rubric": info,
        "criteria": [
            {**meta, "levels": {str(k): v for k, v in criteria[meta["key"]]["levels"].items()}}
            for meta in evaluator.rubric_meta(criteria)
        ],
    }

#end region

#region Saved Essays
@app.post("/saved-essays")
def save_essay(entry: SavedEssayEntry, user=Depends(get_current_user)):
    try:
        data = entry.dict()
        data["user_id"] = user["id"]
        print(f"[saved-essays] inserting user_id={user['id']} title={entry.title!r}")
        result = _db().table("saved_essays").insert(data).execute()
        print(f"[saved-essays] inserted id={result.data[0]['id']}")
        return {"id": result.data[0]["id"]}
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[saved-essays] FAILED: {e}")
        raise HTTPException(500, str(e))

@app.get("/saved-essays")
def list_saved_essays(user=Depends(get_current_user)):
    try:
        result = _db().table("saved_essays") \
            .select("id,title,essay,created_at") \
            .eq("user_id", user["id"]) \
            .order("created_at", desc=True) \
            .execute()
        return result.data
    except Exception as e:
        raise HTTPException(500, str(e))

@app.delete("/saved-essays/{id}")
def delete_saved_essay(id: int, user=Depends(get_current_user)):
    try:
        _db().table("saved_essays").delete().eq("id", id).eq("user_id", user["id"]).execute()
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(500, str(e))

#endregion

#region OCR
def _log_handwriting_sample(contents, suffix, content_type, reader_text, engine):
    """Store the page image and the reader's text for the OCR Labeler queue."""
    image_url = None
    try:
        import uuid as _uuid
        bucket = supabase.storage.from_("ocr-samples")
        filename = f"{_uuid.uuid4()}{suffix}"
        bucket.upload(filename, contents,
                      {"content-type": content_type or "application/octet-stream"})
        image_url = bucket.get_public_url(filename)
        print(f"[OCR] image uploaded: {image_url}")
    except Exception as e:
        print(f"Failed to upload OCR image: {e}")

    try:
        row = supabase.table("ocr_training_data").insert({
            "raw_ocr_text": reader_text,
            "image_url": image_url,
            "engine": engine,
            "accepted": False,
        }).execute()
        if row.data:
            print(f"[OCR] logged sample id={row.data[0]['id']}")
    except Exception as e:
        print(f"Failed to log OCR sample: {e}")


@app.post("/ocr")
async def ocr_from_file(request: Request, file: UploadFile = File(...),
                        user: Optional[dict] = Depends(get_optional_user)):
    if user is None:
        _check_guest_quota(request, "ocr")
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf', '.docx']:
        raise HTTPException(400, "Unsupported file type")

    contents = await file.read()

    # ---------- PDF: hand off to background job ----------
    if suffix == '.pdf':
        from ocr_jobs import start_pdf_ocr_job
        job_id = start_pdf_ocr_job(contents)
        return {"job_id": job_id, "status": "processing", "message": "OCR job submitted"}

    # ---------- DOCX: read directly, no OCR ----------
    if suffix == '.docx':
        try:
            text = ocr_utils.extract_docx_text(contents)
        except Exception as e:
            raise HTTPException(500, f"Failed to read DOCX: {e}")
        from text_quality import looks_like_english_text
        if not looks_like_english_text(text):
            _not_essay("The document doesn't contain English paragraphs.")

        try:
            supabase.table("ocr_training_data").insert({
                "raw_ocr_text": text,
                "accepted": False,
                "engine": "python-docx",
            }).execute()
        except Exception as e:
            print(f"Failed to log DOCX sample: {e}")

        return OCRResponse(
            text=text,
            confidence=100.0,
            method="DOCX text extraction",
            engine="python-docx",
            low_confidence=False,
            word_count=len(text.split()),
        )

    # ---------- Single-image path ----------
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name

    try:
        # 1-2. Check it's a page of writing, then read it with whichever
        # reader (printed or handwriting) produces more real English.
        result = ocr_utils.read_essay_image(contents)
        if not result["is_text"]:
            _not_essay(result["reason"])
        text, confidence, engine = result["text"], result["confidence"], result["engine"]
        print(f"[OCR] selected engine={engine}, len={len(text)}, conf={confidence:.1f}, "
              f"English={result['quality']:.0%}")
        # What the reader itself saw, one written line per line, before any AI
        # correction: this is what the OCR Labeler starts from, and the line
        # breaks must match the page for export_training_data.py.
        reader_text = "\n".join(result.get("lines") or []) or text

        # 3. AI correction — different prompts for printed vs handwriting.
        # Gemini already copies the page exactly; "correcting" it again could
        # change what the student wrote, so its text is used as is.
        if text and len(text.strip()) > 20 and not engine.startswith("gemini"):
            try:
                from evaluator import (
                    ai_correct_ocr_text,
                    ai_correct_handwriting_text,
                )
                if engine == "tesseract":
                    corrected = ai_correct_ocr_text(text)
                else:
                    corrected = ai_correct_handwriting_text(text)
                if corrected and corrected.strip():
                    text = corrected
            except Exception as e:
                print(f"Correction skipped: {e}")

        # 4. Upload image to Supabase Storage + log a handwriting sample for the
        # OCR Labeler. Printed pages (Tesseract) aren't useful there. Done in
        # the background so the student gets their text without waiting for it.
        if engine != "tesseract":
            threading.Thread(
                target=_log_handwriting_sample,
                args=(contents, suffix, file.content_type, reader_text, engine),
                daemon=True,
            ).start()

        return OCRResponse(
            text=text,
            confidence=confidence,
            method=f"Auto-selected OCR ({engine})",
            engine=engine,
            low_confidence=confidence < 60.0,
            word_count=len(text.split()),
        )
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

        
@app.get("/ocr/pending")
def list_pending_ocr(
    limit: int = 50,
    user: dict = Depends(get_current_user),
):
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    try:
        result = (
            _db().table("ocr_training_data")
            .select("id, image_url, raw_ocr_text, engine, created_at")
            .eq("accepted", False)
            .not_.is_("image_url", "null")
            .neq("engine", "tesseract")
            .order("id", desc=True)        # newest (best-read) samples first
            .limit(limit)
            .execute()
        )
        return result.data
    except Exception as e:
        print(f"Failed to fetch pending OCR samples: {e}")
        raise HTTPException(500, str(e))

@app.get("/ocr/accuracy")
def ocr_accuracy(user: dict = Depends(get_current_user)):
    """Each reader's measured accuracy on the OCR Labeler's verified pages."""
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    from ocr_accuracy import summarise
    rows, start = [], 0
    while True:  # PostgREST returns at most 1000 rows per request
        batch = (
            _db().table("ocr_training_data")
            .select("engine, raw_ocr_text, corrected_text")
            .eq("accepted", True)
            .not_.is_("corrected_text", "null")
            .order("id")
            .range(start, start + 999)
            .execute()
        ).data or []
        rows.extend(batch)
        if len(batch) < 1000:
            break
        start += 1000
    pending = (
        _db().table("ocr_training_data").select("id", count="exact")
        .eq("accepted", False).neq("engine", "tesseract").limit(1).execute()
    )
    return {**summarise(rows), "pending_pages": pending.count or 0}

@app.post("/ocr/correct")
def submit_ocr_correction(correction: OCRCorrection,
                          user=Depends(get_current_user)):
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    text = (correction.corrected_text or "").strip()
    if not text:
        raise HTTPException(422, "The transcription is empty")
    res = _db().table("ocr_training_data") \
        .update({"corrected_text": text, "accepted": True}) \
        .eq("id", correction.sample_id) \
        .execute()
    if not res.data:
        raise HTTPException(404, "Sample not found")
    return {"status": "saved"}

@app.delete("/ocr/sample/{sample_id}")
def discard_ocr_sample(sample_id: int, user=Depends(get_current_user)):
    """Remove a sample that can't be labeled (not an essay, unreadable photo)."""
    if not is_admin(user):
        raise HTTPException(403, "Admin only")
    _db().table("ocr_training_data").delete().eq("id", sample_id).execute()
    return {"status": "discarded"}

@app.get("/ocr/status/{job_id}")
def get_ocr_status(job_id: str):
    try:
        status = get_job_status(job_id)
        if not status:
            return JSONResponse(
                status_code=404,
                content={"status": "not_found", "error": "Job ID not found"}
            )
        if status['status'] == 'completed':
            conf = status.get('confidence') or 0.0
            return {
                "status": "completed",
                "text": status['result'],
                "confidence": conf,
                "engine": status.get('engine', 'unknown'),
                "low_confidence": conf < 60.0,
            }
        elif status['status'] == 'failed':
            return {"status": "failed", "error": status['error'], "code": status.get('code')}
        else:
            return {
                "status": "processing",
                "current_engine": status.get('current_engine')
            }
    except Exception as e:
        print(f"Status endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )

# ---------- Guest access ----------
# Visitors who aren't signed in can evaluate essays and read files (they just
# can't save anything). Both call paid / quota-limited AI services, so guests
# get a per-hour allowance per IP; signed-in users aren't limited. Counts are
# kept in memory (one backend process) and reset when it restarts.
GUEST_LIMITS = {
    # Each essay is evaluated by three AIs (three calls), so 15 = 5 essays.
    "evaluate": int(os.environ.get("GUEST_EVALS_PER_HOUR", "15")),
    "ocr": int(os.environ.get("GUEST_UPLOADS_PER_HOUR", "10")),
}
_guest_hits: Dict[str, List[float]] = {}
_guest_lock = threading.Lock()


def _client_ip(request: Request) -> str:
    # Behind a reverse proxy the real client is the last X-Forwarded-For entry
    # (the one the proxy added); earlier entries can be set by the client.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return request.client.host if request.client else "unknown"


def _check_guest_quota(request: Request, kind: str):
    limit = GUEST_LIMITS[kind]
    action = "run AI evaluations (3 per essay)" if kind == "evaluate" else "upload files"
    if limit <= 0:
        raise HTTPException(401, {"code": "sign_in_required",
                                  "message": f"Please sign in to {action}."})
    key, now = f"{kind}:{_client_ip(request)}", time.time()
    with _guest_lock:
        recent = [t for t in _guest_hits.get(key, []) if now - t < 3600]
        if len(recent) >= limit:
            wait = int((3600 - (now - recent[0])) // 60) + 1
            raise HTTPException(429, {"code": "guest_limit", "message": (
                f"Guests can {action} {limit} times an hour. Sign in (it's free) to keep "
                f"going, or try again in about {wait} min.")})
        recent.append(now)
        _guest_hits[key] = recent


def _run_evaluate(req: EvaluationRequest, user: Optional[dict], use_rag: bool):
    # Guests get the school-wide rubric (they have no rubrics of their own).
    criteria, rubric_info = resolve_rubric(user["id"] if user else None,
                                           req.rubric_id if user else None)
    result = run_evaluation(req.text, req.evaluation_type, use_rag=use_rag, rubric=criteria,
                            ai_model=req.ai_model, strict_model=req.strict_model)
    if result.get("error"):
        return EvaluationResponse(
            scores={"main_statement": 0, "organization": 0,
                    "evidence": 0, "analysis": 0, "grammar": 0,
                    "criterion_feedback": {}, "grammar_issues": [],
                    "recommendations": [], "rubric": rubric_info},
            feedback=result.get("feedback") or "Invalid input",
        )
    scores = {**result["scores"], "rubric": rubric_info}
    return EvaluationResponse(scores=scores, feedback=result["feedback"])


@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_essay(req: EvaluationRequest, request: Request,
                   user: Optional[dict] = Depends(get_optional_user)):
    if user is None:
        _check_guest_quota(request, "evaluate")
    try:
        return _run_evaluate(req, user, use_rag=False)
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        raise HTTPException(500, str(e))


@app.post("/evaluate-rag", response_model=EvaluationResponse)
def evaluate_essay_with_rag(req: EvaluationRequest, request: Request,
                            user: Optional[dict] = Depends(get_optional_user)):
    if user is None:
        _check_guest_quota(request, "evaluate")
    try:
        return _run_evaluate(req, user, use_rag=True)
    except Exception as e:
        print(f"❌ RAG Evaluation error: {e}")
        raise HTTPException(500, str(e))

def _insert_tolerant(table: str, row: Dict[str, Any], optional: List[str],
                     nest: Optional[Dict[str, str]] = None):
    """
    Insert a row, dropping optional columns the database doesn't have yet
    (see the README migrations) instead of failing the whole save.
    `nest` maps a dropped column to a jsonb column that keeps its value,
    e.g. {"teacher_scores": "original_scores"}.
    """
    row = dict(row)
    nest = nest or {}
    while True:
        try:
            return _db().table(table).insert(row).execute()
        except Exception as e:
            message = str(e)
            missing = [c for c in optional if c in row and f"'{c}'" in message]
            if not missing:
                raise
            print(f"⚠️ {table}: column(s) {missing} missing; saving without them "
                  f"(run the README migration)")
            for column in missing:
                value = row.pop(column)
                target = nest.get(column)
                if target and value is not None:
                    row[target] = {**(row.get(target) or {}), column: value}


@app.post("/revise-essay")
def revise_essay_endpoint(req: RevisionRequest, user: dict = Depends(get_current_user)):
    """AI-revised reference version of an essay, for the evaluation report.
    Teachers only, so students can't generate a finished essay to hand in."""
    _require_teacher(user, "generate a revised reference essay")
    if len(req.text.split()) < 20:
        raise HTTPException(422, "The essay is too short to revise.")
    result = evaluator.revise_essay(req.text, req.grammar_issues, req.recommendations, req.ai_model)
    if result is None:
        raise HTTPException(503, "No AI model could produce a revision right now. Try again in a moment.")
    return result

#region Knowledge Base
@app.post("/knowledge")
def save_knowledge(entry: KnowledgeEntry, user=Depends(get_current_user)):
    try:
        data = entry.dict()
        data["user_id"] = user["id"]
        # Approved entries feed RAG for everyone, so only a teacher's save is
        # approved; a student's save is kept privately as not approved.
        teacher = is_teacher(user)
        data["accepted"] = teacher
        data["approved_by"] = user.get("email") if teacher else None
        result = _insert_tolerant(
            "knowledge_base", data, optional=["scores", "ai_model", "approved_by"],
            nest={"scores": "criterion_feedback", "ai_model": "criterion_feedback",
                  "approved_by": "criterion_feedback"},
        )
        print(f"✅ Knowledge saved for user: {user['id']}")
        return {"id": result.data[0]["id"]}
    except Exception as e:
        print(f"❌ Error saving knowledge: {e}")
        raise HTTPException(500, f"Supabase error: {str(e)}")

@app.get("/knowledge")
def list_knowledge(limit: int = 50, user=Depends(get_current_user)):
    try:
        print(f"📚 Fetching knowledge for user: {user['id']}")
        result = _db().table("knowledge_base").select("*").eq("user_id", user["id"]).order("created_at", desc=True).limit(limit).execute()
        return result.data
    except Exception as e:
        print(f"❌ Error listing knowledge: {e}")
        raise HTTPException(500, str(e))

@app.get("/knowledge/{id}")
def get_knowledge(id: int, user=Depends(get_current_user)):
    try:
        result = _db().table("knowledge_base").select("*").eq("id", id).eq("user_id", user["id"]).execute()
        if result.data:
            return result.data[0]
        raise HTTPException(404, "Not found")
    except Exception as e:
        print(f"❌ Error getting knowledge {id}: {e}")
        raise HTTPException(500, str(e))
    

#region Teacher Override
@app.post("/override")
def save_override(override: OverrideRequest, user=Depends(get_current_user)):
    """
    Save an override: corrected scores with remarks (`teacher_feedback`,
    required) and suggestions (`suggested_changes`). Anyone signed in can
    override; only a teacher's override is marked approved (approved_by) and
    so feeds the AI's learning (RAG). Others are kept for the user only.
    """
    if not override.teacher_feedback.strip():
        raise HTTPException(422, "Remarks are required.")

    data = override.dict()
    data["user_id"] = user["id"]
    data["approved_by"] = user.get("email") if is_teacher(user) else None
    if not data.get("teacher_scores"):
        data.pop("teacher_scores", None)
    if not data.get("ai_model"):
        by = (data["original_scores"] or {}).get("evaluated_by") or {}
        data["ai_model"] = by.get("label")

    try:
        result = _insert_tolerant(
            "learning_feedback", data,
            optional=["teacher_scores", "title", "ai_model", "approved_by"],
            # Without these columns the values still survive inside the
            # original_scores jsonb, and the Learning Hub reads them there.
            nest={"teacher_scores": "original_scores", "title": "original_scores",
                  "ai_model": "original_scores", "approved_by": "original_scores"},
        )
    except Exception as e:
        print(f"❌ Error saving override: {e}")
        raise HTTPException(500, f"Supabase error: {str(e)}")

    print(f"✅ Override saved for user: {user['id']}")
    return {"id": result.data[0]["id"]}

@app.get("/learning-kb")
def list_learning_feedback(limit: int = 50, user=Depends(get_current_user)):
    try:
        print(f"🧠 Fetching learning feedback for user: {user['id']}")
        result = _db().table("learning_feedback").select("*").eq("user_id", user["id"]).order("created_at", desc=True).limit(limit).execute()
        return result.data
    except Exception as e:
        print(f"❌ Error listing learning feedback: {e}")
        raise HTTPException(500, str(e))

#endregion

#region AI Prompt Testing
@app.post("/test-prompt", response_model=PromptTestResponse)
def test_ai_prompt(req: PromptTestRequest):
    try:
        result = test_prompt(
            ai_provider=req.ai_provider,
            system_prompt=req.system_prompt,
            user_prompt=req.user_prompt,
            model=req.model,
            json_mode=req.json_mode,
            json_schema=req.json_schema,
        )
        return PromptTestResponse(result=result)
    except Exception as e:
        print(f"❌ Prompt test error: {e}")
        return PromptTestResponse(result={"text": f"Error: {str(e)}", "model": "error"})

#endregion

#region Utility
@app.get("/rubric")
def get_rubric():
    return evaluator.RUBRIC

@app.get("/suggestions")
def get_suggestions():
    return evaluator.SUGGESTION_GUIDE

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
#end region