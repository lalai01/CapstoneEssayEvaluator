import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import ocr_utils
import evaluator
from supabase_client import supabase, supabase_admin
from ai_models import test_prompt
from ocr_jobs import get_job_status
from auth import get_current_user, security
from supabase import create_client, Client

# ---------- Google Cloud Vision Credentials ----------
google_creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
if google_creds_json:
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(google_creds_json)
            creds_file = f.name
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_file
        print("✅ Google Cloud credentials loaded.")
    except Exception as e:
        print(f"❌ Failed to set Google credentials: {e}")
else:
    print("⚠️ GOOGLE_APPLICATION_CREDENTIALS_JSON not set.")

# ---------- FastAPI App ----------
app = FastAPI(title="AI Essay Evaluator API")

# ---------- CORS Configuration ----------
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://capstoneessayevaluator.web.app",
    "https://essay-evaluator.duckdns.org",
    "https://api.essay-evaluator.duckdns.org",
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
# (All models unchanged – included for completeness)
class EvaluationRequest(BaseModel):
    text: str
    evaluation_type: str = "analytic"

class EvaluationResponse(BaseModel):
    scores: dict
    feedback: str

class OCRResponse(BaseModel):
    text: str
    confidence: float
    method: str
    page_count: Optional[int] = None
    engine: Optional[str] = None

class KnowledgeEntry(BaseModel):
    title: Optional[str] = None
    essay: str
    grammar: Optional[int] = None
    coherence: Optional[int] = None
    content: Optional[int] = None
    feedback: str
    eval_type: str
    accepted: bool = False
    satisfaction: int = 5
    teacher_feedback: Optional[str] = None

class OverrideRequest(BaseModel):
    original_essay: str
    original_scores: Dict[str, int]
    teacher_feedback: str
    suggested_changes: Optional[str] = None
    accepted: bool
    satisfaction: Optional[int] = 5

class PromptTestRequest(BaseModel):
    ai_provider: str
    system_prompt: str
    user_prompt: str
    model: Optional[str] = None

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

# ---------- Health ----------
@app.get("/health")
def health_check():
    return {"status": "ok"}

def is_admin(user):
    admin_emails = os.environ.get("ADMIN_EMAILS", "admin_essay_capstone@gmail.com").split(",")
    return (user.get("role") == "admin") or (user.get("email") in admin_emails)

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

# ---------- Admin Surveys (🔥 fixed with user_client) ----------
@app.post("/surveys")
def create_survey(
    survey: SurveyCreate,
    user: dict = Depends(get_current_user),
    user_client: Client = Depends(get_user_client)   # kept for fallback
):
    if not is_admin(user):
        raise HTTPException(403, "Admin only")

    data = survey.dict()
    print(f"🔍 Admin user: {user['email']}, role: {user.get('role')}")
    print(f"📦 Creating survey: {data['title']}")

    # Prefer supabase_admin (bypasses RLS) – most reliable
    try:
        if supabase_admin:
            result = supabase_admin.table("surveys").insert(data).execute()
            return {"id": result.data[0]["id"]}
    except Exception as e:
        print(f"⚠️ supabase_admin failed: {e}, falling back to user_client")

    # Fallback to authenticated client
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

# ---------- Admin Questions (🔥 fixed with user_client) ----------
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
    try:
        if supabase_admin:
            result = supabase_admin.table("survey_questions").insert(data).execute()
        else:
            result = user_client.table("survey_questions").insert(data).execute()
    except Exception:
        result = user_client.table("survey_questions").insert(data).execute()
    return {"id": result.data[0]["id"]}

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
        if supabase_admin:
            supabase_admin.table("survey_questions").update(data).eq("id", id).execute()
        else:
            user_client.table("survey_questions").update(data).eq("id", id).execute()
    except Exception:
        user_client.table("survey_questions").update(data).eq("id", id).execute()
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
        if supabase_admin:
            supabase_admin.table("survey_questions").delete().eq("id", id).execute()
        else:
            user_client.table("survey_questions").delete().eq("id", id).execute()
    except Exception:
        user_client.table("survey_questions").delete().eq("id", id).execute()
    return {"status": "deleted"}

# ---------- User Survey Responses ----------
@app.post("/surveys/{survey_id}/respond")
def submit_survey_response(
    survey_id: int,
    payload: SurveyResponseSubmit,
    user: dict = Depends(get_current_user),
    user_client: Client = Depends(get_user_client)
):
    # Verify survey is active
    survey = supabase.table("surveys").select("id,is_active").eq("id", survey_id).eq("is_active", True).execute()
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
    try:
        if supabase_admin:
            responses = supabase_admin.table("survey_responses").select("*").eq("survey_id", survey_id).execute()
        else:
            responses = user_client.table("survey_responses").select("*").eq("survey_id", survey_id).execute()
    except Exception:
        responses = user_client.table("survey_responses").select("*").eq("survey_id", survey_id).execute()
    return responses.data

# ---------- Saved Essays ----------
@app.post("/saved-essays")
def save_essay(entry: SavedEssayEntry, user=Depends(get_current_user)):
    try:
        data = entry.dict()
        data["user_id"] = user["id"]
        result = supabase.table("saved_essays").insert(data).execute()
        return {"id": result.data[0]["id"]}
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/saved-essays")
def list_saved_essays(user=Depends(get_current_user)):
    try:
        result = supabase.table("saved_essays") \
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
        supabase.table("saved_essays").delete().eq("id", id).eq("user_id", user["id"]).execute()
        return {"status": "deleted"}
    except Exception as e:
        raise HTTPException(500, str(e))

# ---------- OCR ----------
@app.post("/ocr")
async def ocr_from_file(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf']:
        raise HTTPException(400, "Unsupported file type")
    contents = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    try:
        if suffix == '.pdf':
            from ocr_jobs import start_pdf_ocr_job
            job_id = start_pdf_ocr_job(contents)
            return {"job_id": job_id, "status": "processing", "message": "OCR job submitted"}
        else:
            text, confidence, engine = ocr_utils.extract_text_from_image(tmp_path)
            method = f"Auto-selected OCR ({engine})"
            return OCRResponse(text=text, confidence=confidence, method=method, engine=engine)
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        if suffix != '.pdf':
            os.unlink(tmp_path)

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
            return {
                "status": "completed",
                "text": status['result'],
                "confidence": 90.0,
                "engine": status.get('engine', 'unknown')
            }
        elif status['status'] == 'failed':
            return {"status": "failed", "error": status['error']}
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

# ---------- Evaluation ----------
@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_essay(req: EvaluationRequest):
    try:
        scores, feedback = evaluator.evaluate_essay(req.text, req.evaluation_type, use_rag=False)
        return EvaluationResponse(scores=scores, feedback=feedback)
    except Exception as e:
        print(f"❌ Evaluation error: {e}")
        raise HTTPException(500, str(e))

@app.post("/evaluate-rag", response_model=EvaluationResponse)
def evaluate_essay_with_rag(req: EvaluationRequest):
    try:
        scores, feedback = evaluator.evaluate_essay(req.text, req.evaluation_type, use_rag=True)
        return EvaluationResponse(scores=scores, feedback=feedback)
    except Exception as e:
        print(f"❌ RAG Evaluation error: {e}")
        raise HTTPException(500, str(e))

# ---------- Knowledge Base ----------
@app.post("/knowledge")
def save_knowledge(entry: KnowledgeEntry, user=Depends(get_current_user)):
    try:
        data = entry.dict()
        data["user_id"] = user["id"]
        result = supabase.table("knowledge_base").insert(data).execute()
        print(f"✅ Knowledge saved for user: {user['id']}")
        return {"id": result.data[0]["id"]}
    except Exception as e:
        print(f"❌ Error saving knowledge: {e}")
        raise HTTPException(500, f"Supabase error: {str(e)}")

@app.get("/knowledge")
def list_knowledge(limit: int = 50, user=Depends(get_current_user)):
    try:
        print(f"📚 Fetching knowledge for user: {user['id']}")
        result = supabase.table("knowledge_base").select("*").eq("user_id", user["id"]).order("created_at", desc=True).limit(limit).execute()
        return result.data
    except Exception as e:
        print(f"❌ Error listing knowledge: {e}")
        raise HTTPException(500, str(e))

@app.get("/knowledge/{id}")
def get_knowledge(id: int, user=Depends(get_current_user)):
    try:
        result = supabase.table("knowledge_base").select("*").eq("id", id).eq("user_id", user["id"]).execute()
        if result.data:
            return result.data[0]
        raise HTTPException(404, "Not found")
    except Exception as e:
        print(f"❌ Error getting knowledge {id}: {e}")
        raise HTTPException(500, str(e))

# ---------- Teacher Override ----------
@app.post("/override")
def save_override(override: OverrideRequest, user=Depends(get_current_user)):
    try:
        data = override.dict()
        data["user_id"] = user["id"]
        result = supabase.table("learning_feedback").insert(data).execute()
        print(f"✅ Override saved for user: {user['id']}")
        return {"id": result.data[0]["id"]}
    except Exception as e:
        print(f"❌ Error saving override: {e}")
        raise HTTPException(500, f"Supabase error: {str(e)}")

@app.get("/learning-kb")
def list_learning_feedback(limit: int = 50, user=Depends(get_current_user)):
    try:
        print(f"🧠 Fetching learning feedback for user: {user['id']}")
        result = supabase.table("learning_feedback").select("*").eq("user_id", user["id"]).order("created_at", desc=True).limit(limit).execute()
        return result.data
    except Exception as e:
        print(f"❌ Error listing learning feedback: {e}")
        raise HTTPException(500, str(e))

# ---------- AI Prompt Testing ----------
@app.post("/test-prompt", response_model=PromptTestResponse)
def test_ai_prompt(req: PromptTestRequest):
    try:
        result = test_prompt(
            ai_provider=req.ai_provider,
            system_prompt=req.system_prompt,
            user_prompt=req.user_prompt,
            model=req.model
        )
        return PromptTestResponse(result=result)
    except Exception as e:
        print(f"❌ Prompt test error: {e}")
        return PromptTestResponse(result={"text": f"Error: {str(e)}", "model": "error"})

# ---------- Utility ----------
@app.get("/rubric")
def get_rubric():
    return evaluator.RUBRIC

@app.get("/suggestions")
def get_suggestions():
    return evaluator.SUGGESTION_GUIDE

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)