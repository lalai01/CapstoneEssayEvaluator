import os
import tempfile
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import ocr_utils
import evaluator
from supabase_client import supabase
from ai_models import test_prompt
from ocr_jobs import get_job_status

# ---------- Google Cloud Vision Credentials (for Render) ----------
google_creds_json = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS_JSON')
if google_creds_json:
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write(google_creds_json)
            creds_file = f.name
        os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_file
        print(f"✅ Google Cloud credentials loaded from environment variable. Temp file: {creds_file}")
    except Exception as e:
        print(f"❌ Failed to set Google credentials: {e}")
else:
    print("⚠️ GOOGLE_APPLICATION_CREDENTIALS_JSON not set. Vision API will use default credentials if available.")

# ---------- FastAPI App ----------
app = FastAPI(title="AI Essay Evaluator API")

# CORS - allow all origins for production (you can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# ---------- Request/Response Models ----------
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

class KnowledgeEntry(BaseModel):
    essay: str
    grammar: int
    coherence: int
    content: int
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

class PromptTestRequest(BaseModel):
    ai_provider: str   # "openai", "deepseek", "gemma"
    system_prompt: str
    user_prompt: str
    model: Optional[str] = None

class PromptTestResponse(BaseModel):
    result: Dict[str, Any]

# ---------- Health Check ----------
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ---------- OCR Endpoint (supports both sync and async) ----------
# No response_model so we can return either OCRResponse or async job dict
@app.post("/ocr")
async def ocr_from_file(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.pdf']:
        raise HTTPException(400, "Unsupported file type")
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    try:
        if suffix == '.pdf':
            result = ocr_utils.extract_text_from_pdf(tmp_path)
            # If result is a tuple (text, page_count) -> synchronous
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[0], str):
                text, page_count = result
                confidence = 75.0
                method = "Tesseract OCR (PDF)"
                return OCRResponse(text=text, confidence=confidence, method=method, page_count=page_count)
            else:
                # Async job – result is job_id
                job_id = result
                return {"job_id": job_id, "status": "processing", "message": "OCR job submitted. Poll /ocr/status/{job_id}"}
        else:
            text, confidence = ocr_utils.extract_text_from_image(tmp_path)
            method = "Auto-selected OCR"
            return OCRResponse(text=text, confidence=confidence, method=method)
    except Exception as e:
        print(f"OCR error: {e}")
        raise HTTPException(500, str(e))
    finally:
        os.unlink(tmp_path)

@app.get("/ocr/status/{job_id}")
def get_ocr_status(job_id: str):
    """Poll for async PDF OCR job status."""
    status = get_job_status(job_id)
    if not status:
        raise HTTPException(404, "Job not found")
    if status['status'] == 'completed':
        return {"status": "completed", "text": status['result'], "confidence": 90.0}
    elif status['status'] == 'failed':
        return {"status": "failed", "error": status['error']}
    else:
        return {"status": "processing"}

# ---------- Evaluation Endpoints ----------
@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_essay(req: EvaluationRequest):
    try:
        # Default: RAG is OFF for this endpoint (to keep original behaviour)
        scores, feedback = evaluator.evaluate_essay(req.text, req.evaluation_type, use_rag=False)
        return EvaluationResponse(scores=scores, feedback=feedback)
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/evaluate-rag", response_model=EvaluationResponse)
def evaluate_essay_with_rag(req: EvaluationRequest):
    try:
        scores, feedback = evaluator.evaluate_essay(req.text, req.evaluation_type, use_rag=True)
        return EvaluationResponse(scores=scores, feedback=feedback)
    except Exception as e:
        raise HTTPException(500, str(e))

# ---------- Knowledge Base (Supabase) ----------
@app.post("/knowledge")
def save_knowledge(entry: KnowledgeEntry):
    try:
        data = entry.dict()
        result = supabase.table("knowledge_base").insert(data).execute()
        return {"id": result.data[0]["id"]}
    except Exception as e:
        raise HTTPException(500, f"Supabase error: {str(e)}")

@app.get("/knowledge")
def list_knowledge(limit: int = 50):
    try:
        result = supabase.table("knowledge_base").select("*").order("created_at", desc=True).limit(limit).execute()
        return result.data
    except Exception as e:
        raise HTTPException(500, str(e))

@app.get("/knowledge/{id}")
def get_knowledge(id: int):
    result = supabase.table("knowledge_base").select("*").eq("id", id).execute()
    if result.data:
        return result.data[0]
    raise HTTPException(404, "Not found")

# ---------- Teacher Override & Learning KB ----------
@app.post("/override")
def save_override(override: OverrideRequest):
    try:
        data = override.dict()
        result = supabase.table("learning_feedback").insert(data).execute()
        return {"id": result.data[0]["id"]}
    except Exception as e:
        raise HTTPException(500, f"Supabase error: {str(e)}")

@app.get("/learning-kb")
def list_learning_feedback(limit: int = 50):
    try:
        result = supabase.table("learning_feedback").select("*").order("created_at", desc=True).limit(limit).execute()
        return result.data
    except Exception as e:
        raise HTTPException(500, str(e))

# ---------- AI Prompt Testing (Multi‑AI) ----------
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
        raise HTTPException(500, f"AI test failed: {str(e)}")

# ---------- Utility Endpoints ----------
@app.get("/rubric")
def get_rubric():
    return evaluator.RUBRIC

@app.get("/suggestions")
def get_suggestions():
    return evaluator.SUGGESTION_GUIDE

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)